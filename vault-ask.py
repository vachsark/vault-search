#!/usr/bin/env python3
"""
vault-ask.py — Ask natural language questions about the vault, grounded in real content.

Combines vault-search (semantic + BM25 hybrid retrieval) with local LLM generation
to answer questions with source citations. Runs entirely locally — zero API cost.

Usage:
    python3 vault-ask.py "How does the heartbeat system work?"
    python3 vault-ask.py "What are the rules for Convex mutations?" --model qwen2.5-coder:14b
    python3 vault-ask.py "What MCP servers are configured?" --top 10
    python3 vault-ask.py "Linesheet schema for products" --path Projects/Linesheet
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

# Import search function from vault-search
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

# We can't import vault-search.py directly (hyphen), so load the module dynamically
import importlib.util
_search_spec = importlib.util.spec_from_file_location(
    "vault_search", Path(__file__).resolve().parent / "vault-search.py"
)
_search_mod = importlib.util.module_from_spec(_search_spec)
_search_spec.loader.exec_module(_search_mod)
search = _search_mod.search
db_path_for_root = _search_mod.db_path_for_root

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VAULT_ROOT = Path(__file__).resolve().parent.parent
VAULT_DB = db_path_for_root(VAULT_ROOT)
OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:9b"
CODE_MODEL = "qwen2.5-coder:14b"
MAX_CONTEXT_CHARS = 12000  # How much retrieved content to include in prompt
MAX_FILE_CHARS = 3000      # Max chars per retrieved file


def read_file_content(rel_path: str, max_chars: int = MAX_FILE_CHARS) -> str:
    """Read actual file content from the vault."""
    full_path = VAULT_ROOT / rel_path
    try:
        text = full_path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... [truncated, {len(text)} chars total]"
        return text
    except Exception:
        return "(file not readable)"


def build_context(results: list[tuple], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Build a context string from search results by reading actual file content."""
    context_parts = []
    total_chars = 0

    for result in results:
        score, path = result[0], result[1]
        content = read_file_content(path)
        entry = f"### {path}\n{content}\n"

        if total_chars + len(entry) > max_chars:
            # Try a shorter version with just the beginning
            remaining = max_chars - total_chars - len(f"### {path}\n\n")
            if remaining > 200:
                entry = f"### {path}\n{content[:remaining]}\n... [truncated]\n"
            else:
                break

        context_parts.append(entry)
        total_chars += len(entry)

    return "\n".join(context_parts)


def generate_answer(question: str, context: str, model: str) -> str:
    """Generate an answer using the local LLM with retrieved context."""
    prompt = f"""You are a helpful assistant answering questions about an Obsidian vault / codebase.
Use ONLY the provided context to answer. If the context doesn't contain enough information, say so.
Be specific and cite file paths when referencing information.

## Retrieved Context

{context}

## Question

{question}

## Answer"""

    # num_ctx prevents Vulkan backend hangs when prompt exceeds default 4096
    # qwen3.5:9b and qwen2.5-coder:14b support 16K+ context; qwen3:8b kept for backward compat (deprecated 2026-03-20)
    num_ctx = 16384 if model in ("qwen3.5:9b", "qwen3:8b", "qwen2.5-coder:14b") else 8192
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 500,
            "num_ctx": num_ctx,
        },
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:  # 180s for cold start: VRAM reload + search + generate
        data = json.loads(resp.read())
    return data["response"].strip()


def ask(
    question: str,
    top_k: int = 7,
    path_filter: str | None = None,
    model: str | None = None,
    show_sources: bool = True,
) -> str:
    """Full RAG pipeline: retrieve → read → generate."""
    # Auto-select model: code model for code questions, general for everything else
    if model is None:
        code_signals = {"code", "function", "schema", "mutation", "query", "component",
                       "import", "export", "class", "def ", "convex", "react", "typescript",
                       "api", "endpoint", "handler", "middleware", "hook"}
        q_lower = question.lower()
        if any(signal in q_lower for signal in code_signals):
            model = CODE_MODEL
        else:
            model = DEFAULT_MODEL

    # Step 1: Retrieve relevant files
    results = search(question, db_path=VAULT_DB, top_k=top_k, path_filter=path_filter, mode="hybrid")

    if not results:
        return "No relevant files found in the vault index. Try running vault-index.py first."

    # Step 2: Read actual file content and build context
    context = build_context(results)

    # Step 3: Generate grounded answer
    answer = generate_answer(question, context, model)

    # Step 4: Append sources
    if show_sources:
        sources = "\n".join(f"  - {r[1]}" for r in results[:5])
        answer += f"\n\n**Sources:**\n{sources}"

    return answer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions about the vault using local RAG"
    )
    parser.add_argument("question", type=str, help="Your question")
    parser.add_argument(
        "--top", type=int, default=7,
        help="Number of files to retrieve (default: 7)"
    )
    parser.add_argument(
        "--path", type=str, default=None,
        help="Filter retrieval to files under this path prefix"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"Override generation model (default: auto-select between {DEFAULT_MODEL} and {CODE_MODEL})"
    )
    parser.add_argument(
        "--no-sources", action="store_true",
        help="Don't show source file list"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON (answer + sources + metadata)"
    )
    args = parser.parse_args()

    t0 = time.time()
    answer = ask(
        args.question,
        top_k=args.top,
        path_filter=args.path,
        model=args.model,
        show_sources=not args.no_sources,
    )
    elapsed = time.time() - t0

    if args.json:
        print(json.dumps({
            "question": args.question,
            "answer": answer,
            "elapsed_s": round(elapsed, 1),
            "model": args.model or "auto",
        }, indent=2))
    else:
        print(answer)
        print(f"\n({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
