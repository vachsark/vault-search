#!/usr/bin/env python3
"""
concept-to-code.py — Bridge between vault knowledge and project code.

Given a concept (e.g., "rate limiting", "auth middleware"), finds:
1. Knowledge notes about the concept (via vault-search)
2. Code implementations in indexed projects (via vault-search semantic + grep fallback)
3. The connection between them

This closes the gap between "what the vault knows" and "where it lives in code."

Usage:
    python3 concept-to-code.py "rate limiting"
    python3 concept-to-code.py "authentication middleware" --project Linesheet
    python3 concept-to-code.py "reward shaping" --knowledge-only
    python3 concept-to-code.py "throttling" --explain
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

VAULT_DIR = os.environ.get("VAULT_DIR", "/home/veech/Documents/TestVault")
SCRIPTS_DIR = os.path.join(VAULT_DIR, "_scripts")

# File extensions considered "code" (not docs/markdown)
CODE_EXTENSIONS = {
    ".ts", ".tsx", ".js", ".jsx", ".py", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".cs", ".swift", ".kt", ".rb", ".sh"
}
# Extensions to exclude even if returned by semantic search
DOC_EXTENSIONS = {".md", ".mdx", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml"}


def _is_code_file(path: str) -> bool:
    """Return True if the path looks like a code file (not docs/config)."""
    ext = Path(path).suffix.lower()
    return ext in CODE_EXTENSIONS


def _is_doc_file(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in DOC_EXTENSIONS


def _parse_vault_search_output(output: str, path_prefix: str = "Projects/") -> list[dict]:
    """Parse vault-search.py stdout into a list of result dicts.

    Output format (one result block):
        0.0307  Projects/Linesheet/.../middleware.ts
                * Next.js Middleware with Clerk Authentication
    """
    entries = []
    lines = output.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Score line: starts with a float, followed by a path
        if line and line[0].isdigit():
            parts = line.split(None, 1)
            if len(parts) == 2:
                try:
                    score = float(parts[0])
                    path = parts[1].strip()
                    # Collect optional section/title lines that follow
                    summary_parts = []
                    j = i + 1
                    while j < len(lines) and lines[j] and not (lines[j].strip() and lines[j].strip()[0].isdigit()):
                        stripped = lines[j].strip()
                        if stripped:
                            summary_parts.append(stripped.lstrip("*§ ").strip())
                        j += 1
                    summary = " — ".join(summary_parts) if summary_parts else ""
                    entries.append({"score": score, "path": path, "summary": summary})
                    i = j
                    continue
                except (ValueError, IndexError):
                    pass
        i += 1
    return entries


def search_knowledge(query: str, top: int = 5) -> list[dict]:
    """Search vault knowledge notes for a concept."""
    result = subprocess.run(
        ["python3", os.path.join(SCRIPTS_DIR, "vault-search.py"),
         query, "--top", str(top), "--no-graph", "--path", "Knowledge/"],
        capture_output=True, text=True, timeout=30, cwd=VAULT_DIR
    )

    entries = _parse_vault_search_output(result.stdout, path_prefix="Knowledge/")
    return [e for e in entries if "Knowledge/" in e["path"]]


def search_code_semantic(query: str, top: int = 10, project_filter: str = "") -> list[dict]:
    """Search code files using vault-search semantic index.

    Returns results filtered to code files only (excludes .md, .txt, etc.).
    When project_filter is set, only includes results from that project.
    """
    path_arg = f"Projects/{project_filter}" if project_filter else "Projects/"

    result = subprocess.run(
        ["python3", os.path.join(SCRIPTS_DIR, "vault-search.py"),
         query, "--top", str(top * 3), "--no-graph", "--path", path_arg],
        capture_output=True, text=True, timeout=30, cwd=VAULT_DIR
    )

    entries = _parse_vault_search_output(result.stdout)

    # Filter to code files only; skip worktrees to reduce duplication
    code_results = []
    for e in entries:
        path = e["path"]
        if not _is_code_file(path):
            continue
        if "worktrees" in path or "node_modules" in path or ".next" in path:
            continue
        project = _extract_project_name(path)
        code_results.append({
            "project": project,
            "file": path,
            "score": e["score"],
            "summary": e["summary"],
            "source": "semantic",
        })

    # Deduplicate by file path
    seen = set()
    unique = []
    for r in code_results:
        if r["file"] not in seen:
            seen.add(r["file"])
            unique.append(r)

    return unique[:top]


def _extract_project_name(path: str) -> str:
    """Extract project name from a Projects/Name/... path."""
    parts = path.split("/")
    try:
        idx = parts.index("Projects")
        return parts[idx + 1] if idx + 1 < len(parts) else "Unknown"
    except ValueError:
        return "Unknown"


def search_code_grep(query: str, project_filter: str = "") -> list[dict]:
    """Grep-based keyword fallback for code not in the semantic index."""
    project_dirs = []
    projects_root = os.path.join(VAULT_DIR, "Projects")

    if project_filter:
        for d in os.listdir(projects_root):
            if project_filter.lower() in d.lower():
                full = os.path.join(projects_root, d)
                if os.path.isdir(full):
                    for sd in os.listdir(full):
                        if sd.endswith("-app") and os.path.isdir(os.path.join(full, sd)):
                            project_dirs.append((d, os.path.join(full, sd)))
                            break
                    else:
                        project_dirs.append((d, full))
    else:
        for d in os.listdir(projects_root):
            full = os.path.join(projects_root, d)
            if os.path.isdir(full):
                for sd in os.listdir(full):
                    if sd.endswith("-app") and os.path.isdir(os.path.join(full, sd)):
                        project_dirs.append((d, os.path.join(full, sd)))

    terms = query.lower().split()
    results = []

    for proj_name, proj_dir in project_dirs:
        for ext in ["ts", "tsx", "js", "py"]:
            try:
                grep_result = subprocess.run(
                    ["grep", "-rlni", "--include", f"*.{ext}",
                     "--exclude-dir=node_modules", "--exclude-dir=.next",
                     "--exclude-dir=dist", "--exclude-dir=.convex",
                     "--exclude-dir=worktrees",
                     terms[0], proj_dir],
                    capture_output=True, text=True, timeout=10
                )
                for fpath in grep_result.stdout.strip().split("\n"):
                    if fpath:
                        try:
                            content = open(fpath, errors="replace").read().lower()
                            matches = sum(1 for t in terms if t in content)
                            if matches >= max(1, len(terms) // 2):
                                rel_path = os.path.relpath(fpath, VAULT_DIR)
                                results.append({
                                    "project": proj_name,
                                    "file": rel_path,
                                    "score": matches / len(terms),
                                    "summary": "",
                                    "source": "grep",
                                    "match_ratio": matches / len(terms),
                                })
                        except Exception:
                            pass
            except subprocess.TimeoutExpired:
                pass

    results.sort(key=lambda r: r["score"], reverse=True)
    seen = set()
    unique = []
    for r in results:
        if r["file"] not in seen:
            seen.add(r["file"])
            unique.append(r)

    return unique[:10]


def search_code(query: str, project_filter: str = "", explain: bool = False) -> dict:
    """Combined semantic + grep code search.

    Returns:
        {
            "semantic": [...],   # vault-search results (code files only)
            "grep": [...],       # grep results not already in semantic results
            "all": [...],        # merged, deduplicated, scored
        }
    """
    semantic = search_code_semantic(query, top=10, project_filter=project_filter)
    grep = search_code_grep(query, project_filter=project_filter)

    # Merge: grep results not already covered by semantic
    semantic_files = {r["file"] for r in semantic}
    grep_only = [r for r in grep if r["file"] not in semantic_files]

    merged = semantic + grep_only
    return {
        "semantic": semantic,
        "grep": grep_only,
        "all": merged,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Bridge between vault knowledge and project code"
    )
    parser.add_argument("concept", help="Concept to search for")
    parser.add_argument("--project", default="", help="Filter to a specific project")
    parser.add_argument("--knowledge-only", action="store_true", help="Only search knowledge")
    parser.add_argument("--code-only", action="store_true", help="Only search code")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--explain", action="store_true",
                        help="Show semantic vs grep breakdown")
    args = parser.parse_args()

    knowledge = []
    code_results = {"semantic": [], "grep": [], "all": []}

    if not args.code_only:
        knowledge = search_knowledge(args.concept)

    if not args.knowledge_only:
        code_results = search_code(args.concept, args.project, explain=args.explain)

    if args.json:
        print(json.dumps({
            "concept": args.concept,
            "knowledge": knowledge,
            "code": code_results["all"],
            "semantic_code": code_results["semantic"],
            "grep_code": code_results["grep"],
        }, indent=2))
        return

    print(f"CONCEPT: {args.concept}")
    print("=" * 50)

    if knowledge:
        print(f"\nKNOWLEDGE ({len(knowledge)} notes):")
        for k in knowledge:
            print(f"  {k['score']:.4f}  {k['path']}")
            if k.get("summary"):
                print(f"           {k['summary']}")

    all_code = code_results["all"]
    if all_code:
        semantic_files = {r["file"] for r in code_results["semantic"]}
        print(f"\nCODE ({len(all_code)} files):")
        for c in all_code:
            tag = "[semantic]" if c["file"] in semantic_files else "[grep]    "
            score_str = f"{c['score']:.4f}" if c["source"] == "semantic" else f"{int(c['score']*100):3d}%  "
            summary = f"  — {c['summary']}" if c.get("summary") else ""
            print(f"  {tag} {score_str}  [{c['project']}] {c['file']}{summary}")

        if args.explain:
            print(f"\n  --- Breakdown ---")
            print(f"  Semantic matches : {len(code_results['semantic'])}")
            print(f"  Grep-only matches: {len(code_results['grep'])}")

    # Bridge summary
    if knowledge and all_code:
        print(f"\nBRIDGE: {len(knowledge)} knowledge notes ↔ {len(all_code)} code files")
        print(f"  The vault knows about '{args.concept}' AND it's implemented in code.")
    elif knowledge and not all_code:
        print(f"\nGAP: Knowledge exists but no code implementation found.")
    elif all_code and not knowledge:
        print(f"\nGAP: Code exists but no knowledge note about it.")
    elif not knowledge and not all_code:
        print(f"\nNo results found for '{args.concept}'.")


if __name__ == "__main__":
    main()
