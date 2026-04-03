#!/usr/bin/env python3
# NOTE: For best performance, install numpy: pip install numpy
# With numpy: ~85ms/query. Without numpy: ~148ms/query (pure Python fallback).
"""
vault-search.py — Local semantic search over files indexed by vault-index.py.

Hybrid search (BM25 + embeddings + Reciprocal Rank Fusion) with optional
LLM re-ranking, HyDE query expansion, and knowledge graph context.
Runs entirely on your machine via Ollama — no API cost, no data leaves
your network.

When entity/relation tables are present (populated by vault-graph.py),
search results automatically include a "Graph Context" section showing
how matching concepts connect to each other.

Includes a catalyst index: pre-computed thematic questions per entity that
enable sub-millisecond concept lookup without Ollama.

Usage:
    vault-search init ~/my-vault           # Build FTS5 index + catalysts
    vault-search "authentication middleware"
    vault-search "convex schema design" --top 10
    vault-search "React hooks" --path Projects/
    vault-search "getTenantId" --mode bm25
    vault-search "auth patterns" --rerank        # LLM re-ranking
    vault-search "auth patterns" --expand --rerank  # full pipeline
    vault-search "attention" --intent "machine learning"  # intent steering
    vault-search 'lex:"reciprocal rank" vec:search fusion' # typed sub-queries
    vault-search "auth patterns" --explain        # scoring details
    vault-search "auth patterns" --fast           # catalyst-only sub-ms lookup
    vault-search "auth patterns" --no-embeddings  # BM25 + catalysts, no Ollama
    vault-search status                           # show index stats

Typed sub-queries:
    lex:"term"     BM25 keyword search for that term
    vec:"concept"  Embedding similarity search for that concept
    hyde:"question" HyDE expansion search for that question
    (unprefixed)   Normal hybrid search

Environment variables:
    OLLAMA_BASE       Ollama API URL (default: http://localhost:11434)
    EMBED_MODEL       Embedding model (default: qwen3-embedding:0.6b)
    EXPAND_MODEL      HyDE expansion model (default: qwen3.5:9b)
    RERANK_MODEL      Re-ranking model (default: qwen3.5:9b)
    VAULT_SEARCH_DB   Database path override (default: auto per root dir)
"""

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import sys
import time
import urllib.request
from pathlib import Path

# Auto-discover ClaudeLab venv for numpy, sentence-transformers, torch
_CLAUDELAB_VENV = Path.home() / "Documents/TestVault/Projects/ClaudeLab/.venv/lib"
for _sp in sorted(_CLAUDELAB_VENV.glob("python*/site-packages"), reverse=True):
    if str(_sp) not in sys.path:
        sys.path.insert(0, str(_sp))
    break

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Cross-encoder reranker (optional: pip install sentence-transformers)
HAS_CROSS_ENCODER = False
_cross_encoder = None
try:
    from sentence_transformers import CrossEncoder as _CE
    HAS_CROSS_ENCODER = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Config (all overridable via env vars)
# ---------------------------------------------------------------------------

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:0.6b")
EXPAND_MODEL = os.environ.get("EXPAND_MODEL", "qwen3.5:9b")  # updated from qwen3:8b (deprecated 2026-03-20)
RERANK_MODEL = os.environ.get("RERANK_MODEL", "qwen3.5:9b")  # updated from qwen3:8b (deprecated 2026-03-20)

RERANK_TOP_N = 20   # Candidates to re-rank from retrieval stage
RERANK_BLEND_K = 5  # Position-aware blending: top-k results trust retrieval more

# Set True via --no-embeddings to skip all Ollama embedding calls
_NO_EMBEDDINGS: bool = False


class EmbeddingsUnavailable(Exception):
    """Raised when embeddings are disabled or Ollama is unreachable."""
CROSS_ENCODER_MODEL = os.environ.get(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def _get_cross_encoder():
    """Lazy-load cross-encoder model (first call ~2s, subsequent calls ~0ms)."""
    global _cross_encoder
    if _cross_encoder is None and HAS_CROSS_ENCODER:
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass
        # Suppress model loading noise
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        try:
            _cross_encoder = _CE(CROSS_ENCODER_MODEL, device=device)
        except Exception:
            pass  # Fall back to Ollama-based reranking
    return _cross_encoder


def db_path_for_root(root: Path) -> Path:
    """Deterministic DB path per root directory. Matches vault-index.py."""
    custom = os.environ.get("VAULT_SEARCH_DB")
    if custom:
        return Path(custom)
    root_hash = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]
    return Path.home() / ".local" / "share" / "vault-search" / f"{root_hash}.db"


def escape_like(s: str) -> str:
    """Escape SQL LIKE wildcards (%, _) so they match literally."""
    return s.replace("%", "\\%").replace("_", "\\_")


# ---------------------------------------------------------------------------
# Math — NumPy accelerated with pure-Python fallback
# ---------------------------------------------------------------------------

def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def unpack_embedding(blob: bytes) -> list[float]:
    """Unpack a binary embedding blob back to a list of floats.

    Auto-detects format by blob size relative to known dimension count (1024):
      - 2048 bytes = float16 (2 bytes/dim)
      - 4096 bytes = float32 (4 bytes/dim)
      - 8192 bytes = float64 (8 bytes/dim)
    Falls back to float32 for unknown sizes.
    """
    n_f16 = len(blob) // 2
    n_f64 = len(blob) // 8

    # float64 detection (legacy): blob is exactly 1024 doubles
    if len(blob) % 8 == 0 and n_f64 == 1024:
        return list(struct.unpack(f'{n_f64}d', blob))
    # float16 detection: blob is exactly 1024 halves (2048 bytes)
    if len(blob) == 2048 and n_f16 == 1024:
        if HAS_NUMPY:
            return np.frombuffer(blob, dtype=np.float16).astype(np.float32).tolist()
        else:
            # Pure-Python: struct has no half-float, so we skip
            pass
    # float32 (current default or fallback)
    n_f32 = len(blob) // 4
    return list(struct.unpack(f'{n_f32}f', blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    n_a = norm(a)
    n_b = norm(b)
    if n_a == 0 or n_b == 0:
        return 0.0
    return dot(a, b) / (n_a * n_b)


def vectorized_search(
    rows: list,
    q_emb: list[float],
    query: str,
    path_filter: str | None,
) -> list[tuple[float, str, str, str | None, tuple[int, int] | None]]:
    """NumPy-accelerated cosine similarity over all embeddings.

    ~9x faster than the pure-Python path (14ms vs 127ms for 1600 files).
    Uses np.frombuffer for zero-copy blob unpacking and matrix multiply
    for batched dot products.
    """
    n = len(rows)
    dims = len(q_emb)
    emb_matrix = np.zeros((n, dims), dtype=np.float32)
    norms_arr = np.zeros(n, dtype=np.float32)
    paths = []
    summaries = []

    for i, (path, blob, enorm, summary) in enumerate(rows):
        if len(blob) == dims * 8:
            emb_matrix[i] = np.frombuffer(blob, dtype=np.float64).astype(np.float32)
        elif len(blob) == dims * 2:
            emb_matrix[i] = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
        else:
            emb_matrix[i] = np.frombuffer(blob, dtype=np.float32)
        norms_arr[i] = enorm or 0.0
        paths.append(path)
        summaries.append(summary or "")

    q_vec = np.array(q_emb, dtype=np.float32)
    q_norm = np.linalg.norm(q_vec)

    if q_norm == 0:
        return []

    dots = emb_matrix @ q_vec
    safe_norms = np.where(norms_arr == 0, 1.0, norms_arr)
    cosines = dots / (safe_norms * q_norm)
    cosines = np.where(norms_arr == 0, 0.0, cosines)

    scores = cosines.copy()
    for i, path in enumerate(paths):
        kw = keyword_score(query, path)
        scores[i] += 0.15 * kw

    order = np.argsort(scores)[::-1]
    return [(float(scores[idx]), paths[idx], summaries[idx], None, None) for idx in order]


def keyword_score(query: str, path: str) -> float:
    """Score based on how many query terms match words in the file path.

    Uses both substring matching and stem-aware matching (shared 5-char
    prefix) so "notification" matches "notify". Returns 0.0-1.0.
    """
    terms = query.lower().split()
    path_str = path.lower()
    for ch in "/-_.":
        path_str = path_str.replace(ch, " ")
    path_words = path_str.split()

    if not terms:
        return 0.0

    def term_matches(term: str) -> bool:
        if term in path.lower():
            return True
        for w in path_words:
            prefix_len = min(len(term), len(w), 5)
            if prefix_len >= 4 and term[:prefix_len] == w[:prefix_len]:
                return True
        return False

    hits = sum(1 for t in terms if term_matches(t))
    return hits / len(terms)


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

_embed_cache: dict[str, list[float]] = {}
_EMBED_CACHE_MAX = 64

# Disk embedding cache — persists across runs; keyed by sha256(model:text)
_VAULT_ROOT = Path(__file__).resolve().parent.parent
EMBED_DISK_CACHE_DIR = _VAULT_ROOT / "_data" / "embed-cache"
_disk_cache_enabled: bool = True  # set to False via --no-cache


def _disk_cache_path(text: str) -> Path:
    """Return the cache file path for a given text + current model."""
    key = hashlib.sha256(f"{EMBED_MODEL}:{text}".encode()).hexdigest()
    return EMBED_DISK_CACHE_DIR / f"{key}.json"


def _load_disk_cache(text: str) -> list[float] | None:
    """Return cached embedding vector, or None on miss."""
    if not _disk_cache_enabled:
        return None
    path = _disk_cache_path(text)
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return None


def _save_disk_cache(text: str, emb: list[float]) -> None:
    """Write embedding vector to disk cache."""
    if not _disk_cache_enabled:
        return
    try:
        EMBED_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _disk_cache_path(text).write_text(json.dumps(emb))
    except Exception:
        pass  # Cache write failure is non-fatal


def ollama_embed(text: str) -> list[float]:
    if _NO_EMBEDDINGS:
        raise EmbeddingsUnavailable("Embeddings disabled via --no-embeddings")
    # 1. In-memory cache (process-lifetime, instant)
    if text in _embed_cache:
        return _embed_cache[text]
    # 2. Disk cache (cross-run, ~1ms)
    cached = _load_disk_cache(text)
    if cached is not None:
        _embed_cache[text] = cached
        return cached
    # 3. Ollama API call (~5500ms cold)
    payload = json.dumps({
        "model": EMBED_MODEL,
        "prompt": text,
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    emb = data["embedding"]
    if len(_embed_cache) >= _EMBED_CACHE_MAX:
        _embed_cache.pop(next(iter(_embed_cache)))
    _embed_cache[text] = emb
    _save_disk_cache(text, emb)
    return emb


def hyde_expand(query: str, intent: str | None = None) -> str:
    """HyDE: generate a hypothetical document to improve embedding alignment.

    Instead of embedding the raw query, we ask the local LLM to generate a
    short passage that WOULD answer the query, then embed that. Document-to-
    document similarity is more reliable than question-to-document.

    When intent is provided, it steers the expansion toward the right domain
    (e.g., "attention" + intent="machine learning" → ML attention, not UX).
    """
    intent_ctx = f" in the context of {intent}" if intent else ""
    prompt = (
        f"Write a short technical passage (2-3 sentences or a brief code snippet) "
        f"that directly answers or demonstrates: {query}{intent_ctx}\n"
        f"Be specific. Use technical terminology. No preamble."
    )
    payload = json.dumps({
        "model": EXPAND_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.3, "num_predict": 120, "num_ctx": 8192},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read())
        expanded = data["response"].strip()
        return expanded if expanded else query
    except Exception:
        return query


# ---------------------------------------------------------------------------
# Re-ranking
# ---------------------------------------------------------------------------

_rerank_cache: dict[str, float] = {}


def rerank_score(query: str, doc_snippet: str, intent: str | None = None) -> float:
    """Score a single document's relevance to a query (0.0-1.0).

    Prefers cross-encoder (sentence-transformers) when available — ~66ms/pair
    on GPU, more accurate. Falls back to Ollama LLM prompt scoring (~120ms).
    """
    effective_query = f"{query} ({intent})" if intent else query
    cache_key = f"{effective_query}||{doc_snippet[:100]}"
    if cache_key in _rerank_cache:
        return _rerank_cache[cache_key]

    score = 0.5

    # Try cross-encoder first (faster, more accurate)
    ce = _get_cross_encoder()
    if ce is not None:
        try:
            raw = float(ce.predict([(effective_query, doc_snippet)])[0])
            # Sigmoid to normalize raw logits to 0-1
            score = 1.0 / (1.0 + math.exp(-raw))
        except Exception:
            score = _rerank_score_ollama(query, doc_snippet, intent)
    else:
        score = _rerank_score_ollama(query, doc_snippet, intent)

    _rerank_cache[cache_key] = score
    return score


def _rerank_score_ollama(query: str, doc_snippet: str, intent: str | None = None) -> float:
    """Fallback: Ollama LLM-based relevance scoring (~120ms/pair)."""
    intent_line = f"\nIntent: {intent}" if intent else ""
    prompt = (
        "Rate relevance 0-10. Only output the number.\n"
        f"Query: {query}{intent_line}\n"
        f"Document: {doc_snippet}\n"
        "Score:"
    )
    payload = json.dumps({
        "model": RERANK_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 3, "num_ctx": 2048},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        text = data["response"].strip()
        m = re.search(r'(\d+(?:\.\d+)?)', text)
        raw = float(m.group(1)) if m else 5.0
        return min(max(raw / 10.0, 0.0), 1.0)
    except Exception:
        return 0.5


def rerank_results(
    query: str,
    results: list[tuple],
    conn: sqlite3.Connection,
    top_n: int = RERANK_TOP_N,
    intent: str | None = None,
) -> list[tuple]:
    """Re-rank top candidates using LLM scoring with position-aware blending.

    Takes the top_n results from retrieval, scores each with the LLM,
    then blends retrieval rank with reranker score. High-confidence
    retrieval results (top RERANK_BLEND_K) trust retrieval more to prevent
    the reranker from destroying exact matches.
    """
    if not results:
        return results

    candidates = results[:top_n]
    rest = results[top_n:]

    paths = [r[1] for r in candidates]
    paths_in = ", ".join("?" * len(paths))
    content_map: dict[str, str] = {}

    q_terms = [t for t in query.lower().split() if len(t) >= 3]

    try:
        rows = conn.execute(
            f"SELECT path, content, summary FROM files WHERE path IN ({paths_in})",
            paths,
        ).fetchall()
        for path, content, summary in rows:
            text = content or ""
            if not text:
                content_map[path] = (summary or path)[:400]
                continue

            text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.DOTALL)

            if len(text) <= 500:
                content_map[path] = text[:400]
                continue

            # Sliding window: find the 400-char window with most term hits
            text_lower = text.lower()
            best_start = 0
            best_hits = 0
            step = 100
            for start in range(0, len(text) - 400, step):
                window = text_lower[start:start + 400]
                hits = sum(1 for t in q_terms if t in window)
                if hits > best_hits:
                    best_hits = hits
                    best_start = start

            if best_hits > 0:
                content_map[path] = text[best_start:best_start + 400].strip()
            else:
                content_map[path] = (summary or text[:400]).strip()[:400]
    except Exception:
        for candidate in candidates:
            path, summary = candidate[1], candidate[2]
            if path not in content_map:
                content_map[path] = (summary or path)[:400]

    # Normalize retrieval scores to 0-1 before blending.
    # RRF scores are tiny (~0.016-0.033), reranker scores are 0.0-1.0.
    max_retrieval = max(c[0] for c in candidates) if candidates else 1.0
    if max_retrieval == 0:
        max_retrieval = 1.0

    # Batch-score with cross-encoder if available (much faster than one-by-one)
    snippets = []
    for candidate in candidates:
        path, summary = candidate[1], candidate[2]
        snippets.append(content_map.get(path, summary[:400] if summary else path))

    effective_query = f"{query} ({intent})" if intent else query
    ce = _get_cross_encoder()
    if ce is not None:
        try:
            pairs = [(effective_query, s) for s in snippets]
            raw_scores = ce.predict(pairs)
            rr_scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]
        except Exception:
            rr_scores = [rerank_score(query, s, intent) for s in snippets]
    else:
        rr_scores = [rerank_score(query, s, intent) for s in snippets]

    scored: list[tuple] = []
    for rank, candidate in enumerate(candidates):
        retrieval_score, path, summary = candidate[0], candidate[1], candidate[2]
        c_heading = candidate[3] if len(candidate) > 3 else None
        c_lines = candidate[4] if len(candidate) > 4 else None
        rr_score = rr_scores[rank]
        norm_retrieval = retrieval_score / max_retrieval

        if rank < RERANK_BLEND_K:
            blended = 0.6 * norm_retrieval + 0.4 * rr_score
        else:
            blended = 0.4 * norm_retrieval + 0.6 * rr_score

        scored.append((blended, rr_score, rank, path, summary, c_heading, c_lines))

    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [(b, p, s, ch, cl) for b, _, _, p, s, ch, cl in scored]

    return reranked + rest


# ---------------------------------------------------------------------------
# BM25 / FTS5
# ---------------------------------------------------------------------------

def _fts5_available(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='files_fts'"
    ).fetchone()
    if not row:
        return False
    count = conn.execute("SELECT COUNT(*) FROM files_fts").fetchone()[0]
    return count > 0


def bm25_search(
    conn: sqlite3.Connection,
    query: str,
    path_filter: str | None,
    limit: int = 100,
) -> list[tuple[str, float]]:
    """Full-text BM25 search via SQLite FTS5 trigram index.

    Returns [(path, score)] sorted best-first. Uses AND matching first,
    falls back to OR if fewer than 3 results. The trigram tokenizer treats
    each term as a substring match — ideal for code identifiers.
    """
    if not _fts5_available(conn):
        return []

    words = [w for w in re.findall(r'\w+', query.lower()) if len(w) >= 3]
    if not words:
        return []

    base = (
        "SELECT path, -bm25(files_fts) AS score "
        "FROM files_fts WHERE content MATCH ? "
    )
    if path_filter:
        base += "AND path LIKE ? ESCAPE '\\' "
    base += "ORDER BY score DESC LIMIT ?"

    def run(fts_expr: str) -> list[tuple[str, float]]:
        params: list = [fts_expr]
        if path_filter:
            params.append(f"{escape_like(path_filter)}%")
        params.append(limit)
        try:
            return conn.execute(base, params).fetchall()
        except Exception:
            return []

    and_expr = " ".join(f'"{w}"' for w in words)
    results = run(and_expr)

    if len(results) < 3 and len(words) > 1:
        or_expr = " OR ".join(f'"{w}"' for w in words)
        results = run(or_expr)

    return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def rrf_fusion(
    semantic: list[tuple],
    bm25: list[tuple[str, float]],
    k: int = 60,
    top_k: int = 10,
) -> list[tuple]:
    """Reciprocal Rank Fusion: merge semantic and BM25 ranked lists.

    Each document gets a score of sum(1 / (k + rank_i)) across all lists.
    k=60 is the standard default (Cormack et al. 2009).
    Preserves chunk metadata (heading, line range) from semantic results.
    """
    all_docs: dict[str, dict] = {}

    for rank, item in enumerate(semantic):
        path, summary = item[1], item[2]
        c_heading = item[3] if len(item) > 3 else None
        c_lines = item[4] if len(item) > 4 else None
        all_docs.setdefault(path, {"summary": summary, "rrf": 0.0,
                                   "chunk_heading": c_heading, "chunk_lines": c_lines})
        all_docs[path]["rrf"] += 1.0 / (k + rank + 1)

    for rank, (path, score) in enumerate(bm25):
        if path not in all_docs:
            all_docs[path] = {"summary": "", "rrf": 0.0,
                              "chunk_heading": None, "chunk_lines": None}
        all_docs[path]["rrf"] += 1.0 / (k + rank + 1)

    fused = [(v["rrf"], path, v["summary"], v["chunk_heading"], v["chunk_lines"])
             for path, v in all_docs.items()]
    fused.sort(reverse=True)
    return fused[:top_k]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _fallback_semantic(
    rows: list,
    q_emb: list[float],
    query: str,
) -> list[tuple]:
    """Pure-Python semantic search fallback (no NumPy)."""
    q_norm_val = norm(q_emb)
    results: list[tuple] = []
    for path, emb_data, emb_norm, summary in rows:
        try:
            if isinstance(emb_data, bytes):
                emb = unpack_embedding(emb_data)
            else:
                emb = json.loads(emb_data)
                emb_norm = norm(emb)
        except (json.JSONDecodeError, TypeError, struct.error):
            continue
        if q_norm_val == 0 or (emb_norm or 0) == 0:
            cos = 0.0
        else:
            cos = dot(q_emb, emb) / (q_norm_val * emb_norm)
        kw = keyword_score(query, path)
        score = cos + 0.15 * kw
        results.append((score, path, summary or "", None, None))
    results.sort(key=lambda x: x[0], reverse=True)
    return results


def _print_explain(log: list[str], results: list[tuple]) -> None:
    """Print explain log and per-result scoring details to stderr."""
    print("\n--- EXPLAIN ---", file=sys.stderr)
    for line in log:
        print(f"  {line}", file=sys.stderr)
    print("", file=sys.stderr)
    print("  Per-result scores:", file=sys.stderr)
    for i, result in enumerate(results[:20]):
        score, path = result[0], result[1]
        c_heading = result[3] if len(result) > 3 else None
        heading_info = f" § {c_heading}" if c_heading else ""
        print(f"  {i+1:3d}. {score:.4f}  {path}{heading_info}", file=sys.stderr)
    print("--- END EXPLAIN ---\n", file=sys.stderr)


def parse_typed_subqueries(query: str) -> dict[str, list[str]]:
    """Parse typed sub-query prefixes from a query string.

    Supports lex:"term", vec:"concept", hyde:"question" prefixes.
    Unprefixed text is collected under the "plain" key.

    Returns {"lex": [...], "vec": [...], "hyde": [...], "plain": [...]}.
    """
    result: dict[str, list[str]] = {"lex": [], "vec": [], "hyde": [], "plain": []}

    # Match lex:"...", vec:"...", hyde:"..." (with or without quotes)
    pattern = r'(lex|vec|hyde):(?:"([^"]+)"|(\S+))'
    remainder = query
    for match in re.finditer(pattern, query):
        prefix = match.group(1)
        value = match.group(2) or match.group(3)
        result[prefix].append(value)
        remainder = remainder.replace(match.group(0), "", 1)

    plain = remainder.strip()
    if plain:
        result["plain"].append(plain)

    return result


# Heading terms that are generic meta-section labels — not useful as expansion terms.
# These appear across many notes as section headers but carry no topical signal.
_EXPANSION_HEADING_STOPLIST = {
    # Generic note structure headers
    "assessment", "growth", "session", "overview", "summary", "background",
    "context", "introduction", "conclusion", "discussion", "analysis",
    "notes", "references", "sources", "connections", "links", "related",
    "examples", "applications", "implications", "limitations", "challenges",
    "next steps", "action items", "todo", "status", "updates", "history",
    "definition", "definitions", "description", "details", "info",
    "key points", "key takeaways", "takeaways", "highlights", "tags",
    "metadata", "properties", "frontmatter", "resources", "further reading",
    "see also", "appendix", "table of contents", "toc", "index",
    # Vault-specific generic headers
    "synthesis", "reflection", "observations", "findings", "results",
    "implementation", "usage", "setup", "configuration", "installation",
    "troubleshooting", "faq", "questions", "answers", "review", "feedback",
    "project", "projects", "tasks", "goals", "objectives", "milestones",
    "timeline", "schedule", "plan", "planning", "roadmap",
    # Single-word generic terms that leak from headings
    "section", "chapter", "part", "phase", "stage", "step", "steps",
    "approach", "method", "methods", "technique", "techniques", "framework",
    "theory", "practice", "principles", "concepts", "ideas",
}


def _extract_vault_expansion_terms(
    results: list[tuple],
    conn: sqlite3.Connection,
    original_query: str,
    top_n: int = 4,
) -> list[str]:
    """
    Extract key terms from top search results for iterative query expansion.

    Pulls: Obsidian [[wikilinks]], YAML front-matter tags, capitalized proper
    nouns, and high-frequency n-grams not already in the original query.
    Returns up to 10 expansion terms.
    """
    q_words = set(re.findall(r'\w+', original_query.lower()))
    terms: list[str] = []
    seen: set[str] = set()

    paths = [r[1] for r in results[:top_n]]
    if not paths:
        return terms

    paths_in = ", ".join("?" * len(paths))
    try:
        rows = conn.execute(
            f"SELECT path, content, summary FROM files WHERE path IN ({paths_in})",
            paths,
        ).fetchall()
    except Exception:
        return terms

    for _, content, summary in rows:
        text = content or summary or ""
        if not text:
            continue

        # Obsidian [[wikilinks]] — the vault's native concept vocabulary
        for m in re.finditer(r'\[\[([^\]|#]{2,60})(?:[|#][^\]]{0,60})?\]\]', text):
            link = m.group(1).strip()
            link_lower = link.lower().replace("-", " ")
            words = set(link_lower.split())
            if words - q_words and link_lower not in seen:
                terms.append(link)
                seen.add(link_lower)

        # YAML front-matter tags: `tags: [foo, bar]` or `- tag`
        fm_match = re.match(r'^---\n(.*?)\n---', text, re.DOTALL)
        if fm_match:
            fm = fm_match.group(1)
            for m in re.finditer(r'[\-\s]([a-z][a-z0-9_-]{2,30})', fm):
                tag = m.group(1).replace("-", " ")
                if tag not in seen and tag not in q_words:
                    terms.append(tag)
                    seen.add(tag)

        # Heading lines (## Section Name) — highest-signal structure markers
        for m in re.finditer(r'^#{1,3}\s+(.{3,60})$', text, re.MULTILINE):
            heading = m.group(1).strip()
            heading_lower = heading.lower()
            # Skip generic meta-section headers that add no topical signal
            if heading_lower in _EXPANSION_HEADING_STOPLIST:
                continue
            # Also skip if every word in the heading is in the stoplist
            heading_words = set(re.findall(r'\w+', heading_lower))
            if heading_words and heading_words.issubset(_EXPANSION_HEADING_STOPLIST):
                continue
            words = heading_words
            # Only add if heading introduces new vocabulary
            if len(words - q_words) >= 2 and heading_lower not in seen:
                terms.append(heading)
                seen.add(heading_lower)

        # Backtick identifiers (code, function names, keys)
        for m in re.finditer(r'`([a-zA-Z_][a-zA-Z0-9_:.-]{2,40})`', text):
            ident = m.group(1)
            ident_lower = ident.lower()
            if ident_lower not in seen and ident_lower not in q_words:
                terms.append(ident)
                seen.add(ident_lower)

    return terms[:10]


def iterative_search(
    query: str,
    db_path: Path,
    top_k: int = 5,
    path_filter: str | None = None,
    mode: str = "hybrid",
    expand: bool = False,
    rerank: bool = False,
    intent: str | None = None,
    explain: bool = False,
) -> list[tuple]:
    """
    Two-pass iterative retrieval (MSA-inspired).

    Pass 1: standard search with the original query — finds the obvious matches.
    Extract: key terms from pass-1 results (wikilinks, headings, identifiers).
    Pass 2: search again with query + extracted terms — finds cross-referenced
            sections that use different vocabulary than the original query.
    Merge: RRF-fuse both result sets, deduplicate, return top-k.

    Example: "how does the autoresearch system handle failed experiments"
      Pass 1 → finds autoresearch.md, goals.md
      Extract: [[output-judge]], [[quality-triage]], "3-attempt limit"
      Pass 2 → also finds auto-implement.md, heartbeat/schedule.md
      Result: the full failure-handling pipeline, not just the entry point.
    """
    if explain:
        print("[ITERATE] Pass 1: standard search", file=sys.stderr)

    # Multi-hop queries need more candidates — use at least 10 in iterate mode
    # so the term extractor has enough cross-references to work with.
    effective_top_k = max(top_k, 10)

    # Pass 1 — standard retrieval
    conn = sqlite3.connect(str(db_path))
    pass1 = search(
        query, db_path, top_k=effective_top_k * 2,
        path_filter=path_filter, mode=mode,
        expand=expand, rerank=False,  # skip rerank on pass 1 — do it after merge
        intent=intent, explain=False,  # suppress nested explain
    )

    if not pass1:
        if rerank:
            pass1 = search(query, db_path, top_k=top_k, path_filter=path_filter,
                           mode=mode, expand=expand, rerank=rerank, intent=intent,
                           explain=explain)
        conn.close()
        return pass1

    # Extract expansion terms from pass-1 results
    expansion_terms = _extract_vault_expansion_terms(pass1, conn, query)
    conn.close()

    if explain:
        print(f"[ITERATE] Extracted {len(expansion_terms)} expansion terms: "
              f"{expansion_terms}", file=sys.stderr)

    if not expansion_terms:
        # Nothing to expand — fall back to single-pass (with rerank if requested)
        if rerank:
            return search(query, db_path, top_k=top_k, path_filter=path_filter,
                          mode=mode, expand=expand, rerank=True, intent=intent,
                          explain=explain)
        return pass1[:top_k]

    # Build expanded query
    expanded_query = query + " " + " ".join(expansion_terms)
    if explain:
        print(f"[ITERATE] Expanded query: {expanded_query[:120]}...", file=sys.stderr)
        print("[ITERATE] Pass 2: expanded search", file=sys.stderr)

    # Pass 2 — retrieval with expanded query (no HyDE expansion to avoid double-expansion)
    pass2 = search(
        expanded_query, db_path, top_k=effective_top_k * 2,
        path_filter=path_filter, mode=mode,
        expand=False,  # don't HyDE-expand an already-expanded query
        rerank=False,
        intent=intent, explain=False,
    )

    if explain:
        pass1_paths = {r[1] for r in pass1}
        new_in_pass2 = [r for r in pass2 if r[1] not in pass1_paths]
        print(f"[ITERATE] Pass 2 found {len(new_in_pass2)} new paths: "
              f"{[r[1] for r in new_in_pass2[:5]]}", file=sys.stderr)

    # RRF-fuse pass 1 and pass 2 result lists
    # Build path→best-tuple maps from each pass
    p1_map: dict[str, tuple] = {r[1]: r for r in pass1}
    p2_map: dict[str, tuple] = {r[1]: r for r in pass2}

    all_docs: dict[str, dict] = {}
    for rank, item in enumerate(pass1):
        path = item[1]
        all_docs.setdefault(path, {"item": item, "rrf": 0.0})
        all_docs[path]["rrf"] += 1.0 / (60 + rank + 1)

    for rank, item in enumerate(pass2):
        path = item[1]
        if path not in all_docs:
            all_docs[path] = {"item": item, "rrf": 0.0}
        all_docs[path]["rrf"] += 1.0 / (60 + rank + 1)

    fused = sorted(all_docs.values(), key=lambda x: x["rrf"], reverse=True)
    merged = [(v["rrf"], v["item"][1], v["item"][2],
               v["item"][3] if len(v["item"]) > 3 else None,
               v["item"][4] if len(v["item"]) > 4 else None)
              for v in fused]

    # Optional rerank on merged candidates
    if rerank:
        if explain:
            print("[ITERATE] Re-ranking merged results...", file=sys.stderr)
        conn2 = sqlite3.connect(str(db_path))
        merged = rerank_results(query, merged, conn2, intent=intent)[:top_k]
        conn2.close()
    else:
        merged = merged[:top_k]

    if explain:
        print(f"[ITERATE] Final: {len(merged)} results after merge", file=sys.stderr)

    return merged


def search(
    query: str,
    db_path: Path,
    top_k: int = 5,
    path_filter: str | None = None,
    mode: str = "hybrid",
    expand: bool = False,
    rerank: bool = False,
    intent: str | None = None,
    explain: bool = False,
) -> list[tuple]:
    """Search the index. Returns list of (score, path, summary, chunk_heading, chunk_lines).

    chunk_heading is the matched section heading (or None for file-level matches).
    chunk_lines is (start_line, end_line) tuple (or None).

    mode:
      hybrid   — semantic + BM25 via Reciprocal Rank Fusion (default)
      semantic — cosine similarity on whole-file embeddings only
      bm25     — keyword matching via FTS5 trigram index only

    expand:
      HyDE (Hypothetical Document Embeddings) — generates a synthetic
      passage before embedding. Improves recall for conceptual queries.

    rerank:
      Re-scores top candidates using LLM-based relevance scoring.
      Adds ~2-3s but significantly improves result quality.

    intent:
      Domain context that steers expansion, reranking, and snippet
      extraction (e.g., "machine learning" disambiguates "attention").

    explain:
      When True, prints scoring details at each pipeline stage to stderr.
    """
    explain_log: list[str] = []  # Collect explain output
    t_start = time.time()

    def _explain(msg: str) -> None:
        if explain:
            elapsed = (time.time() - t_start) * 1000
            explain_log.append(f"[{elapsed:7.1f}ms] {msg}")

    if not db_path.exists():
        print(f"Error: Index not found at {db_path}.", file=sys.stderr)
        print("Run vault-index.py first to build the index.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))

    columns = {row[1] for row in conn.execute("PRAGMA table_info(files)").fetchall()}
    has_norms = "embedding_norm" in columns
    if not has_norms:
        conn.execute("ALTER TABLE files ADD COLUMN embedding_norm REAL")
        conn.commit()

    _explain(f"DB opened: {db_path}")
    if intent:
        _explain(f"Intent: {intent}")

    # --- Typed sub-query handling ---
    subqueries = parse_typed_subqueries(query)
    has_typed = any(subqueries[k] for k in ("lex", "vec", "hyde"))

    if has_typed:
        _explain(f"Typed sub-queries: lex={subqueries['lex']} vec={subqueries['vec']} "
                 f"hyde={subqueries['hyde']} plain={subqueries['plain']}")

        # Run each sub-query type through its pipeline, collect results for RRF
        all_ranked_lists: list[list[tuple]] = []
        all_bm25_lists: list[list[tuple[str, float]]] = []

        # lex: sub-queries → BM25 only
        for lq in subqueries["lex"]:
            t0 = time.time()
            bm25_res = bm25_search(conn, lq, path_filter, limit=top_k * 4)
            _explain(f"lex:\"{lq}\" → {len(bm25_res)} BM25 results ({(time.time()-t0)*1000:.0f}ms)")
            all_bm25_lists.append(bm25_res)

        # Load embeddings once for vec/hyde/plain queries
        if path_filter:
            rows = conn.execute(
                "SELECT path, embedding, embedding_norm, summary FROM files WHERE path LIKE ? ESCAPE '\\'",
                (f"{escape_like(path_filter)}%",)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT path, embedding, embedding_norm, summary FROM files"
            ).fetchall()
        summary_map = {path: (summary or "") for path, _, _, summary in rows}

        # vec: sub-queries → embedding search only
        for vq in subqueries["vec"]:
            t0 = time.time()
            vq_emb = ollama_embed(vq)
            if HAS_NUMPY and all(isinstance(r[1], bytes) for r in rows[:10]):
                sem = vectorized_search(rows, vq_emb, vq, path_filter)
            else:
                sem = _fallback_semantic(rows, vq_emb, vq)
            _explain(f"vec:\"{vq}\" → {len(sem)} semantic results ({(time.time()-t0)*1000:.0f}ms)")
            all_ranked_lists.append(sem)

        # hyde: sub-queries → HyDE expansion + embedding search
        for hq in subqueries["hyde"]:
            t0 = time.time()
            expanded = hyde_expand(hq, intent)
            hq_emb = ollama_embed(expanded)
            if HAS_NUMPY and all(isinstance(r[1], bytes) for r in rows[:10]):
                sem = vectorized_search(rows, hq_emb, hq, path_filter)
            else:
                sem = _fallback_semantic(rows, hq_emb, hq)
            _explain(f"hyde:\"{hq}\" → expanded + {len(sem)} semantic results ({(time.time()-t0)*1000:.0f}ms)")
            all_ranked_lists.append(sem)

        # plain text → normal hybrid
        for pq in subqueries["plain"]:
            t0 = time.time()
            pq_emb = ollama_embed(pq)
            if HAS_NUMPY and all(isinstance(r[1], bytes) for r in rows[:10]):
                sem = vectorized_search(rows, pq_emb, pq, path_filter)
            else:
                sem = _fallback_semantic(rows, pq_emb, pq)
            bm25_res = bm25_search(conn, pq, path_filter, limit=top_k * 4)
            all_ranked_lists.append(sem)
            all_bm25_lists.append(bm25_res)
            _explain(f"plain:\"{pq}\" → {len(sem)} semantic + {len(bm25_res)} BM25 ({(time.time()-t0)*1000:.0f}ms)")

        # Multi-list RRF fusion
        all_docs: dict[str, dict] = {}
        for ranked_list in all_ranked_lists:
            for rank, item in enumerate(ranked_list):
                path, summary = item[1], item[2]
                c_heading = item[3] if len(item) > 3 else None
                c_lines = item[4] if len(item) > 4 else None
                all_docs.setdefault(path, {"summary": summary, "rrf": 0.0,
                                           "chunk_heading": c_heading, "chunk_lines": c_lines})
                all_docs[path]["rrf"] += 1.0 / (60 + rank + 1)

        for bm25_list in all_bm25_lists:
            for rank, (path, score) in enumerate(bm25_list):
                if path not in all_docs:
                    all_docs[path] = {"summary": summary_map.get(path, ""), "rrf": 0.0,
                                      "chunk_heading": None, "chunk_lines": None}
                all_docs[path]["rrf"] += 1.0 / (60 + rank + 1)

        fused = [(v["rrf"], path, v["summary"], v["chunk_heading"], v["chunk_lines"])
                 for path, v in all_docs.items()]
        fused.sort(reverse=True)
        fused = fused[:max(top_k, RERANK_TOP_N) if rerank else top_k]

        _explain(f"Multi-list RRF fusion → {len(fused)} candidates")

        if rerank:
            _explain("Re-ranking...")
            t0 = time.time()
            reranked = rerank_results(query, fused, conn, intent=intent)
            _explain(f"Re-ranking done ({(time.time()-t0)*1000:.0f}ms)")
            fused = reranked[:top_k]

        if explain:
            _explain(f"Final: {len(fused)} results")
            _print_explain(explain_log, fused)

        conn.close()
        return fused

    # --- Standard (non-typed) search path ---

    # BM25-only path
    if mode == "bm25":
        words = [w for w in re.findall(r'\w+', query.lower()) if len(w) >= 3]
        if not words:
            print(f"No BM25 results: all query terms are shorter than 3 characters. "
                  f"Try --mode semantic for short queries.", file=sys.stderr)
            conn.close()
            return []
        t0 = time.time()
        bm25_results = bm25_search(conn, query, path_filter, limit=top_k * 3)
        _explain(f"BM25 search → {len(bm25_results)} results ({(time.time()-t0)*1000:.0f}ms)")
        if not bm25_results:
            print("No BM25 results found. Try --mode semantic or re-run vault-index.py.",
                  file=sys.stderr)
            conn.close()
            return []
        paths_in = ", ".join("?" * len(bm25_results))
        path_list = [p for p, _ in bm25_results]
        summary_map: dict[str, str] = {}
        rows = conn.execute(
            f"SELECT path, summary FROM files WHERE path IN ({paths_in})",
            path_list
        ).fetchall()
        for p, s in rows:
            summary_map[p] = s or ""

        results = [(score, path, summary_map.get(path, ""), None, None)
                   for path, score in bm25_results[:top_k]]
        if explain:
            _explain(f"Final: {len(results)} results")
            _print_explain(explain_log, results)
        conn.close()
        return results

    # Get query embedding (optional — falls back to BM25+catalysts if unavailable)
    embed_query = query
    if expand and not _NO_EMBEDDINGS:
        print("Expanding query with HyDE...", file=sys.stderr)
        t0 = time.time()
        embed_query = hyde_expand(query, intent)
        _explain(f"HyDE expansion ({(time.time()-t0)*1000:.0f}ms): {embed_query[:100]}...")

    t0 = time.time()
    q_emb = None
    try:
        q_emb = ollama_embed(embed_query)
        _explain(f"Embedding query ({(time.time()-t0)*1000:.0f}ms)")
    except EmbeddingsUnavailable:
        _explain("Embeddings disabled — using BM25+catalysts only")
    except Exception as e:
        if _NO_EMBEDDINGS:
            _explain("Embeddings disabled — using BM25+catalysts only")
        else:
            print(f"Warning: Cannot reach Ollama at {OLLAMA_BASE}: {e}", file=sys.stderr)
            print("Falling back to BM25+catalysts. Use --no-embeddings to silence this.",
                  file=sys.stderr)

    # If embeddings unavailable, fall back to BM25 (+ catalyst context appended by caller)
    if q_emb is None:
        _explain("BM25-only fallback path")
        words = [w for w in re.findall(r'\w+', query.lower()) if len(w) >= 3]
        if not words:
            conn.close()
            return []
        bm25_results = bm25_search(conn, query, path_filter, limit=top_k * 3)
        if not bm25_results:
            conn.close()
            return []
        paths_in = ", ".join("?" * len(bm25_results))
        path_list = [p for p, _ in bm25_results]
        sm: dict[str, str] = {}
        for p, s in conn.execute(
            f"SELECT path, summary FROM files WHERE path IN ({paths_in})", path_list
        ).fetchall():
            sm[p] = s or ""
        results = [(score, path, sm.get(path, ""), None, None)
                   for path, score in bm25_results[:top_k]]
        if explain:
            _explain(f"Final: {len(results)} results (BM25 fallback)")
            _print_explain(explain_log, results)
        conn.close()
        return results

    # Check chunks
    has_chunks = False
    try:
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        has_chunks = chunk_count > 0
    except Exception:
        pass

    # Load file-level embeddings
    if path_filter:
        rows = conn.execute(
            "SELECT path, embedding, embedding_norm, summary FROM files WHERE path LIKE ? ESCAPE '\\'",
            (f"{escape_like(path_filter)}%",)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT path, embedding, embedding_norm, summary FROM files"
        ).fetchall()

    if not rows:
        print("No files in index" + (f" matching '{path_filter}'" if path_filter else "") + ".",
              file=sys.stderr)
        conn.close()
        sys.exit(1)

    _explain(f"Loaded {len(rows)} file embeddings" + (f", {chunk_count} chunks" if has_chunks else ""))
    summary_map = {path: (summary or "") for path, _, _, summary in rows}

    # Score all files
    t0 = time.time()
    if HAS_NUMPY and all(isinstance(r[1], bytes) for r in rows[:10]):
        semantic_results = vectorized_search(rows, q_emb, query, path_filter)
    else:
        q_norm = norm(q_emb)
        semantic_results: list[tuple] = []
        for path, emb_data, emb_norm, summary in rows:
            try:
                if isinstance(emb_data, bytes):
                    emb = unpack_embedding(emb_data)
                else:
                    emb = json.loads(emb_data)
                    emb_norm = norm(emb)
            except (json.JSONDecodeError, TypeError, struct.error):
                continue
            if q_norm == 0 or (emb_norm or 0) == 0:
                cos = 0.0
            else:
                cos = dot(q_emb, emb) / (q_norm * emb_norm)
            kw = keyword_score(query, path)
            score = cos + 0.15 * kw
            semantic_results.append((score, path, summary or "", None, None))
        semantic_results.sort(key=lambda x: x[0], reverse=True)

    _explain(f"Semantic search → {len(semantic_results)} results, "
             f"top={semantic_results[0][0]:.4f} ({(time.time()-t0)*1000:.0f}ms)")

    # Chunk-level search — find the best matching chunk per file
    # Performance: limit chunk search to top-N candidate files from semantic + BM25 results.
    # At 500k+ chunks, loading all takes ~5s. Restricting to top 200 candidate files
    # covers all results that could possibly surface in the final top_k output while
    # reducing chunk I/O by ~96%.
    CHUNK_CANDIDATE_LIMIT = 200
    if has_chunks:
        # Always pre-filter chunks to top semantic candidates.
        # With 577k+ chunks for Projects/ and 7.6k for Knowledge/, loading ALL chunks
        # for a path_filter is expensive (~5s) and redundant — chunks outside the top
        # semantic candidates can never appear in the final top_k output.
        # Fix 2026-03-22: apply the candidate pre-filter for path_filter case too.
        candidate_paths = [r[1] for r in semantic_results[:CHUNK_CANDIDATE_LIMIT]]
        if candidate_paths:
            if path_filter:
                # Intersect candidates with path_filter to restrict scope
                placeholders = ",".join("?" * len(candidate_paths))
                chunk_rows = conn.execute(
                    f"SELECT file_path, chunk_index, heading, start_line, end_line, "
                    f"embedding, embedding_norm "
                    f"FROM chunks WHERE file_path LIKE ? ESCAPE '\\' AND file_path IN ({placeholders})",
                    (f"{escape_like(path_filter)}%", *candidate_paths)
                ).fetchall()
            else:
                placeholders = ",".join("?" * len(candidate_paths))
                chunk_rows = conn.execute(
                    f"SELECT file_path, chunk_index, heading, start_line, end_line, "
                    f"embedding, embedding_norm "
                    f"FROM chunks WHERE file_path IN ({placeholders})",
                    candidate_paths
                ).fetchall()
        else:
            # Fallback: no semantic candidates — use path_filter alone or load all
            if path_filter:
                chunk_rows = conn.execute(
                    "SELECT file_path, chunk_index, heading, start_line, end_line, "
                    "embedding, embedding_norm "
                    "FROM chunks WHERE file_path LIKE ? ESCAPE '\\'",
                    (f"{escape_like(path_filter)}%",)
                ).fetchall()
            else:
                chunk_rows = conn.execute(
                    "SELECT file_path, chunk_index, heading, start_line, end_line, "
                    "embedding, embedding_norm "
                    "FROM chunks"
                ).fetchall()

        if chunk_rows:
            # Track best chunk per file: (score, heading, start_line, end_line)
            best_chunk: dict[str, tuple[float, str | None, int | None, int | None]] = {}

            if HAS_NUMPY:
                n_chunks = len(chunk_rows)
                dims = len(q_emb)
                chunk_matrix = np.zeros((n_chunks, dims), dtype=np.float32)
                chunk_norms = np.zeros(n_chunks, dtype=np.float32)
                chunk_paths = []
                chunk_headings = []
                chunk_starts = []
                chunk_ends = []

                for i, (fp, ci, heading, sline, eline, blob, cnorm) in enumerate(chunk_rows):
                    if len(blob) == dims * 8:
                        chunk_matrix[i] = np.frombuffer(blob, dtype=np.float64).astype(np.float32)
                    elif len(blob) == dims * 2:
                        chunk_matrix[i] = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
                    else:
                        chunk_matrix[i] = np.frombuffer(blob, dtype=np.float32)
                    chunk_norms[i] = cnorm or 0.0
                    chunk_paths.append(fp)
                    chunk_headings.append(heading)
                    chunk_starts.append(sline)
                    chunk_ends.append(eline)

                q_vec = np.array(q_emb, dtype=np.float32)
                q_n = np.linalg.norm(q_vec)
                if q_n > 0:
                    dots = chunk_matrix @ q_vec
                    safe_norms = np.where(chunk_norms == 0, 1.0, chunk_norms)
                    cosines = dots / (safe_norms * q_n)
                    cosines = np.where(chunk_norms == 0, 0.0, cosines)

                    for i in range(n_chunks):
                        fp = chunk_paths[i]
                        score = float(cosines[i]) + 0.15 * keyword_score(query, fp)
                        if fp not in best_chunk or score > best_chunk[fp][0]:
                            best_chunk[fp] = (score, chunk_headings[i],
                                              chunk_starts[i], chunk_ends[i])
            else:
                q_n = norm(q_emb)
                for fp, ci, heading, sline, eline, blob, cnorm in chunk_rows:
                    try:
                        emb = unpack_embedding(blob)
                    except Exception:
                        continue
                    if q_n == 0 or (cnorm or 0) == 0:
                        cos = 0.0
                    else:
                        cos = dot(q_emb, emb) / (q_n * cnorm)
                    score = cos + 0.15 * keyword_score(query, fp)
                    if fp not in best_chunk or score > best_chunk[fp][0]:
                        best_chunk[fp] = (score, heading, sline, eline)

            # Merge: if chunk score > file score, promote and carry metadata
            file_data: dict[str, tuple[float, str | None, int | None, int | None]] = {}
            for item in semantic_results:
                file_data[item[1]] = (item[0], None, None, None)

            for fp, (chunk_score, heading, sline, eline) in best_chunk.items():
                current = file_data.get(fp, (0.0, None, None, None))[0]
                if chunk_score > current:
                    file_data[fp] = (chunk_score, heading, sline, eline)

            semantic_results = [
                (score, path, summary_map.get(path, ""), heading,
                 (sline, eline) if sline is not None else None)
                for path, (score, heading, sline, eline) in file_data.items()
            ]
            semantic_results.sort(key=lambda x: x[0], reverse=True)

    if mode == "semantic":
        if rerank:
            print("Re-ranking with LLM...", file=sys.stderr)
            t0 = time.time()
            reranked = rerank_results(query, semantic_results[:top_k * 4], conn, intent=intent)
            _explain(f"Re-ranking ({(time.time()-t0)*1000:.0f}ms)")
            results = reranked[:top_k]
            if explain:
                _explain(f"Final: {len(results)} results (semantic + rerank)")
                _print_explain(explain_log, results)
            conn.close()
            return results
        results = semantic_results[:top_k]
        if explain:
            _explain(f"Final: {len(results)} results (semantic only)")
            _print_explain(explain_log, results)
        conn.close()
        return results

    # Hybrid: fuse semantic + BM25 via RRF
    candidate_k = max(top_k * 4, 50)
    t0 = time.time()
    bm25_results = bm25_search(conn, query, path_filter, limit=candidate_k)
    _explain(f"BM25 search → {len(bm25_results)} results ({(time.time()-t0)*1000:.0f}ms)")

    if not bm25_results:
        if rerank:
            print("Re-ranking with LLM...", file=sys.stderr)
            t0 = time.time()
            reranked = rerank_results(query, semantic_results[:top_k * 4], conn, intent=intent)
            _explain(f"Re-ranking ({(time.time()-t0)*1000:.0f}ms)")
            results = reranked[:top_k]
            if explain:
                _explain(f"Final: {len(results)} results (semantic + rerank, no BM25)")
                _print_explain(explain_log, results)
            conn.close()
            return results
        results = semantic_results[:top_k]
        if explain:
            _explain(f"Final: {len(results)} results (semantic only, no BM25)")
            _print_explain(explain_log, results)
        conn.close()
        return results

    fused = rrf_fusion(
        semantic_results[:candidate_k],
        bm25_results,
        top_k=max(top_k, RERANK_TOP_N) if rerank else top_k,
    )
    _explain(f"RRF fusion → {len(fused)} candidates")

    # Community boost: nudge results sharing Leiden communities with query entities
    t0 = time.time()
    fused = community_boost(fused, conn, query)
    _explain(f"Community boost ({(time.time()-t0)*1000:.0f}ms)")

    if rerank:
        print("Re-ranking with LLM...", file=sys.stderr)
        t0 = time.time()
        reranked = rerank_results(query, fused, conn, intent=intent)
        _explain(f"Re-ranking ({(time.time()-t0)*1000:.0f}ms)")
        results = reranked[:top_k]
        if explain:
            _explain(f"Final: {len(results)} results (hybrid + rerank)")
            _print_explain(explain_log, results)
        conn.close()
        return results

    if explain:
        _explain(f"Final: {len(fused)} results (hybrid)")
        _print_explain(explain_log, fused)

    conn.close()
    return fused


# ---------------------------------------------------------------------------
# Catalyst index — pre-computed thematic questions for sub-ms entity lookup
# ---------------------------------------------------------------------------

def _catalyst_index_exists(conn: sqlite3.Connection) -> bool:
    """Check if the catalyst FTS5 table exists and has data."""
    try:
        count = conn.execute("SELECT COUNT(*) FROM catalysts").fetchone()[0]
        return count > 0
    except Exception:
        return False


def _get_top_catalyst_entities(conn: sqlite3.Connection, limit: int = 200) -> list:
    """Get entities ranked by mention count — most connected = most useful as catalysts."""
    try:
        return conn.execute("""
            SELECT name, type, COUNT(*) as mentions
            FROM entities
            GROUP BY name
            ORDER BY mentions DESC
            LIMIT ?
        """, (limit,)).fetchall()
    except Exception:
        return []


def _get_entity_context(conn: sqlite3.Connection, entity_name: str) -> tuple:
    """Get the notes and relations for an entity."""
    notes = conn.execute(
        "SELECT DISTINCT source_file FROM entities WHERE name = ?",
        (entity_name,)
    ).fetchall()
    relations = conn.execute(
        "SELECT source_entity, relation, target_entity FROM relations "
        "WHERE source_entity = ? OR target_entity = ? LIMIT 10",
        (entity_name, entity_name)
    ).fetchall()
    return [r[0] for r in notes], relations


def _generate_catalyst_questions(entity_name: str, entity_type: str,
                                  notes: list, relations: list) -> list[str]:
    """Generate 3-5 catalyst questions for an entity WITHOUT an LLM call.

    Derived from knowledge graph structure alone — instant, free, deterministic.
    """
    questions = []
    related = set()
    for src, rel, tgt in relations:
        other = tgt if src == entity_name else src
        related.add(other)

    # Q1: What is this?
    questions.append(f"What is {entity_name} and how does it work?")

    # Q2: What connects to it?
    if related:
        sample = list(related)[:3]
        questions.append(f"How does {entity_name} relate to {', '.join(sample)}?")

    # Q3: Type-specific
    if entity_type == "concept":
        questions.append(f"What are the key mechanisms behind {entity_name}?")
    elif entity_type == "technique":
        questions.append(f"How is {entity_name} applied in practice?")
    elif entity_type == "theory":
        questions.append(f"What evidence supports or challenges {entity_name}?")
    elif entity_type == "person":
        questions.append(f"What contributions did {entity_name} make?")
    else:
        questions.append(f"What are the implications of {entity_name}?")

    # Q4: Cross-domain (if notes span multiple domain prefixes)
    note_prefixes = set()
    for n in notes:
        parts = Path(n).stem.split('--')
        if len(parts) >= 2:
            note_prefixes.add(parts[0])
    if len(note_prefixes) >= 2:
        questions.append(
            f"How does {entity_name} bridge {' and '.join(list(note_prefixes)[:3])}?"
        )

    # Q5: Coverage
    if len(notes) > 3:
        questions.append(
            f"What are the most important aspects of {entity_name} "
            f"across {len(notes)} vault notes?"
        )

    return questions


def build_catalyst_index(conn: sqlite3.Connection, limit: int = 200,
                          verbose: bool = False) -> int:
    """Build (or rebuild) the catalyst FTS5 table in the given DB connection.

    Returns the number of entities indexed.
    """
    # Check graph tables exist
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "entities" not in tables or "relations" not in tables:
        if verbose:
            print("  No entity/relation tables found — skipping catalyst index", file=sys.stderr)
        return 0

    entities = _get_top_catalyst_entities(conn, limit)
    if not entities:
        if verbose:
            print("  No entities found — skipping catalyst index", file=sys.stderr)
        return 0

    # Drop and recreate tables
    conn.execute("DROP TABLE IF EXISTS catalysts_fts")
    conn.execute("DROP TABLE IF EXISTS catalysts")
    conn.execute("""
        CREATE TABLE catalysts (
            entity TEXT NOT NULL,
            type TEXT,
            mentions INTEGER,
            questions TEXT,
            notes TEXT
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE catalysts_fts USING fts5(
            entity, questions, content=catalysts, content_rowid=rowid
        )
    """)

    total_questions = 0
    for name, etype, mentions in entities:
        notes, relations = _get_entity_context(conn, name)
        questions = _generate_catalyst_questions(name, etype or "concept", notes, relations)
        q_text = " ".join(questions)
        n_text = " ".join(notes[:5])
        conn.execute(
            "INSERT INTO catalysts (entity, type, mentions, questions, notes) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, etype, mentions, q_text, n_text)
        )
        conn.execute(
            "INSERT INTO catalysts_fts (rowid, entity, questions) "
            "VALUES (last_insert_rowid(), ?, ?)",
            (name, q_text)
        )
        total_questions += len(questions)

    conn.commit()
    return len(entities)


def catalyst_search(conn: sqlite3.Connection, query: str,
                     top: int = 5) -> list[dict]:
    """Sub-millisecond catalyst lookup via FTS5.

    Returns a list of dicts with entity, type, mentions, questions, notes, elapsed_ms.
    """
    if not _catalyst_index_exists(conn):
        return []

    words = [w for w in query.split() if len(w) > 2]
    if not words:
        return []

    fts_query = " OR ".join(words)
    t0 = time.time()
    try:
        rows = conn.execute("""
            SELECT c.entity, c.type, c.mentions, c.questions, c.notes, rank
            FROM catalysts_fts
            JOIN catalysts c ON c.rowid = catalysts_fts.rowid
            WHERE catalysts_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (fts_query, top)).fetchall()
    except Exception:
        return []
    elapsed = (time.time() - t0) * 1000

    results = []
    for entity, etype, mentions, questions, notes, rank in rows:
        results.append({
            "entity": entity,
            "type": etype,
            "mentions": mentions,
            "questions": [q.strip() + "?" for q in questions.split("?") if q.strip()],
            "notes": [n for n in (notes or "").split() if n],
            "elapsed_ms": elapsed,
        })
    return results


def _print_catalyst_results(results: list[dict], query: str) -> None:
    """Print catalyst search results to stdout."""
    if not results:
        print(f"No catalyst results for: {query}")
        return
    elapsed = results[0].get("elapsed_ms", 0) if results else 0
    print(f"Catalyst results for: {query}  ({elapsed:.2f}ms)\n")
    for r in results:
        print(f"  {r['entity']} ({r['type']}, {r['mentions']} mentions)")
        for q in r['questions'][:3]:
            print(f"    - {q}")
        print()


# ---------------------------------------------------------------------------
# Community boost — Leiden community co-membership scoring
# ---------------------------------------------------------------------------

def community_boost(
    results: list[tuple],
    conn: "sqlite3.Connection",
    query: str,
    boost_weight: float = 0.005,
) -> list[tuple]:
    """Boost results whose source files share Leiden communities with query entities.

    Returns re-sorted results with community co-membership bonus applied.
    The boost is small (default 0.005 per community overlap) to nudge
    rather than dominate the ranking.
    """
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if "communities" not in tables or "entities" not in tables:
            return results

        # Find entities matching the query
        query_lower = query.lower().strip()
        query_words = [w for w in query_lower.split() if len(w) > 3]

        matched = set()
        # Exact phrase match
        for r in conn.execute(
            "SELECT DISTINCT name FROM entities WHERE lower(name) LIKE ?",
            (f"%{query_lower}%",)
        ).fetchall():
            matched.add(r[0])
        # Word matches
        for word in query_words:
            for r in conn.execute(
                "SELECT DISTINCT name FROM entities WHERE lower(name) LIKE ?",
                (f"%{word}%",)
            ).fetchall():
                ename = r[0].lower()
                if word in ename.split() or ename.startswith(word) or ename.endswith(word):
                    matched.add(r[0])

        if not matched:
            return results

        # Get communities for matched entities
        query_communities = set()
        for ent in matched:
            row = conn.execute(
                "SELECT community_id FROM communities WHERE entity = ?", (ent,)
            ).fetchone()
            if row:
                query_communities.add(row[0])

        if not query_communities:
            return results

        # For each result file, check if its entities share communities
        boosted = []
        for item in results:
            score, path = item[0], item[1]
            fname = os.path.basename(path)

            # Find entities from this file
            file_entities = conn.execute(
                "SELECT DISTINCT name FROM entities WHERE source_file = ?",
                (fname,)
            ).fetchall()

            overlap = 0
            for (ent,) in file_entities:
                row = conn.execute(
                    "SELECT community_id FROM communities WHERE entity = ?", (ent,)
                ).fetchone()
                if row and row[0] in query_communities:
                    overlap += 1

            new_score = score + overlap * boost_weight
            boosted.append((new_score, *item[1:]))

        boosted.sort(key=lambda x: x[0], reverse=True)
        return boosted
    except Exception:
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# Graph context — entity/relation lookup from vault-graph tables
# ---------------------------------------------------------------------------

# Relation normalization (matches vault-graph.py)
_RELATION_MAP = {
    "builds_on": "builds_on", "based_on": "builds_on", "is_based_on": "builds_on",
    "derives_from": "builds_on", "derived_from": "builds_on", "draws_from": "builds_on",
    "inspired_by": "builds_on", "influenced_by": "builds_on",
    "contradicts": "contradicts", "contrasts_with": "contradicts", "competes_with": "contradicts",
    "differs_from": "contradicts", "refutes": "contradicts",
    "applies_to": "applies_to", "applies": "applies_to", "used_in": "applies_to",
    "used_for": "applies_to", "used_by": "applies_to",
    "implements": "implements", "implemented_by": "implements", "formalizes": "implements",
    "extends": "extends", "generalizes": "extends", "variant_of": "extends",
    "part_of": "part_of", "is_part_of": "part_of", "includes": "part_of",
    "contains": "part_of", "comprises": "part_of", "component_of": "part_of",
    "uses": "uses", "employs": "uses", "requires": "uses", "depends_on": "uses",
    "enables": "enables", "facilitates": "enables", "supports": "enables",
    "leads_to": "enables", "drives": "enables",
    "causes": "causes", "triggers": "causes", "affects": "causes",
    "modulates": "causes", "inhibits": "causes",
    "explains": "explains", "describes": "explains", "defines": "explains",
    "predicts": "explains", "demonstrates": "explains",
    "developed_by": "developed_by", "created_by": "developed_by",
    "coined_by": "developed_by", "proposed_by": "developed_by",
    "type_of": "type_of", "is_type_of": "type_of", "is_a": "type_of",
    "instance_of": "type_of", "example_of": "type_of",
    "complements": "complements", "combines": "complements",
    "integrates_with": "complements", "bridges": "complements",
}

def _norm_rel(r: str) -> str:
    return _RELATION_MAP.get(r, "relates_to")


def graph_context(db_path: Path, query: str, result_paths: list[str]) -> str | None:
    """Look up graph entities related to the query and result files.

    Returns a formatted string of entity connections, or None if no graph
    tables exist or no entities match.
    """
    try:
        conn = sqlite3.connect(str(db_path))
        # Check if graph tables exist
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if "entities" not in tables or "relations" not in tables:
            conn.close()
            return None

        # Find entities matching the query
        query_lower = query.lower().strip()
        query_words = [w for w in query_lower.split() if len(w) > 2]
        matched_entities = {}  # entity -> relevance score

        # 1. Exact/full phrase match (highest priority)
        rows = conn.execute(
            "SELECT DISTINCT name FROM entities WHERE name LIKE ?",
            (f"%{query_lower}%",)
        ).fetchall()
        for r in rows:
            matched_entities[r[0]] = matched_entities.get(r[0], 0) + 10

        # 2. Individual word matches (only if word is 4+ chars to reduce noise)
        for word in query_words:
            if len(word) < 4:
                continue
            rows = conn.execute(
                "SELECT DISTINCT name FROM entities WHERE name LIKE ?",
                (f"%{word}%",)
            ).fetchall()
            for r in rows:
                # Only count if entity name actually contains the word as a
                # meaningful match (not just a substring of a longer word)
                ename = r[0]
                if word in ename.split() or ename.startswith(word) or ename.endswith(word):
                    matched_entities[ename] = matched_entities.get(ename, 0) + 1

        # Also find entities from top result files (low weight — just for context)
        result_filenames = [os.path.basename(p) for p in result_paths[:3]]
        for fname in result_filenames:
            rows = conn.execute(
                "SELECT DISTINCT name FROM entities WHERE source_file = ?",
                (fname,)
            ).fetchall()
            for r in rows:
                # Only add if already matched by query (don't introduce new entities)
                if r[0] in matched_entities:
                    matched_entities[r[0]] = matched_entities.get(r[0], 0) + 1

        if not matched_entities:
            conn.close()
            return None

        # Rank entities: relevance for sorting, raw connection count for display
        entity_info = {}  # {name: (sort_score, raw_connections)}
        for ent, relevance in matched_entities.items():
            count = conn.execute(
                "SELECT COUNT(*) FROM (SELECT id FROM relations WHERE source_entity=? "
                "UNION ALL SELECT id FROM relations WHERE target_entity=?)",
                (ent, ent)
            ).fetchone()[0]
            entity_info[ent] = (relevance * max(count, 1), count)

        top_entities = sorted(entity_info, key=lambda e: entity_info[e][0], reverse=True)[:6]

        lines = []
        for ent in top_entities:
            # Get type
            etype = conn.execute(
                "SELECT type FROM entities WHERE name = ? LIMIT 1", (ent,)
            ).fetchone()
            etype = etype[0] if etype else "concept"

            raw_connections = entity_info[ent][1]
            lines.append(f"  {ent} ({etype}, {raw_connections} connections)")

            # Get top relations (outgoing)
            outgoing = conn.execute(
                "SELECT relation, target_entity FROM relations WHERE source_entity = ? LIMIT 5",
                (ent,)
            ).fetchall()
            for rel, target in outgoing:
                lines.append(f"    → {_norm_rel(rel)} → {target}")

            # Get top relations (incoming)
            incoming = conn.execute(
                "SELECT source_entity, relation FROM relations WHERE target_entity = ? LIMIT 3",
                (ent,)
            ).fetchall()
            for source, rel in incoming:
                lines.append(f"    ← {source} ← {_norm_rel(rel)}")

        # --- Community context (Leiden) ---
        community_lines = []
        if "communities" in tables:
            # Find communities for matched entities
            query_communities: dict[int, list[str]] = {}  # community_id -> [entity names]
            for ent in top_entities:
                row = conn.execute(
                    "SELECT community_id, community_size FROM communities WHERE entity = ?",
                    (ent,)
                ).fetchone()
                if row:
                    cid, csize = row
                    query_communities.setdefault(cid, []).append(ent)

            if query_communities:
                community_lines.append("")
                community_lines.append("── Community Context (Leiden) ──")
                for cid, ents in sorted(query_communities.items(),
                                         key=lambda x: len(x[1]), reverse=True)[:3]:
                    # Get community meta for label
                    meta_row = conn.execute(
                        "SELECT top_entities, density FROM community_meta WHERE community_id = ?",
                        (cid,)
                    ).fetchone()
                    label = ""
                    if meta_row and meta_row[0]:
                        try:
                            top_ents = json.loads(meta_row[0])[:5]
                        except (json.JSONDecodeError, TypeError):
                            top_ents = [e.strip() for e in meta_row[0].split(",")[:5]]
                        label = f" [{', '.join(top_ents)}]"

                    community_lines.append(f"  Community {cid}{label}")
                    community_lines.append(f"    Members in query: {', '.join(ents)}")

                    # Find sibling entities in same community (not already shown)
                    shown = set(top_entities) | set(ents)
                    siblings = conn.execute(
                        "SELECT entity FROM communities WHERE community_id = ? "
                        "AND entity NOT IN ({}) ORDER BY RANDOM() LIMIT 5".format(
                            ",".join("?" * len(shown))
                        ),
                        (cid, *shown)
                    ).fetchall()
                    if siblings:
                        sib_names = [s[0] for s in siblings]
                        community_lines.append(f"    Related: {', '.join(sib_names)}")

        conn.close()

        if not lines and not community_lines:
            return None

        output_parts = []
        if lines:
            output_parts.append("── Graph Context ──\n" + "\n".join(lines))
        if community_lines:
            output_parts.append("\n".join(community_lines))

        return "\n".join(output_parts) if output_parts else None
    except Exception:
        return None


# ---------------------------------------------------------------------------

def _cmd_init(root: Path, db_path: Path, catalyst_limit: int = 200) -> None:
    """Build catalysts from the existing index (FTS5 + graph must already exist)."""
    if not db_path.exists():
        print(f"Error: No index found at {db_path}", file=sys.stderr)
        print(f"Run vault-index.py {root} first to build the FTS5 index.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))

    # Count what's already there
    try:
        file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    except Exception:
        file_count = 0

    print(f"Index: {db_path}")
    print(f"Files indexed: {file_count}")

    # Build catalyst index
    t0 = time.time()
    n = build_catalyst_index(conn, limit=catalyst_limit, verbose=True)
    elapsed = time.time() - t0

    if n > 0:
        total_q = conn.execute("SELECT COUNT(*) FROM catalysts").fetchone()[0]
        print(f"Catalyst index: {n} entities, {total_q} rows ({elapsed:.1f}s)")
    else:
        print("Catalyst index: skipped (no entity/relation tables — run vault-graph.py first)")

    conn.close()


def _cmd_status(db_path: Path) -> None:
    """Show index statistics."""
    if not db_path.exists():
        print(f"No index at {db_path}")
        print("Run: vault-search init <vault-root>")
        return

    conn = sqlite3.connect(str(db_path))
    stats = {}

    for table, label in [
        ("files", "files indexed"),
        ("chunks", "chunks"),
        ("entities", "entities"),
        ("relations", "relations"),
        ("communities", "community memberships"),
        ("catalysts", "catalyst entries"),
    ]:
        try:
            stats[label] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception:
            stats[label] = None

    conn.close()

    print(f"Index: {db_path}")
    print(f"Size:  {db_path.stat().st_size / 1024 / 1024:.1f} MB")
    for label, count in stats.items():
        if count is not None:
            print(f"  {label}: {count:,}")
        else:
            print(f"  {label}: (table not present)")


def _run_search_and_print(args, db_path: Path) -> None:
    """Shared search + output logic for the main search path."""
    if args.iterate:
        results = iterative_search(
            args.query,
            db_path=db_path,
            top_k=args.top,
            path_filter=args.path,
            mode=args.mode,
            expand=args.expand,
            rerank=args.rerank,
            intent=args.intent,
            explain=args.explain,
        )
    else:
        results = search(
            args.query,
            db_path=db_path,
            top_k=args.top,
            path_filter=args.path,
            mode=args.mode,
            expand=args.expand,
            rerank=args.rerank,
            intent=args.intent,
            explain=args.explain,
        )

    # Low-confidence detection (skip in --explain mode to avoid noise duplication)
    # RRF scores (~0.016-0.035) are much smaller than cosine similarity scores (0-1).
    # Use mode-appropriate thresholds:
    #   hybrid/bm25: RRF scores, flag if spread < 0.003 (top results indistinguishable)
    #   semantic:    cosine+keyword scores, flag if top < 0.15 or spread < 0.05
    if results and not args.explain:
        top_score = results[0][0]
        low_conf = False
        if args.mode == "semantic":
            if top_score < 0.15:
                low_conf = True
            elif len(results) >= 3:
                top3 = [r[0] for r in results[:3]]
                spread = max(top3) - min(top3)
                if spread < 0.05:
                    low_conf = True
        else:
            if len(results) >= 3:
                top3 = [r[0] for r in results[:3]]
                spread = max(top3) - min(top3)
                if spread < 0.003:
                    low_conf = True
        if low_conf:
            print("[LOW CONFIDENCE] Top results may not be relevant — "
                  "consider refining your query", file=sys.stderr)

    # Optionally append catalyst context to results
    if getattr(args, 'fast', False) or getattr(args, 'no_embeddings', False):
        # Catalyst results already printed or will be shown in --fast mode
        pass

    result_paths = [r[1] for r in results]

    if args.json:
        output = []
        for result in results:
            score, path, summary = result[0], result[1], result[2]
            c_heading = result[3] if len(result) > 3 else None
            c_lines = result[4] if len(result) > 4 else None
            entry = {
                "score": round(score, 4),
                "path": path,
                "summary": summary,
            }
            if c_heading:
                entry["chunk_heading"] = c_heading
            if c_lines:
                entry["chunk_lines"] = list(c_lines)
            output.append(entry)
        if not args.no_graph:
            gc = graph_context(db_path, args.query, result_paths)
            if gc:
                print(json.dumps({"results": output, "graph_context": gc}, indent=2))
            else:
                print(json.dumps(output, indent=2))
        else:
            print(json.dumps(output, indent=2))
    else:
        for result in results:
            score, path, summary = result[0], result[1], result[2]
            c_heading = result[3] if len(result) > 3 else None
            c_lines = result[4] if len(result) > 4 else None
            short_summary = summary.replace("\n", " ").strip()
            if len(short_summary) > 100:
                short_summary = short_summary[:97] + "..."
            print(f"{score:.4f}  {path}")
            if c_heading:
                line_info = f" (lines {c_lines[0]}-{c_lines[1]})" if c_lines else ""
                print(f"        \u00a7 {c_heading}{line_info}")
            print(f"        {short_summary}")
            print()

        if not args.no_graph:
            gc = graph_context(db_path, args.query, result_paths)
            if gc:
                print(gc)

        # Catalyst context appended to BM25/no-embeddings results
        if getattr(args, 'no_embeddings', False) and db_path.exists():
            conn = sqlite3.connect(str(db_path))
            if _catalyst_index_exists(conn):
                cat_results = catalyst_search(conn, args.query, top=3)
                if cat_results:
                    elapsed = cat_results[0].get("elapsed_ms", 0)
                    print(f"\n── Catalyst Context ({elapsed:.2f}ms) ──")
                    for r in cat_results[:3]:
                        print(f"  {r['entity']} ({r['type']}, {r['mentions']} mentions)")
                        for q in r['questions'][:2]:
                            print(f"    - {q}")
            conn.close()


def main() -> None:
    _vault_root = str(Path(__file__).resolve().parent.parent)

    parser = argparse.ArgumentParser(
        description="Local semantic search over files indexed by vault-index.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  vault-search init [root]      Build catalyst index from existing FTS5/graph index
  vault-search status [root]    Show index statistics
  vault-search "query" [root]   Search (hybrid BM25 + embeddings, default)

Flags:
  --fast            Catalyst-only sub-ms entity lookup (no file search)
  --no-embeddings   BM25 + catalysts only, no Ollama required
  --top N           Number of results (default: 5)
  --path PREFIX     Filter results to files under this path
  --mode            hybrid|semantic|bm25 (default: hybrid)
  --expand          HyDE query expansion
  --rerank          LLM re-ranking
  --intent DOMAIN   Domain context to disambiguate queries
  --iterate         Two-pass iterative retrieval
  --explain         Show scoring details
  --no-graph        Skip graph context
  --no-cache        Bypass embedding disk cache
  --json            Output as JSON
"""
    )
    parser.add_argument(
        "query", type=str,
        help="Search query, or 'init' / 'status' subcommand"
    )
    parser.add_argument(
        "root", nargs="?", type=str, default=_vault_root,
        help="Root directory that was indexed (default: vault root)"
    )
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--mode", choices=["hybrid", "semantic", "bm25"], default="hybrid"
    )
    parser.add_argument("--expand", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--intent", type=str, default=None)
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--no-graph", action="store_true", dest="no_graph")
    parser.add_argument("--no-cache", action="store_true", dest="no_cache")
    parser.add_argument("--iterate", action="store_true")
    # New flags
    parser.add_argument(
        "--fast", action="store_true",
        help="Catalyst-only sub-ms entity lookup (no file search, no Ollama)"
    )
    parser.add_argument(
        "--no-embeddings", action="store_true", dest="no_embeddings",
        help="BM25 + catalysts only — skip all Ollama embedding calls"
    )
    parser.add_argument(
        "--catalyst-limit", type=int, default=200, dest="catalyst_limit",
        help="Number of top entities to index as catalysts (default: 200)"
    )

    args = parser.parse_args()

    # Apply flags to module-level globals
    if args.no_cache:
        global _disk_cache_enabled
        _disk_cache_enabled = False

    if args.no_embeddings or args.fast:
        global _NO_EMBEDDINGS
        _NO_EMBEDDINGS = True

    # Resolve root + DB path
    root = Path(args.root).resolve()
    if args.db:
        db_path = Path(args.db)
    else:
        db_path = db_path_for_root(root)

    # --- Subcommands ---
    if args.query == "init":
        _cmd_init(root, db_path, catalyst_limit=args.catalyst_limit)
        return

    if args.query == "status":
        _cmd_status(db_path)
        return

    # --- --fast: catalyst-only path ---
    if args.fast:
        if not db_path.exists():
            print(f"Error: No index at {db_path}. Run: vault-search init {root}",
                  file=sys.stderr)
            sys.exit(1)
        conn = sqlite3.connect(str(db_path))
        if not _catalyst_index_exists(conn):
            print("No catalyst index found. Run: vault-search init", file=sys.stderr)
            conn.close()
            sys.exit(1)
        results = catalyst_search(conn, args.query, top=args.top)
        conn.close()
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            _print_catalyst_results(results, args.query)
        return

    # --- Standard search path ---
    _run_search_and_print(args, db_path)


if __name__ == "__main__":
    main()
