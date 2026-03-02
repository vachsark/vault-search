#!/usr/bin/env python3
# NOTE: For best performance, install numpy: pip install numpy
# With numpy: ~85ms/query. Without numpy: ~148ms/query (pure Python fallback).
"""
vault-search.py — Local semantic search over files indexed by vault-index.py.

Hybrid search (BM25 + embeddings + Reciprocal Rank Fusion) with optional
LLM re-ranking and HyDE query expansion. Runs entirely on your machine
via Ollama — no API cost, no data leaves your network.

Usage:
    python3 vault-search.py "authentication middleware"
    python3 vault-search.py "convex schema design" --top 10
    python3 vault-search.py "React hooks" --path Projects/
    python3 vault-search.py "getTenantId" --mode bm25
    python3 vault-search.py "auth patterns" --rerank        # LLM re-ranking
    python3 vault-search.py "auth patterns" --expand --rerank  # full pipeline

Environment variables:
    OLLAMA_BASE       Ollama API URL (default: http://localhost:11434)
    EMBED_MODEL       Embedding model (default: qwen3-embedding:0.6b)
    EXPAND_MODEL      HyDE expansion model (default: qwen3:8b)
    RERANK_MODEL      Re-ranking model (default: qwen3:8b)
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
import urllib.request
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Config (all overridable via env vars)
# ---------------------------------------------------------------------------

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:0.6b")
EXPAND_MODEL = os.environ.get("EXPAND_MODEL", "qwen3:8b")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "qwen3:8b")

RERANK_TOP_N = 20   # Candidates to re-rank from retrieval stage
RERANK_BLEND_K = 5  # Position-aware blending: top-k results trust retrieval more


def db_path_for_root(root: Path) -> Path:
    """Deterministic DB path per root directory. Matches vault-index.py."""
    custom = os.environ.get("VAULT_SEARCH_DB")
    if custom:
        return Path(custom)
    root_hash = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]
    return Path.home() / ".local" / "share" / "vault-search" / f"{root_hash}.db"


# ---------------------------------------------------------------------------
# Math — NumPy accelerated with pure-Python fallback
# ---------------------------------------------------------------------------

def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def unpack_embedding(blob: bytes) -> list[float]:
    """Unpack a binary blob back to a list of floats.

    Auto-detects float32 vs float64 format based on blob size.
    """
    if len(blob) % 8 == 0 and len(blob) // 8 == 1024:
        n = len(blob) // 8
        return list(struct.unpack(f'{n}d', blob))
    else:
        n = len(blob) // 4
        return list(struct.unpack(f'{n}f', blob))


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
) -> list[tuple[float, str, str]]:
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
    return [(float(scores[idx]), paths[idx], summaries[idx]) for idx in order]


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


def ollama_embed(text: str) -> list[float]:
    if text in _embed_cache:
        return _embed_cache[text]
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
    return emb


def hyde_expand(query: str) -> str:
    """HyDE: generate a hypothetical document to improve embedding alignment.

    Instead of embedding the raw query, we ask the local LLM to generate a
    short passage that WOULD answer the query, then embed that. Document-to-
    document similarity is more reliable than question-to-document.
    """
    prompt = (
        f"Write a short technical passage (2-3 sentences or a brief code snippet) "
        f"that directly answers or demonstrates: {query}\n"
        f"Be specific. Use technical terminology. No preamble."
    )
    payload = json.dumps({
        "model": EXPAND_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.3, "num_predict": 120},
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


def rerank_score(query: str, doc_snippet: str) -> float:
    """Score a single document's relevance to a query (0.0-1.0).

    Uses the local LLM as a cross-encoder: given query + document text,
    output a relevance score 0-10 which we normalize to 0.0-1.0.
    ~120ms per call when model is warm.
    """
    cache_key = f"{query}||{doc_snippet[:100]}"
    if cache_key in _rerank_cache:
        return _rerank_cache[cache_key]

    prompt = (
        "Rate relevance 0-10. Only output the number.\n"
        f"Query: {query}\n"
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
        score = min(max(raw / 10.0, 0.0), 1.0)
    except Exception:
        score = 0.5

    _rerank_cache[cache_key] = score
    return score


def rerank_results(
    query: str,
    results: list[tuple[float, str, str]],
    conn: sqlite3.Connection,
    top_n: int = RERANK_TOP_N,
) -> list[tuple[float, str, str]]:
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

    paths = [path for _, path, _ in candidates]
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
        for _, path, summary in candidates:
            if path not in content_map:
                content_map[path] = (summary or path)[:400]

    # Normalize retrieval scores to 0-1 before blending.
    # RRF scores are tiny (~0.016-0.033), reranker scores are 0.0-1.0.
    max_retrieval = max(s for s, _, _ in candidates) if candidates else 1.0
    if max_retrieval == 0:
        max_retrieval = 1.0

    scored: list[tuple[float, float, int, str, str]] = []
    for rank, (retrieval_score, path, summary) in enumerate(candidates):
        snippet = content_map.get(path, summary[:400] if summary else path)
        rr_score = rerank_score(query, snippet)
        norm_retrieval = retrieval_score / max_retrieval

        if rank < RERANK_BLEND_K:
            blended = 0.6 * norm_retrieval + 0.4 * rr_score
        else:
            blended = 0.4 * norm_retrieval + 0.6 * rr_score

        scored.append((blended, rr_score, rank, path, summary))

    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [(blended, path, summary) for blended, _, _, path, summary in scored]

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
        base += "AND path LIKE ? "
    base += "ORDER BY score DESC LIMIT ?"

    def run(fts_expr: str) -> list[tuple[str, float]]:
        params: list = [fts_expr]
        if path_filter:
            params.append(f"{path_filter}%")
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
    semantic: list[tuple[float, str, str]],
    bm25: list[tuple[str, float]],
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[float, str, str]]:
    """Reciprocal Rank Fusion: merge semantic and BM25 ranked lists.

    Each document gets a score of sum(1 / (k + rank_i)) across all lists.
    k=60 is the standard default (Cormack et al. 2009).
    """
    all_docs: dict[str, dict] = {}

    for rank, (score, path, summary) in enumerate(semantic):
        all_docs.setdefault(path, {"summary": summary, "rrf": 0.0})
        all_docs[path]["rrf"] += 1.0 / (k + rank + 1)

    for rank, (path, score) in enumerate(bm25):
        if path not in all_docs:
            all_docs[path] = {"summary": "", "rrf": 0.0}
        all_docs[path]["rrf"] += 1.0 / (k + rank + 1)

    fused = [(v["rrf"], path, v["summary"]) for path, v in all_docs.items()]
    fused.sort(reverse=True)
    return fused[:top_k]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    query: str,
    db_path: Path,
    top_k: int = 5,
    path_filter: str | None = None,
    mode: str = "hybrid",
    expand: bool = False,
    rerank: bool = False,
) -> list[tuple[float, str, str]]:
    """Search the index. Returns list of (score, path, summary).

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
    """
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

    # BM25-only path
    if mode == "bm25":
        words = [w for w in re.findall(r'\w+', query.lower()) if len(w) >= 3]
        if not words:
            print(f"No BM25 results: all query terms are shorter than 3 characters. "
                  f"Try --mode semantic for short queries.", file=sys.stderr)
            conn.close()
            return []
        bm25_results = bm25_search(conn, query, path_filter, limit=top_k * 3)
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
        conn.close()
        return [(score, path, summary_map.get(path, ""))
                for path, score in bm25_results[:top_k]]

    # Get query embedding
    embed_query = query
    if expand:
        print("Expanding query with HyDE...", file=sys.stderr)
        embed_query = hyde_expand(query)

    try:
        q_emb = ollama_embed(embed_query)
    except Exception as e:
        print(f"Error: Cannot reach Ollama at {OLLAMA_BASE}: {e}", file=sys.stderr)
        print("Make sure Ollama is running: ollama serve", file=sys.stderr)
        sys.exit(1)

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
            "SELECT path, embedding, embedding_norm, summary FROM files WHERE path LIKE ?",
            (f"{path_filter}%",)
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

    summary_map = {path: (summary or "") for path, _, _, summary in rows}

    # Score all files
    if HAS_NUMPY and all(isinstance(r[1], bytes) for r in rows[:10]):
        semantic_results = vectorized_search(rows, q_emb, query, path_filter)
    else:
        q_norm = norm(q_emb)
        semantic_results: list[tuple[float, str, str]] = []
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
            semantic_results.append((score, path, summary or ""))
        semantic_results.sort(key=lambda x: x[0], reverse=True)

    # Chunk-level search
    if has_chunks:
        if path_filter:
            chunk_rows = conn.execute(
                "SELECT file_path, chunk_index, heading, embedding, embedding_norm "
                "FROM chunks WHERE file_path LIKE ?",
                (f"{path_filter}%",)
            ).fetchall()
        else:
            chunk_rows = conn.execute(
                "SELECT file_path, chunk_index, heading, embedding, embedding_norm "
                "FROM chunks"
            ).fetchall()

        if chunk_rows:
            best_chunk_score: dict[str, tuple[float, str | None]] = {}

            if HAS_NUMPY:
                n_chunks = len(chunk_rows)
                dims = len(q_emb)
                chunk_matrix = np.zeros((n_chunks, dims), dtype=np.float32)
                chunk_norms = np.zeros(n_chunks, dtype=np.float32)
                chunk_paths = []
                chunk_headings = []

                for i, (fp, ci, heading, blob, cnorm) in enumerate(chunk_rows):
                    if len(blob) == dims * 8:
                        chunk_matrix[i] = np.frombuffer(blob, dtype=np.float64).astype(np.float32)
                    else:
                        chunk_matrix[i] = np.frombuffer(blob, dtype=np.float32)
                    chunk_norms[i] = cnorm or 0.0
                    chunk_paths.append(fp)
                    chunk_headings.append(heading)

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
                        if fp not in best_chunk_score or score > best_chunk_score[fp][0]:
                            best_chunk_score[fp] = (score, chunk_headings[i])
            else:
                q_n = norm(q_emb)
                for fp, ci, heading, blob, cnorm in chunk_rows:
                    try:
                        emb = unpack_embedding(blob)
                    except Exception:
                        continue
                    if q_n == 0 or (cnorm or 0) == 0:
                        cos = 0.0
                    else:
                        cos = dot(q_emb, emb) / (q_n * cnorm)
                    score = cos + 0.15 * keyword_score(query, fp)
                    if fp not in best_chunk_score or score > best_chunk_score[fp][0]:
                        best_chunk_score[fp] = (score, heading)

            file_scores: dict[str, float] = {path: score for score, path, _ in semantic_results}
            for fp, (chunk_score, heading) in best_chunk_score.items():
                current = file_scores.get(fp, 0.0)
                if chunk_score > current:
                    file_scores[fp] = chunk_score

            semantic_results = [
                (score, path, summary_map.get(path, ""))
                for path, score in file_scores.items()
            ]
            semantic_results.sort(key=lambda x: x[0], reverse=True)

    if mode == "semantic":
        if rerank:
            print("Re-ranking with LLM...", file=sys.stderr)
            reranked = rerank_results(query, semantic_results[:top_k * 4], conn)
            conn.close()
            return reranked[:top_k]
        conn.close()
        return semantic_results[:top_k]

    # Hybrid: fuse semantic + BM25 via RRF
    candidate_k = max(top_k * 4, 50)
    bm25_results = bm25_search(conn, query, path_filter, limit=candidate_k)

    if not bm25_results:
        if rerank:
            print("Re-ranking with LLM...", file=sys.stderr)
            reranked = rerank_results(query, semantic_results[:top_k * 4], conn)
            conn.close()
            return reranked[:top_k]
        conn.close()
        return semantic_results[:top_k]

    fused = rrf_fusion(
        semantic_results[:candidate_k],
        bm25_results,
        top_k=max(top_k, RERANK_TOP_N) if rerank else top_k,
    )

    if rerank:
        print("Re-ranking with LLM...", file=sys.stderr)
        reranked = rerank_results(query, fused, conn)
        conn.close()
        return reranked[:top_k]

    conn.close()
    return fused


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local semantic search over files indexed by vault-index.py"
    )
    parser.add_argument(
        "query", type=str,
        help="Search query"
    )
    parser.add_argument(
        "root", nargs="?", type=str, default=".",
        help="Root directory that was indexed (default: current directory)"
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Number of results (default: 5)"
    )
    parser.add_argument(
        "--path", type=str, default=None,
        help="Filter results to files under this path prefix"
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="Database path (default: auto per root dir)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--mode", choices=["hybrid", "semantic", "bm25"], default="hybrid",
        help="Search mode (default: hybrid)"
    )
    parser.add_argument(
        "--expand", action="store_true",
        help="HyDE query expansion — generates a hypothetical document "
             "before embedding (adds ~1s, improves recall)"
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="Re-rank top candidates using LLM scoring (adds ~2-3s, "
             "significantly improves quality)"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if args.db:
        db_path = Path(args.db)
    else:
        db_path = db_path_for_root(root)

    results = search(
        args.query,
        db_path=db_path,
        top_k=args.top,
        path_filter=args.path,
        mode=args.mode,
        expand=args.expand,
        rerank=args.rerank,
    )

    if args.json:
        output = []
        for score, path, summary in results:
            output.append({
                "score": round(score, 4),
                "path": path,
                "summary": summary,
            })
        print(json.dumps(output, indent=2))
    else:
        for score, path, summary in results:
            short_summary = summary.replace("\n", " ").strip()
            if len(short_summary) > 100:
                short_summary = short_summary[:97] + "..."
            print(f"{score:.4f}  {path}")
            print(f"        {short_summary}")
            print()


if __name__ == "__main__":
    main()
