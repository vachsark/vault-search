#!/usr/bin/env python3
"""
vault-index.py — Build a local semantic search index for any directory.

Walks a directory of markdown/code files, generates embeddings via Ollama,
produces short summaries, and stores everything in a local SQLite database.
Supports incremental indexing (only re-embeds changed files).

Usage:
    python3 vault-index.py /path/to/your/docs
    python3 vault-index.py ~/notes --path Projects/  # index subdirectory
    python3 vault-index.py ~/notes --no-summary       # fast mode
    python3 vault-index.py ~/notes --test              # index 5 files

Environment variables:
    OLLAMA_BASE       Ollama API URL (default: http://localhost:11434)
    EMBED_MODEL       Embedding model (default: qwen3-embedding:0.6b)
    SUMMARY_MODEL     Summary model (default: qwen2.5-coder:7b)
    VAULT_SEARCH_DB   Database path (default: ~/.local/share/vault-search/<hash>.db)
"""

import argparse
import hashlib
import json
import math
import os
import sqlite3
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Config (all overridable via env vars or CLI)
# ---------------------------------------------------------------------------

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:0.6b")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "qwen2.5-coder:7b")
MAX_CONTENT_CHARS = 8000
FTS_CONTENT_CHARS = 50_000

# Chunking config
MAX_CHUNK_CHARS = 2000
MIN_CHUNK_CHARS = 200
OVERLAP_CHARS = 300

SKIP_DIRS = {
    "node_modules", ".git", ".next", "dist", "build",
    ".obsidian", "__pycache__", ".claude", ".trash",
    ".venv", "venv", "env", ".env",
}

INDEXABLE_EXTENSIONS = {
    ".md", ".txt", ".rst",                          # docs
    ".ts", ".tsx", ".js", ".jsx",                    # web
    ".py", ".sh", ".bash",                           # scripting
    ".rs", ".go", ".rb", ".java", ".kt", ".swift",  # compiled
    ".c", ".cpp", ".h", ".hpp",                      # systems
    ".yaml", ".yml", ".toml", ".json", ".ini",       # config
}

SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico", ".webp",
    ".pdf", ".docx", ".xlsx", ".pptx",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".lock", ".woff", ".woff2", ".ttf", ".eot",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".db", ".sqlite", ".sqlite3",
    ".min.js", ".min.css",
    ".map", ".excalidraw",
}


def db_path_for_root(root: Path) -> Path:
    """Deterministic DB path per root directory. Allows multiple indexes."""
    custom = os.environ.get("VAULT_SEARCH_DB")
    if custom:
        return Path(custom)
    root_hash = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]
    return Path.home() / ".local" / "share" / "vault-search" / f"{root_hash}.db"


# ---------------------------------------------------------------------------
# Embedding binary format
# ---------------------------------------------------------------------------

def pack_embedding(emb: list[float]) -> bytes:
    return struct.pack(f'{len(emb)}f', *emb)


def unpack_embedding(blob: bytes) -> list[float]:
    if len(blob) % 8 == 0 and len(blob) // 8 == 1024:
        n = len(blob) // 8
        return list(struct.unpack(f'{n}d', blob))
    else:
        n = len(blob) // 4
        return list(struct.unpack(f'{n}f', blob))


def embedding_norm(emb: list[float]) -> float:
    return math.sqrt(sum(x * x for x in emb))


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            path            TEXT PRIMARY KEY,
            content_hash    TEXT,
            embedding       BLOB,
            embedding_norm  REAL,
            summary         TEXT,
            content         TEXT,
            indexed_at      TEXT
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
            path UNINDEXED,
            content,
            tokenize='trigram'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path       TEXT NOT NULL,
            chunk_index     INTEGER NOT NULL,
            heading         TEXT,
            start_line      INTEGER,
            end_line        INTEGER,
            embedding       BLOB,
            embedding_norm  REAL,
            UNIQUE(file_path, chunk_index)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path)")
    conn.commit()
    migrate_db(conn)
    return conn


def migrate_db(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(files)").fetchall()}
    if "embedding_norm" not in columns:
        conn.execute("ALTER TABLE files ADD COLUMN embedding_norm REAL")
        conn.commit()
    if "content" not in columns:
        conn.execute("ALTER TABLE files ADD COLUMN content TEXT")
        conn.commit()

    row = conn.execute(
        "SELECT COUNT(*) FROM files WHERE typeof(embedding) = 'text'"
    ).fetchone()
    if row[0] == 0:
        return

    print(f"Migrating {row[0]} embeddings from JSON to binary format...")
    rows = conn.execute(
        "SELECT path, embedding FROM files WHERE typeof(embedding) = 'text'"
    ).fetchall()
    converted = 0
    for path, emb_json in rows:
        try:
            emb = json.loads(emb_json)
            blob = pack_embedding(emb)
            n = embedding_norm(emb)
            conn.execute(
                "UPDATE files SET embedding = ?, embedding_norm = ? WHERE path = ?",
                (blob, n, path)
            )
            converted += 1
        except Exception as e:
            print(f"  Warning: {path}: {e}", file=sys.stderr)
    conn.commit()
    print(f"Migrated {converted}/{row[0]} embeddings to binary format")


def migrate_fts(conn: sqlite3.Connection, vault_root: Path) -> None:
    fts_count = conn.execute("SELECT COUNT(*) FROM files_fts").fetchone()[0]
    files_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    if files_count == 0 or fts_count >= int(files_count * 0.9):
        return

    missing = files_count - fts_count
    print(f"Building FTS5 full-text index ({missing} files to index)...")
    rows = conn.execute("SELECT path FROM files").fetchall()
    done = 0
    for (rel,) in rows:
        fpath = vault_root / rel
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        conn.execute("DELETE FROM files_fts WHERE path = ?", (rel,))
        conn.execute(
            "INSERT INTO files_fts(path, content) VALUES (?, ?)",
            (rel, text[:FTS_CONTENT_CHARS])
        )
        done += 1
    conn.commit()
    print(f"FTS5 index ready ({done} files)")


def get_stored_hash(conn: sqlite3.Connection, path: str) -> str | None:
    row = conn.execute(
        "SELECT content_hash FROM files WHERE path = ?", (path,)
    ).fetchone()
    return row[0] if row else None


def upsert_file(conn: sqlite3.Connection, path: str, content_hash: str,
                 embedding: list[float], summary: str, content: str = "") -> None:
    blob = pack_embedding(embedding)
    emb_norm = embedding_norm(embedding)
    conn.execute("""
        INSERT INTO files (path, content_hash, embedding, embedding_norm, summary, content, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(path) DO UPDATE SET
            content_hash   = excluded.content_hash,
            embedding      = excluded.embedding,
            embedding_norm = excluded.embedding_norm,
            summary        = excluded.summary,
            content        = excluded.content,
            indexed_at     = excluded.indexed_at
    """, (path, content_hash, blob, emb_norm, summary, content[:FTS_CONTENT_CHARS]))
    if content:
        conn.execute("DELETE FROM files_fts WHERE path = ?", (path,))
        conn.execute(
            "INSERT INTO files_fts(path, content) VALUES (?, ?)",
            (path, content[:FTS_CONTENT_CHARS])
        )


def prune_missing(conn: sqlite3.Connection, vault_root: Path) -> int:
    rows = conn.execute("SELECT path FROM files").fetchall()
    removed = 0
    for (rel_path,) in rows:
        full = vault_root / rel_path
        if not full.exists():
            conn.execute("DELETE FROM files WHERE path = ?", (rel_path,))
            conn.execute("DELETE FROM files_fts WHERE path = ?", (rel_path,))
            conn.execute("DELETE FROM chunks WHERE file_path = ?", (rel_path,))
            removed += 1
    if removed:
        conn.commit()
    return removed


def prune_chunks(conn: sqlite3.Connection, file_path: str) -> None:
    conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def ollama_embed(text: str) -> list[float]:
    text = text.replace("\x00", "")
    for limit in (len(text), 4000, 2000, 1000):
        chunk = text[:limit]
        payload = json.dumps({
            "model": EMBED_MODEL,
            "prompt": chunk,
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            return data["embedding"]
        except urllib.error.HTTPError as e:
            if e.code == 500 and limit > 1000:
                continue
            raise
    raise RuntimeError("Embedding failed even with 1000-char chunk")


def ollama_embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    cleaned = [t.replace("\x00", "")[:MAX_CONTENT_CHARS] for t in texts]
    payload = json.dumps({
        "model": EMBED_MODEL,
        "input": cleaned,
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        return data["embeddings"]
    except Exception:
        return [ollama_embed(t) for t in texts]


def ollama_summarize(content: str, filepath: str) -> str:
    truncated = content[:4000]
    prompt = (
        f"File: {filepath}\n\n"
        f"```\n{truncated}\n```\n\n"
        "Write a 2-3 sentence summary of what this file does. "
        "Be specific about functionality. No preamble, just the summary."
    )
    payload = json.dumps({
        "model": SUMMARY_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 150},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["response"].strip()
    except Exception as e:
        return f"(summary failed: {e})"


# ---------------------------------------------------------------------------
# File walking
# ---------------------------------------------------------------------------

def should_skip_dir(dirname: str) -> bool:
    return dirname in SKIP_DIRS or dirname.startswith(".")


def should_index(filepath: Path) -> bool:
    ext = filepath.suffix.lower()
    if ext in SKIP_EXTENSIONS:
        return False
    if ext in INDEXABLE_EXTENSIONS:
        return True
    return False


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def collect_files(root: Path) -> list[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if should_index(fpath):
                files.append(fpath)
    return sorted(files)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_markdown(text: str) -> list[dict]:
    lines = text.split("\n")
    chunks: list[dict] = []
    current_heading = None
    current_lines: list[str] = []
    current_start = 1

    def flush():
        if current_lines:
            content = "\n".join(current_lines)
            if content.strip():
                chunks.append({
                    "heading": current_heading,
                    "content": content,
                    "start_line": current_start,
                    "end_line": current_start + len(current_lines) - 1,
                })

    for i, line in enumerate(lines):
        if line.startswith("## "):
            flush()
            current_heading = line.lstrip("# ").strip()
            current_lines = [line]
            current_start = i + 1
        else:
            current_lines.append(line)

    flush()

    merged: list[dict] = []
    for chunk in chunks:
        if merged and len(merged[-1]["content"]) < MIN_CHUNK_CHARS:
            merged[-1]["content"] += "\n" + chunk["content"]
            merged[-1]["end_line"] = chunk["end_line"]
            if chunk["heading"] and not merged[-1]["heading"]:
                merged[-1]["heading"] = chunk["heading"]
        else:
            merged.append(chunk)

    if len(merged) > 1 and len(merged[-1]["content"]) < MIN_CHUNK_CHARS:
        merged[-2]["content"] += "\n" + merged[-1]["content"]
        merged[-2]["end_line"] = merged[-1]["end_line"]
        merged.pop()

    final: list[dict] = []
    for chunk in merged:
        content = chunk["content"]
        if len(content) <= MAX_CHUNK_CHARS:
            final.append(chunk)
        else:
            offset = 0
            while offset < len(content):
                end = min(offset + MAX_CHUNK_CHARS, len(content))
                final.append({
                    "heading": chunk["heading"],
                    "content": content[offset:end],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                })
                offset = end - OVERLAP_CHARS if end < len(content) else end
            if not final:
                final.append(chunk)

    return final


def chunk_code(text: str) -> list[dict]:
    chunks: list[dict] = []
    offset = 0
    while offset < len(text):
        end = min(offset + MAX_CHUNK_CHARS, len(text))
        chunk_text = text[offset:end]
        start_line = text[:offset].count("\n") + 1
        end_line = start_line + chunk_text.count("\n")
        chunks.append({
            "heading": None,
            "content": chunk_text,
            "start_line": start_line,
            "end_line": end_line,
        })
        if end >= len(text):
            break
        offset = end - OVERLAP_CHARS
    return chunks


def chunk_file(text: str, ext: str) -> list[dict]:
    if len(text) <= MAX_CONTENT_CHARS:
        return [{
            "heading": None,
            "content": text,
            "start_line": 1,
            "end_line": text.count("\n") + 1,
        }]
    if ext in (".md", ".txt", ".rst"):
        chunks = chunk_markdown(text)
    else:
        chunks = chunk_code(text)
    if not chunks:
        return [{
            "heading": None,
            "content": text[:MAX_CHUNK_CHARS],
            "start_line": 1,
            "end_line": text[:MAX_CHUNK_CHARS].count("\n") + 1,
        }]
    return chunks


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def run_index(vault_root: Path, sub_path: Path | None, db_path: Path,
              test_mode: bool = False, force: bool = False,
              skip_summary: bool = False, rechunk: bool = False) -> None:
    root = vault_root / sub_path if sub_path else vault_root
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    try:
        ollama_embed("connection test")
    except Exception as e:
        print(f"Error: Cannot reach Ollama at {OLLAMA_BASE}: {e}", file=sys.stderr)
        print("Make sure Ollama is running: ollama serve", file=sys.stderr)
        sys.exit(1)

    conn = init_db(db_path)
    pruned = prune_missing(conn, vault_root)
    if pruned:
        print(f"Pruned {pruned} deleted files from index")
    migrate_fts(conn, vault_root)
    files = collect_files(root)
    if test_mode:
        files = files[:5]
        print(f"TEST MODE: indexing first {len(files)} files\n")

    total = len(files)
    new_count = 0
    skip_count = 0
    err_count = 0

    BATCH_SIZE = 20
    batch_texts: list[str] = []
    batch_meta: list[tuple[str, str, Path, str]] = []

    def flush_batch() -> None:
        nonlocal new_count, err_count
        if not batch_texts:
            return
        try:
            embeddings = ollama_embed_batch(batch_texts)
        except Exception as e:
            print(f"  [batch error] {e}", file=sys.stderr)
            for rel, _, _, _ in batch_meta:
                err_count += 1
            return
        for (rel, chash, _, raw_text), emb in zip(batch_meta, embeddings):
            upsert_file(conn, rel, chash, emb, "(pending)", content=raw_text)
            new_count += 1
            print(f"  [+] {rel}")
        conn.commit()

    for i, fpath in enumerate(files, 1):
        rel = str(fpath.relative_to(vault_root))
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            err_count += 1
            continue
        if not text.strip():
            skip_count += 1
            continue

        chash = content_hash(text)
        if not force and get_stored_hash(conn, rel) == chash:
            skip_count += 1
            continue

        embed_text = f"File: {rel}\n\n{text[:MAX_CONTENT_CHARS]}"
        batch_texts.append(embed_text)
        batch_meta.append((rel, chash, fpath, text))

        if len(batch_texts) >= BATCH_SIZE:
            print(f"  [{i}/{total}] Embedding batch of {len(batch_texts)}...")
            flush_batch()
            batch_texts.clear()
            batch_meta.clear()

    if batch_texts:
        print(f"  Embedding final batch of {len(batch_texts)}...")
        flush_batch()
        batch_texts.clear()
        batch_meta.clear()

    # Chunk embedding for large files
    if rechunk or force:
        chunk_candidates = conn.execute("SELECT path FROM files").fetchall()
    else:
        chunk_candidates = conn.execute("""
            SELECT f.path FROM files f
            LEFT JOIN (SELECT file_path, COUNT(*) as cnt FROM chunks GROUP BY file_path) c
                ON f.path = c.file_path
            WHERE c.cnt IS NULL OR c.cnt = 0
        """).fetchall()

    if chunk_candidates:
        total_chunks = 0
        chunk_batch_texts: list[str] = []
        chunk_batch_meta: list[tuple[str, int, dict]] = []
        CHUNK_BATCH_SIZE = 20

        def flush_chunk_batch() -> None:
            nonlocal total_chunks
            if not chunk_batch_texts:
                return
            try:
                embeddings = ollama_embed_batch(chunk_batch_texts)
            except Exception:
                return
            for (fp, ci, cd), emb in zip(chunk_batch_meta, embeddings):
                blob = pack_embedding(emb)
                emb_n = embedding_norm(emb)
                conn.execute("""
                    INSERT INTO chunks (file_path, chunk_index, heading, start_line,
                                        end_line, embedding, embedding_norm)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(file_path, chunk_index) DO UPDATE SET
                        heading = excluded.heading, start_line = excluded.start_line,
                        end_line = excluded.end_line, embedding = excluded.embedding,
                        embedding_norm = excluded.embedding_norm
                """, (fp, ci, cd.get("heading"), cd.get("start_line"),
                      cd.get("end_line"), blob, emb_n))
                total_chunks += 1
            conn.commit()

        print(f"\nChunking {len(chunk_candidates)} files...")
        for ci, (rel,) in enumerate(chunk_candidates, 1):
            fpath = vault_root / rel
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            ext = fpath.suffix.lower()
            file_chunks = chunk_file(text, ext)
            if len(file_chunks) <= 1:
                continue

            prune_chunks(conn, rel)
            for chunk_idx, chunk in enumerate(file_chunks):
                embed_text = f"File: {rel}\n\n{chunk['content'][:MAX_CHUNK_CHARS]}"
                chunk_batch_texts.append(embed_text)
                chunk_batch_meta.append((rel, chunk_idx, chunk))
                if len(chunk_batch_texts) >= CHUNK_BATCH_SIZE:
                    flush_chunk_batch()
                    chunk_batch_texts.clear()
                    chunk_batch_meta.clear()

        if chunk_batch_texts:
            flush_chunk_batch()

        conn.execute("DELETE FROM chunks WHERE file_path NOT IN (SELECT path FROM files)")
        conn.commit()
        print(f"  {total_chunks} chunks created")

    # Summaries
    if not skip_summary:
        pending = conn.execute(
            "SELECT path FROM files WHERE summary = '(pending)'"
        ).fetchall()
        if pending:
            print(f"\nSummarizing {len(pending)} files...")
            for i, (rel,) in enumerate(pending, 1):
                fpath = vault_root / rel
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                summary = ollama_summarize(text[:MAX_CONTENT_CHARS], rel)
                conn.execute(
                    "UPDATE files SET summary = ? WHERE path = ?", (summary, rel)
                )
                conn.commit()
                print(f"  [{i}/{len(pending)}] ~ {rel}")

    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    chunked_files = conn.execute(
        "SELECT COUNT(DISTINCT file_path) FROM chunks"
    ).fetchone()[0]
    total_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    print(f"\nDone. {total} files scanned ({skip_count} unchanged, {new_count} indexed"
          + (f", {err_count} errors" if err_count else "") + ")")
    print(f"Total: {total_files} files, {chunk_count} chunks across {chunked_files} files")
    print(f"Database: {db_path}")
    conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a local semantic search index via Ollama embeddings"
    )
    parser.add_argument(
        "root", type=str,
        help="Root directory to index"
    )
    parser.add_argument(
        "--path", type=str, default=None,
        help="Subdirectory within root to index (default: entire root)"
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="Database path (default: auto per root dir)"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Index only 5 files and display results"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-index all files even if unchanged"
    )
    parser.add_argument(
        "--no-summary", action="store_true",
        help="Skip summary generation (embeddings only)"
    )
    parser.add_argument(
        "--rechunk", action="store_true",
        help="Force re-chunking of all files"
    )
    args = parser.parse_args()

    vault_root = Path(args.root).resolve()
    if not vault_root.is_dir():
        print(f"Error: {vault_root} is not a directory", file=sys.stderr)
        sys.exit(1)

    if args.db:
        db_path = Path(args.db)
    else:
        db_path = db_path_for_root(vault_root)

    sub_path = Path(args.path) if args.path else None

    print(f"Root:     {vault_root}")
    print(f"Indexing: {vault_root / sub_path if sub_path else vault_root}")
    print(f"Database: {db_path}")
    print(f"Models:   embed={EMBED_MODEL}, summary={SUMMARY_MODEL}")
    if args.no_summary:
        print(f"Mode:     embeddings only (no summaries)")
    print()

    t0 = time.time()
    run_index(vault_root, sub_path, db_path,
              test_mode=args.test, force=args.force,
              skip_summary=args.no_summary, rechunk=args.rechunk)
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
