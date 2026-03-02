# vault-search

Local semantic search for your files. Hybrid retrieval (BM25 + embeddings + RRF) with optional LLM re-ranking. Runs entirely on your machine via [Ollama](https://ollama.ai) — no API cost, no data leaves your network.

## What it does

1. **Index** a directory of markdown, code, and config files into a local SQLite database with embeddings
2. **Search** using hybrid retrieval that combines semantic similarity with keyword matching
3. Optionally **re-rank** results with an LLM for significantly better quality on conceptual queries

## Quick start

```bash
# 1. Install Ollama and pull the models
ollama pull qwen3-embedding:0.6b   # embeddings (required)
ollama pull qwen3:8b               # re-ranking + HyDE (optional)

# 2. Index your files
python3 vault-index.py ~/notes

# 3. Search
python3 vault-search.py "authentication middleware" ~/notes
python3 vault-search.py "React hooks" ~/notes --top 10
python3 vault-search.py "auth patterns" ~/notes --rerank          # LLM re-ranking
python3 vault-search.py "best practices" ~/notes --expand --rerank  # full pipeline
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) running locally
- Optional: `numpy` for ~9x faster search (`pip install numpy`)

No other dependencies — uses only Python stdlib + Ollama HTTP API.

## How it works

### Indexing (`vault-index.py`)

- Walks your directory, reads markdown/code/config files
- Generates embeddings via Ollama (`qwen3-embedding:0.6b` by default)
- Builds an FTS5 trigram index for keyword search
- Chunks large files (markdown by `## ` headers, code by 6000-char windows with overlap)
- Stores everything in SQLite — one DB per indexed directory
- Incremental: only re-embeds changed files on subsequent runs

### Search (`vault-search.py`)

Three search modes:

| Mode                 | How it works                                     | Best for           |
| -------------------- | ------------------------------------------------ | ------------------ |
| **hybrid** (default) | Semantic + BM25 fused via Reciprocal Rank Fusion | General queries    |
| **semantic**         | Cosine similarity on embeddings only             | Conceptual queries |
| **bm25**             | FTS5 trigram keyword matching only               | Exact identifiers  |

### Re-ranking (`--rerank`)

Adds an LLM scoring step after retrieval. The top 20 candidates are scored by the LLM for relevance (0-10 scale), then blended with retrieval scores using position-aware weighting. Uses a sliding-window snippet finder to extract the most relevant ~400 chars from each document.

### HyDE expansion (`--expand`)

Generates a hypothetical document that would answer your query, then embeds that instead of the raw query. Document-to-document similarity is more reliable than question-to-document. Adds ~1s.

### Architecture

```
Query → [HyDE expand] → Embed → Cosine similarity (file + chunk level)
                                        ↓
                              Reciprocal Rank Fusion ← BM25 (FTS5 trigram)
                                        ↓
                              [LLM Re-rank top 20]
                                        ↓
                                    Results
```

Key design choices:

- **Dual-granularity scoring**: Both file-level and chunk-level embeddings are scored, with the best chunk score promoting a file's rank. Large files don't get penalized for embedding dilution.
- **Path-aware keyword bonus**: Query terms that appear in file paths get a 15% score boost with stem-aware matching ("notification" matches "notify").
- **NumPy batch cosine**: When numpy is available, all cosine similarities are computed via a single matrix multiply — ~9x faster than the pure-Python loop.
- **Position-aware blend**: Top-5 retrieval results trust the retrieval score more (60/40) to protect exact matches; the rest trust the reranker more (40/60).

## Configuration

All settings are configurable via environment variables:

| Variable          | Default                                 | Description                        |
| ----------------- | --------------------------------------- | ---------------------------------- |
| `OLLAMA_BASE`     | `http://localhost:11434`                | Ollama API URL                     |
| `EMBED_MODEL`     | `qwen3-embedding:0.6b`                  | Embedding model                    |
| `SUMMARY_MODEL`   | `qwen2.5-coder:7b`                      | Summary generation model (indexer) |
| `EXPAND_MODEL`    | `qwen3:8b`                              | HyDE expansion model               |
| `RERANK_MODEL`    | `qwen3:8b`                              | Re-ranking model                   |
| `VAULT_SEARCH_DB` | `~/.local/share/vault-search/<hash>.db` | Database path                      |

Example with different models:

```bash
EMBED_MODEL=nomic-embed-text RERANK_MODEL=llama3.1:8b python3 vault-search.py "my query" ~/docs
```

## CLI reference

### vault-index.py

```
python3 vault-index.py <root>              # index a directory
python3 vault-index.py <root> --path src/  # index subdirectory only
python3 vault-index.py <root> --force      # re-index everything
python3 vault-index.py <root> --no-summary # skip LLM summaries (faster)
python3 vault-index.py <root> --rechunk    # force re-chunk all files
python3 vault-index.py <root> --test       # index only 5 files (dry run)
python3 vault-index.py <root> --db my.db   # custom database path
```

### vault-search.py

```
python3 vault-search.py "query" [root]           # search (root defaults to .)
python3 vault-search.py "query" ~/notes --top 10  # more results
python3 vault-search.py "query" ~/notes --path Projects/  # path filter
python3 vault-search.py "query" ~/notes --mode bm25       # keyword-only
python3 vault-search.py "query" ~/notes --mode semantic    # embeddings-only
python3 vault-search.py "query" ~/notes --rerank           # LLM re-ranking
python3 vault-search.py "query" ~/notes --expand           # HyDE expansion
python3 vault-search.py "query" ~/notes --expand --rerank  # full pipeline
python3 vault-search.py "query" ~/notes --json             # JSON output
python3 vault-search.py "query" --db path/to/index.db      # custom DB
```

## Performance

Benchmarked on ~1,800 files / ~3,800 chunks (AMD Ryzen, Vulkan GPU):

| Mode                      | Latency |
| ------------------------- | ------- |
| Hybrid (default)          | ~560ms  |
| With `--rerank`           | ~2-3s   |
| With `--expand --rerank`  | ~3-4s   |
| NumPy cosine (1600 files) | ~14ms   |
| Pure Python cosine        | ~127ms  |

## File types indexed

Markdown, text, RST, TypeScript, JavaScript, Python, Shell, Rust, Go, Ruby, Java, Kotlin, Swift, C/C++, YAML, TOML, JSON, INI.

Skipped: images, PDFs, archives, binaries, lock files, `node_modules`, `.git`, build output.

## License

MIT
