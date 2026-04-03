# vault-search

**Your Obsidian vault has 10,000 notes. Finding the right one takes 60 seconds of grep. vault-search does it in 0.3ms.**

Local hybrid search + knowledge graph for your notes. BM25 keyword search (FTS5), semantic embeddings, entity/relationship extraction, Leiden community detection, and RRF fusion — all running on your machine. No API keys. No data leaves your network.

```
$ python3 vault-search.py "how does reinforcement learning relate to dopamine" ~/notes

0.0847  Knowledge/cog--dopamine-reward-circuits.md
        Dopamine and Reward Prediction Error
        "...the phasic dopamine signal encodes a temporal difference error,
        identical in structure to the TD learning update in RL..."

0.0821  Knowledge/cs--temporal-difference-learning.md
        Temporal Difference Learning

── Graph Context ──
  dopamine (biology, 34 connections)
    → causes → reward prediction error
    → relates_to → reinforcement learning
    → enables → habit formation
  reinforcement learning (concept, 28 connections)
    → builds_on → cybernetics
    → relates_to → dopamine signaling
    ← inverse reinforcement learning ← extends
```

---

## Quick start

No Ollama required for the fast path — BM25 + catalysts work out of the box with zero dependencies.

```bash
# Clone and index
git clone https://github.com/vachsark/vault-search
cd vault-search

# Index your vault (BM25 only — no Ollama needed)
python3 vault-index.py ~/my-obsidian-vault --no-summary

# Search
python3 vault-search.py "reinforcement learning and dopamine" ~/my-obsidian-vault
```

**With Ollama** (semantic search + graph extraction):

```bash
ollama pull qwen3-embedding:0.6b   # embeddings (~500MB)
ollama pull qwen3:8b               # graph extraction + reranking (optional)

python3 vault-index.py ~/my-obsidian-vault
python3 vault-graph.py index ~/my-obsidian-vault
python3 vault-search.py "attention mechanism" ~/my-obsidian-vault
```

Zero Python dependencies beyond stdlib. Optional: `pip install numpy` for 9x faster similarity search.

---

## The problem with RAG

Standard RAG for a knowledge base looks like this:

1. Take a query
2. Embed it
3. Fetch top-k chunks
4. Stuff 60–90K tokens into an LLM prompt
5. Get an answer

At 10,000 notes, this costs ~$0.05–0.15 per query and takes 3–8 seconds. The context window fills with noise. You lose cross-document connections entirely.

vault-search is the retrieval layer that makes this unnecessary. Instead of feeding your LLM 90K tokens of raw chunks, you get:

- **5–8K tokens** of ranked, relevant context
- **Graph connections** between concepts (what builds on what, what contradicts what)
- **Sub-ms lookups** for already-indexed queries
- **Zero API cost** for the retrieval step

The difference: vault-search is a search engine, not a chatbot wrapper.

---

## Architecture

```
Query
  │
  ├─ BM25 (FTS5 trigram)          0.3ms  ─────────────────┐
  │                                                         │
  ├─ Embedding similarity          ~85ms (numpy, cached)   ├─ RRF Fusion → Ranked results
  │    └─ [HyDE expansion]         +~2s (optional)          │              + Graph Context
  │                                                         │
  └─ Knowledge Graph lookup        ~0.5ms ─────────────────┘
       └─ Leiden communities
       └─ Entity connections
       └─ Causal chains

Optional: LLM reranker (top-20 candidates) → +~6s, higher precision
```

Everything stored in a single SQLite database per vault. No external services. No graph databases. No vector stores. Just files and FTS5.

---

## Features and benchmarks

Benchmarked on ~6,000 notes, 40K+ graph relations (AMD Ryzen 5 7600X):

| Feature                               | Latency    | Notes                         |
| ------------------------------------- | ---------- | ----------------------------- |
| BM25 keyword search                   | **0.3ms**  | FTS5 trigram, AND→OR fallback |
| Semantic search (numpy, cached embed) | **~85ms**  | Cosine similarity, disk cache |
| Hybrid BM25 + semantic (cached)       | **~170ms** | RRF fusion                    |
| Hybrid, cold embed (no cache)         | ~4s        | One Ollama call               |
| With `--rerank`                       | ~6–8s      | LLM reranks top-20            |
| Graph context lookup                  | **~0.5ms** | SQLite entity join            |
| Knowledge graph extraction (per note) | ~30–40s    | One-time, incremental         |

**Works without Ollama**: BM25 + graph traversal runs entirely on stdlib. Add Ollama for semantic search and graph extraction.

---

## Search modes

Three retrieval strategies, mix-and-match:

| Mode               | How                           | Best for                        |
| ------------------ | ----------------------------- | ------------------------------- |
| `hybrid` (default) | BM25 + embeddings, RRF fusion | General queries                 |
| `bm25`             | FTS5 trigram only             | Exact identifiers, code symbols |
| `semantic`         | Cosine similarity only        | Conceptual / cross-lingual      |

**Typed sub-queries** — combine strategies in one call:

```bash
python3 vault-search.py 'lex:"exact term" vec:"concept" hyde:"what is X"' ~/notes
```

| Prefix            | Strategy             | Best for                        |
| ----------------- | -------------------- | ------------------------------- |
| `lex:"term"`      | BM25 keyword         | Exact identifiers, code symbols |
| `vec:"concept"`   | Embedding similarity | Conceptual / semantic           |
| `hyde:"question"` | HyDE expansion       | Exploratory "how does X work"   |
| (unprefixed)      | Hybrid               | General use                     |

Results from each sub-query are fused via Reciprocal Rank Fusion before (optional) reranking.

---

## Knowledge graph

vault-search extracts a structured knowledge graph from your notes using a local LLM. Every note becomes a set of entities and typed relationships stored in SQLite alongside the search index.

```bash
# Build the graph (one-time, ~30-40s per note)
python3 vault-graph.py index ~/notes

# Query entity connections
python3 vault-graph.py query ~/notes "attention" --hops 2

# Find the shortest conceptual path between two ideas
python3 knowledge-path.py "prospect theory" "transformer architecture"

# Trace causal chains
python3 causal-trace.py "cortisol" "decision making"

# Find cross-domain synthesis candidates
python3 synthesis-suggest.py ~/notes
```

Entity types: `concept`, `technique`, `theory`, `person`, `field`, `system`, `anatomy`, `biology`, `event`, `publication`

Relationship types are normalized from 500+ LLM-invented variants to 15 canonical types: `relates_to`, `builds_on`, `contradicts`, `applies_to`, `implements`, `extends`, `part_of`, `uses`, `enables`, `causes`, `explains`, `developed_by`, `type_of`, `complements`

**Leiden community detection** — cluster your knowledge graph into topic communities and find bridge concepts spanning multiple domains:

```bash
python3 leiden-communities.py --query "dopamine"    # which community?
python3 leiden-communities.py --stats-only           # community overview
```

---

## Karpathy knowledge base workflow

This is the search engine for the workflow [Karpathy described](https://twitter.com/karpathy): building a personal knowledge base of papers, notes, and ideas as training data for your own thinking.

The problem with that workflow at scale: retrieval. When you have 5,000+ notes, grep doesn't work. Embeddings alone miss exact matches. RAG blows up your context window.

vault-search layers BM25 (for precision), embeddings (for recall), and a knowledge graph (for connections) into a single retrieval pipeline that costs $0 per query and runs in sub-second time. Your notes stay local. The graph connects them automatically.

Typical use: pipe search results directly into your LLM context instead of full documents.

```bash
# Get ranked context for an LLM prompt — 5-8K tokens instead of 90K
python3 vault-search.py "how does X work" ~/notes --json | jq '.results[:5]'

# Or ask directly (uses vault-search internally, then synthesizes)
python3 vault-ask.py "What does my vault know about scheduling optimization?"
```

---

## Comparison

|                                | vault-search      | Basic RAG        | `grep -r`         |
| ------------------------------ | ----------------- | ---------------- | ----------------- |
| Semantic search                | Yes (with Ollama) | Yes              | No                |
| Keyword search                 | Yes (BM25/FTS5)   | Sometimes        | Yes               |
| Works without API keys         | Yes               | No               | Yes               |
| Knowledge graph                | Yes               | No               | No                |
| Cross-concept connections      | Yes               | No               | No                |
| Community detection            | Yes               | No               | No                |
| Token cost per query           | ~0                | $0.05–0.15       | ~0                |
| Context size sent to LLM       | 5–8K tokens       | 60–90K tokens    | Manual            |
| Query latency (BM25)           | 0.3ms             | ~3–8s            | ~60s on 10K files |
| Query latency (hybrid, cached) | ~170ms            | ~3–8s            | —                 |
| Data leaves your machine       | Never             | Yes (OpenAI/etc) | Never             |

---

## All tools

| Script                  | What it does                                             |
| ----------------------- | -------------------------------------------------------- |
| `vault-index.py`        | Build search index (embeddings + BM25). Incremental.     |
| `vault-search.py`       | Hybrid search with graph context. Main entry point.      |
| `vault-graph.py`        | Extract entities + relations from notes using local LLM. |
| `vault-ask.py`          | Natural language Q&A grounded in your actual notes.      |
| `knowledge-path.py`     | Shortest conceptual path between two ideas.              |
| `causal-trace.py`       | Trace causal/mechanistic chains between concepts.        |
| `synthesis-suggest.py`  | Surface cross-domain synthesis candidates.               |
| `leiden-communities.py` | Community detection in the knowledge graph.              |
| `concept-to-code.py`    | Bridge knowledge notes to code implementations.          |
| `verify-citations.py`   | Check that URLs in your notes are still alive.           |

---

## Configuration

All settings via environment variables:

| Variable                 | Default                                 | Description            |
| ------------------------ | --------------------------------------- | ---------------------- |
| `OLLAMA_BASE`            | `http://localhost:11434`                | Ollama API URL         |
| `EMBED_MODEL`            | `qwen3-embedding:0.6b`                  | Embedding model        |
| `GRAPH_MODEL`            | `qwen3:8b`                              | Graph extraction model |
| `EXPAND_MODEL`           | `qwen3:8b`                              | HyDE expansion model   |
| `RERANK_MODEL`           | `qwen3:8b`                              | Re-ranking model       |
| `VAULT_SEARCH_DB`        | `~/.local/share/vault-search/<hash>.db` | Database path          |
| `VAULT_SEARCH_CACHE_DIR` | `~/.cache/vault-search/embed-cache`     | Disk embedding cache   |

Use `--no-cache` after switching embedding models to force re-embedding.

---

## File types indexed

Markdown, text, RST, TypeScript, JavaScript, Python, Shell, Rust, Go, Ruby, Java, Kotlin, Swift, C/C++, YAML, TOML, JSON, INI.

Skipped: images, PDFs, archives, binaries, lock files, `node_modules`, `.git`, build output.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) running locally (optional — BM25 works without it)
- `pip install numpy` — optional, ~9x faster similarity search
- `pip install leidenalg python-igraph` — optional, for community detection

No other dependencies. Uses only Python stdlib + SQLite + optional Ollama HTTP API.

---

## License

MIT — Copyright (c) 2026 Vache Asatryan
