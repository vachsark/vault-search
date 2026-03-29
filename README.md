# vault-search

Local semantic search + knowledge graph for your notes. Hybrid retrieval (BM25 + embeddings + RRF) with automatic entity/relationship extraction. Runs entirely on your machine via [Ollama](https://ollama.ai) — no API cost, no data leaves your network.

## What it does

1. **Index** your markdown files with embeddings for semantic search
2. **Extract** a knowledge graph — entities and relationships — from your notes using a local LLM
3. **Search** with hybrid retrieval that returns ranked files AND shows how concepts connect

```
$ python3 vault-search.py "reinforcement learning" ~/notes

0.0323  Knowledge/cs--inverse-reinforcement-learning.md
        Inverse Reinforcement Learning

0.0320  Knowledge/cs--temporal-difference-learning.md
        Temporal Difference Learning

── Graph Context ──
  reinforcement learning (concept, 28 connections)
    → applies_to → reward hacking
    → builds_on → cybernetics
    ← markov decision process ← relates_to
  reinforcement learning from human feedback (technique, 10 connections)
    → relates_to → reward hacking
    ← direct preference optimization ← relates_to
```

## Quick start

```bash
# 1. Install Ollama and pull the models
ollama pull qwen3-embedding:0.6b   # embeddings (required)
ollama pull qwen3.5:9b             # graph extraction + re-ranking (optional)

# 2. Index your files (embeddings + BM25)
python3 vault-index.py ~/notes

# 3. Build the knowledge graph (entity/relationship extraction)
python3 vault-graph.py index ~/notes

# 4. Search — returns files + graph context automatically
python3 vault-search.py "attention mechanism" ~/notes
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) running locally
- Optional: `numpy` for ~9x faster search (`pip install numpy`)

No other dependencies — uses only Python stdlib + Ollama HTTP API.

## The tools

### vault-index.py — Build the search index

Walks your directory, generates embeddings, builds a BM25 index. Incremental — only re-indexes changed files.

```bash
python3 vault-index.py ~/notes                # index a directory
python3 vault-index.py ~/notes --force        # re-index everything
python3 vault-index.py ~/notes --no-summary   # skip LLM summaries (faster)
python3 vault-index.py ~/notes --rechunk      # force re-chunk all files
```

### vault-graph.py — Build the knowledge graph

Extracts entities (concepts, people, theories, techniques) and relationships (builds_on, contradicts, implements, etc.) from your notes using a local LLM. Stores in the same SQLite database as the search index.

```bash
python3 vault-graph.py index ~/notes                  # extract from all .md files
python3 vault-graph.py index ~/notes --incremental    # only new/changed files
python3 vault-graph.py query ~/notes "attention"      # find entity connections
python3 vault-graph.py query ~/notes "dopamine" --hops 2  # traverse 2 levels
python3 vault-graph.py stats ~/notes                  # show graph statistics
python3 vault-graph.py export ~/notes --top 100       # export as JSON
python3 vault-graph.py normalize-db ~/notes           # migrate entity types to canonical labels
```

Entity types: `concept`, `technique`, `theory`, `person`, `field`, `system`, `anatomy`, `biology`, `event`, `publication`

Relationship types are normalized to 15 canonical types: `relates_to`, `builds_on`, `contradicts`, `applies_to`, `implements`, `extends`, `part_of`, `uses`, `enables`, `causes`, `explains`, `developed_by`, `type_of`, `complements`

### vault-search.py — Search with graph context

Hybrid search that combines semantic similarity with keyword matching. When graph tables exist in the database, results automatically include a "Graph Context" section showing entity connections. Returns chunk-level results with section headings and line numbers.

```bash
python3 vault-search.py "query" ~/notes                       # hybrid search
python3 vault-search.py "query" ~/notes --rerank              # LLM re-ranking (better quality)
python3 vault-search.py "query" ~/notes --expand --rerank     # full pipeline (HyDE + rerank)
python3 vault-search.py "query" ~/notes --mode bm25           # keyword-only (fastest)
python3 vault-search.py "query" ~/notes --no-graph            # skip graph context
python3 vault-search.py "query" ~/notes --json                # JSON output
python3 vault-search.py "query" ~/notes --path Projects/      # filter by path
python3 vault-search.py "query" ~/notes --no-cache            # bypass disk embedding cache
python3 vault-search.py "query" ~/notes --intent "domain"     # steer HyDE toward a domain
python3 vault-search.py "query" ~/notes --explain             # show per-result scoring breakdown
```

**Typed sub-queries** — mix retrieval strategies in a single query:

```bash
python3 vault-search.py 'lex:"exact term" vec:"concept" hyde:"question"' ~/notes
```

| Prefix            | Strategy             | Best for                                   |
| ----------------- | -------------------- | ------------------------------------------ |
| `lex:"term"`      | BM25 keyword match   | Exact identifiers, code symbols            |
| `vec:"concept"`   | Embedding similarity | Conceptual / semantic queries              |
| `hyde:"question"` | HyDE expansion       | Exploratory "what is / how does" questions |
| (unprefixed)      | Hybrid (default)     | General use                                |

Results from each sub-query are fused via Reciprocal Rank Fusion before re-ranking.

### verify-citations.py — Check URLs in markdown files

Verifies that URLs in your notes are alive. Catches dead links (404), redirects, and hallucinated references (arxiv IDs with wrong format, placeholder DOIs, etc.). Runs concurrently — no external dependencies, stdlib only.

```bash
python3 verify-citations.py note.md                    # check one file
python3 verify-citations.py note.md --fix              # print suggested replacements
python3 verify-citations.py --dir Knowledge/           # scan a directory
python3 verify-citations.py --dir Knowledge/ --json    # JSON output
python3 verify-citations.py --dir Knowledge/ --issues-only  # only show files with problems
```

### knowledge-path.py — Find paths between concepts

Finds the shortest conceptual path between two ideas using the knowledge graph. Like "six degrees of separation" for your notes.

```bash
python3 knowledge-path.py "prospect theory" "transformer architecture"
python3 knowledge-path.py "dopamine" "market microstructure" --max-hops 8
python3 knowledge-path.py "ADHD" "reinforcement learning" --all   # show multiple paths
python3 knowledge-path.py "attention" "memory" --stats            # show graph size first
```

### synthesis-suggest.py — Find cross-domain synthesis candidates

Surfaces concept pairs with high neighbor overlap that span different disciplines — the most promising cross-domain connections not yet bridged by a synthesis note.

```bash
python3 synthesis-suggest.py ~/notes                    # top 10 candidates
python3 synthesis-suggest.py ~/notes --top 20
python3 synthesis-suggest.py ~/notes --min-jaccard 0.25 # stricter threshold
python3 synthesis-suggest.py ~/notes --json             # machine-readable output
python3 synthesis-suggest.py ~/notes --ucb              # UCB exploration bonus (up-weights under-explored entities)
```

Requires discipline-prefixed filenames (`discipline--topic.md` naming convention). If your notes use a different convention, set `DISCIPLINE_SEPARATOR` in the script.

### causal-trace.py — Trace causal chains between concepts

Walks the knowledge graph to find causal or mechanistic chains between two concepts. Unlike knowledge-path (shortest path), this follows directional relationships like `causes`, `enables`, `builds_on`.

```bash
python3 causal-trace.py "cortisol" "decision making"
python3 causal-trace.py "dopamine" "addiction" --allow-reverse  # also follow reverse edges
python3 causal-trace.py "inflation" "unemployment" --max-depth 6
```

### concept-to-code.py — Bridge knowledge to code

Finds where a knowledge concept maps to actual code implementations in your repository. Searches both the knowledge graph (concept → related notes) and the codebase (grep for concept terms in source files).

```bash
python3 concept-to-code.py "spaced repetition"     # find code implementing this concept
python3 concept-to-code.py "attention mechanism"    # find implementations
```

### vault-ask.py — Natural language Q&A over your vault

Ask a question in plain English, get an answer grounded in your actual notes. Uses vault-search to find relevant context, then synthesizes an answer.

```bash
python3 vault-ask.py "What does the vault know about scheduling optimization?"
python3 vault-ask.py "How does the learning engine work?"
```

### leiden-communities.py — Community detection in the knowledge graph

Detects clusters of densely connected concepts using the Leiden algorithm. Useful for understanding the structure of your knowledge base and finding gaps between communities.

```bash
python3 leiden-communities.py                       # detect all communities
python3 leiden-communities.py --stats-only          # just show statistics
python3 leiden-communities.py --query "dopamine"    # which community is this concept in?
python3 leiden-communities.py --sweep               # parameter sweep for resolution
python3 leiden-communities.py --export              # export community assignments
```

Requires `leidenalg` and `igraph`: `pip install leidenalg python-igraph`

## How it works

### Search pipeline

```
Query → [HyDE expand] → Embed → Cosine similarity (file + chunk level)
                                        ↓
                              Reciprocal Rank Fusion ← BM25 (FTS5 trigram)
                                        ↓
                              [LLM Re-rank top 20]
                                        ↓
                              Results + Graph Context
```

Three search modes:

| Mode                 | How it works                                     | Best for           |
| -------------------- | ------------------------------------------------ | ------------------ |
| **hybrid** (default) | Semantic + BM25 fused via Reciprocal Rank Fusion | General queries    |
| **semantic**         | Cosine similarity on embeddings only             | Conceptual queries |
| **bm25**             | FTS5 trigram keyword matching only               | Exact identifiers  |

### Knowledge graph

The graph extractor sends each note to a local LLM with a structured prompt asking for entities and relationships. Results are stored in two SQLite tables (`entities`, `relations`) in the same database as the search index. At query time, vault-search looks up matching entities and appends their connections to the output.

Key design choices:

- **Same database**: Graph lives alongside search index — one DB per indexed directory
- **Incremental**: Only processes new notes, skips already-extracted files
- **Normalized relations**: LLMs invent creative relationship types. We normalize 500+ variants to 15 canonical types so the graph is consistent
- **No external services**: No Neo4j, no graph databases — just SQLite

### Search design choices

- **Dual-granularity scoring**: Both file-level and chunk-level embeddings, with best chunk score promoting file rank
- **Path-aware keyword bonus**: Query terms in file paths get a 15% boost with stem matching
- **NumPy batch cosine**: Single matrix multiply for all similarities when numpy is available (~9x faster)
- **Position-aware blend**: Top-5 results trust retrieval more (60/40); the rest trust the reranker more (40/60)
- **Mode-aware low-confidence detection**: RRF scores (hybrid/bm25) use a spread threshold of 0.003; semantic mode uses absolute score < 0.15 — prevents false low-confidence warnings when scores are normally small

## Configuration

| Variable                 | Default                                 | Description                                                 |
| ------------------------ | --------------------------------------- | ----------------------------------------------------------- |
| `OLLAMA_BASE`            | `http://localhost:11434`                | Ollama API URL                                              |
| `EMBED_MODEL`            | `qwen3-embedding:0.6b`                  | Embedding model                                             |
| `GRAPH_MODEL`            | `qwen3.5:9b`                            | Graph extraction model                                      |
| `EXPAND_MODEL`           | `qwen3.5:9b`                            | HyDE expansion model                                        |
| `RERANK_MODEL`           | `qwen3.5:9b`                            | Re-ranking model                                            |
| `VAULT_SEARCH_DB`        | `~/.local/share/vault-search/<hash>.db` | Database path                                               |
| `VAULT_SEARCH_CACHE_DIR` | `~/.cache/vault-search/embed-cache`     | Disk embedding cache directory (use `--no-cache` to bypass) |

## Performance

Benchmarked on ~6,000 notes with 40K+ graph relations (AMD Ryzen 5 7600X, RX 9070 XT):

| Operation                           | Time    |
| ----------------------------------- | ------- |
| Search (hybrid + graph), cold embed | ~4s     |
| Search (hybrid + graph), cached     | ~170ms  |
| Search (BM25 + graph)               | ~170ms  |
| Search with `--rerank`              | ~6-8s   |
| Graph extraction per note           | ~30-40s |
| Incremental re-index                | seconds |

The disk embedding cache (enabled by default) skips the ~4s Ollama embed call on repeated queries by persisting vectors to `~/.cache/vault-search/embed-cache/`. Use `--no-cache` after switching embedding models.

## File types indexed

Markdown, text, RST, TypeScript, JavaScript, Python, Shell, Rust, Go, Ruby, Java, Kotlin, Swift, C/C++, YAML, TOML, JSON, INI.

Skipped: images, PDFs, archives, binaries, lock files, `node_modules`, `.git`, build output.

## License

MIT
