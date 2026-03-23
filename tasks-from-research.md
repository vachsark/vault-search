# vault-search — Tasks from Research Session (2026-03-12)

Research findings mapped to search quality improvements. Read at session start.

---

## High Priority

### 1. Information-theoretic relevance scoring

**Source**: `cs--information-theory-shannon.md`, `cs--probabilistic-graphical-models.md`
**Problem**: RRF fusion assigns weights by rank position, not by how informative a result is. A BM25 hit on a rare term is more informative than a hit on a common term, but both get the same rank-based weight.
**Task**: Weight results by information content:

- **IDF-aware scoring**: A match on a term with high IDF (rare in corpus) should get a boost — it carries more Shannon information bits
- `information_content = -log2(P(term))` where P(term) = document frequency / total documents
- Apply as a multiplier on BM25 scores before RRF fusion
- This is the core insight from Shannon: rare signals carry more information

### 2. Probabilistic relevance model for reranking

**Source**: `cs--probabilistic-graphical-models.md`, `cs--information-theory-shannon.md`
**Problem**: LLM reranking (qwen3:8b) is slow (~120ms/candidate) and binary (relevant/not). A probabilistic model could be faster and produce continuous scores.
**Task**: Consider a lightweight probabilistic relevance model:

- Model P(relevant | query, document) using features: BM25 score, cosine similarity, term overlap, document length
- Train on implicit feedback: results that users click/read are positive, skipped results are negative
- The PGM note's "naive Bayes" concept: assume feature independence → P(rel|features) proportional to product of P(feature|rel)
- This could replace or supplement LLM reranking for the common case, falling back to LLM for ambiguous candidates

### 3. Query intent classification

**Source**: `cs--information-theory-shannon.md`, `cs--probabilistic-graphical-models.md`
**Problem**: The `--intent` flag requires manual specification. Most queries have obvious intent from context.
**Task**: Auto-detect query intent:

- Use the information theory concept of mutual information: which vault domain (cs, econ, math, etc.) has highest MI with the query terms?
- Simple implementation: compute TF-IDF of query terms against each domain's vocabulary → highest-scoring domain is the intent
- Fall back to user-specified `--intent` when auto-detection confidence is low (entropy of domain distribution > threshold)

---

## Medium Priority

### 4. Hierarchical chunk retrieval

**Source**: `cs--database-indexing-strategies.md` (B-tree levels), `cs--storage-engines.md`
**Problem**: Fixed-size chunks lose context. A question about "how X relates to Y" might need the parent section, not just the paragraph.
**Task**: Store chunks at multiple granularities:

- **Paragraph level** (current) — precise but loses context
- **Section level** (## heading to next ##) — better for conceptual queries
- **Document level** — best for "what is this note about?" queries
- The B-tree analogy: search at the leaf level (paragraph) first, but traverse up to parent nodes (section, document) when leaf-level confidence is low
- Use the query type to select granularity: short keyword queries → paragraph, questions → section, broad topics → document

### 5. Index compression using coding theory

**Source**: `math--coding-theory.md`, `cs--information-theory-shannon.md`
**Problem**: SQLite index grows with corpus size. At 654 notes it's fine, but at 5000+ notes the embedding storage becomes significant.
**Task**: Apply dimensionality reduction to embeddings:

- Product quantization: split 768-dim embedding into 8 sub-vectors of 96 dims, quantize each to nearest centroid (256 centroids per sub-space)
- Storage: 768 floats (3072 bytes) → 8 bytes per embedding (384x compression)
- The coding theory note's "rate-distortion tradeoff": you lose some precision but gain massive storage savings
- Recall drops ~2-5% for 384x compression — acceptable for a first-pass retrieval stage

---

## Future

### 6. Cross-note link prediction

**Source**: `cs--network-science-applied.md`, `cs--probabilistic-graphical-models.md`
**Idea**: Predict missing wikilinks between notes. If note A is similar to note B and both link to note C, but A doesn't link to B, suggest the link. This is the "link prediction" problem from network science — use embedding similarity + shared neighbors as features. Could power a "suggested connections" feature for vault maintenance.
