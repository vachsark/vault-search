#!/usr/bin/env python3
"""
vault-graph.py — Knowledge graph extraction for your notes.

Extracts entities and relationships from markdown notes using a local
Ollama model. Stores them in SQLite alongside the vault-search index.
No Neo4j, no external services — just SQLite and Ollama.

Usage:
    python3 vault-graph.py index ~/notes                  # Extract entities from all .md files
    python3 vault-graph.py index ~/notes --incremental    # Only process new/changed notes
    python3 vault-graph.py query ~/notes "attention"      # Find entity and its connections
    python3 vault-graph.py query ~/notes "attention" --hops 2  # Traverse 2 levels deep
    python3 vault-graph.py stats ~/notes                  # Show graph statistics
    python3 vault-graph.py export ~/notes --top 100       # Export as JSON

Environment:
    OLLAMA_BASE    Ollama URL (default: http://localhost:11434)
    GRAPH_MODEL    Model for extraction (default: qwen3.5:9b)
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
GRAPH_MODEL = os.environ.get("GRAPH_MODEL", "qwen3.5:9b")  # updated from qwen3:8b (deprecated 2026-03-20)


def db_path_for_root(root: str) -> str:
    """Deterministic DB path per root directory. Matches vault-search.py / vault-index.py."""
    import hashlib
    custom = os.environ.get("VAULT_SEARCH_DB")
    if custom:
        return custom
    root_hash = hashlib.sha256(os.path.realpath(root).encode()).hexdigest()[:12]
    return os.path.expanduser(f"~/.local/share/vault-search/{root_hash}.db")

# Normalize the 500+ LLM-invented relation types to 15 canonical types
RELATION_MAP = {
    # builds_on
    "builds_on": "builds_on", "based_on": "builds_on", "is_based_on": "builds_on",
    "derives_from": "builds_on", "derived_from": "builds_on", "draws_from": "builds_on",
    "inspired_by": "builds_on", "influenced_by": "builds_on", "founded_by": "builds_on",
    "foundational_for": "builds_on", "foundational_to": "builds_on", "basis_for": "builds_on",
    "built_on": "builds_on", "prerequisite_for": "builds_on", "prerequisite": "builds_on",
    # contradicts
    "contradicts": "contradicts", "contrasts_with": "contradicts", "competes_with": "contradicts",
    "differs_from": "contradicts", "refutes": "contradicts", "distinct_from": "contradicts",
    "is_alternative_to": "contradicts", "alternatives_to": "contradicts", "replaces": "contradicts",
    "supersedes": "contradicts",
    # applies_to
    "applies_to": "applies_to", "applies": "applies_to", "used_in": "applies_to",
    "used_for": "applies_to", "used_by": "applies_to", "is_used_in": "applies_to",
    "applies, to": "applies_to",
    # implements
    "implements": "implements", "implemented_by": "implements", "implemented_in": "implements",
    "implemented_via": "implements", "instantiates": "implements", "formalizes": "implements",
    "formalized_by": "implements", "operationalizes": "implements",
    # extends
    "extends": "extends", "extends_to": "extends", "generalizes": "extends",
    "specializes_in": "extends", "variant_of": "extends", "is_variant_of": "extends",
    "is_version_of": "extends", "subclass_of": "extends", "is_subtype_of": "extends",
    # part_of
    "part_of": "part_of", "is_part_of": "part_of", "includes": "part_of",
    "contains": "part_of", "comprises": "part_of", "composed_of": "part_of",
    "has_part": "part_of", "has_component": "part_of", "component_of": "part_of",
    "belongs_to": "part_of", "subset_of": "part_of", "is_subset_of": "part_of",
    "decomposes_into": "part_of",
    # uses
    "uses": "uses", "employs": "uses", "utilizes": "uses", "requires": "uses",
    "depends_on": "uses", "relies_on": "uses", "uses_technique": "uses",
    # enables
    "enables": "enables", "facilitates": "enables", "supports": "enables",
    "contributes_to": "enables", "drives": "enables", "leads_to": "enables",
    "produces": "enables", "generates": "enables", "enhances": "enables",
    "improves": "enables", "accelerates_adoption": "enables",
    # causes
    "causes": "causes", "triggers": "causes", "induces": "causes",
    "affects": "causes", "modulates": "causes", "inhibits": "causes",
    "activates": "causes", "suppresses": "causes", "regulates": "causes",
    # explains
    "explains": "explains", "describes": "explains", "defines": "explains",
    "predicts": "explains", "models": "explains", "demonstrates": "explains",
    "illustrates": "explains", "measures": "explains",
    # developed_by
    "developed_by": "developed_by", "created_by": "developed_by", "coined_by": "developed_by",
    "proposed_by": "developed_by", "introduced_by": "developed_by", "authored": "developed_by",
    "wrote": "developed_by", "founded": "developed_by",
    # type_of
    "type_of": "type_of", "is_type_of": "type_of", "is_a": "type_of",
    "instance_of": "type_of", "is_instance_of": "type_of", "example_of": "type_of",
    "classified_by": "type_of", "category_of": "type_of", "is_a_type_of": "type_of",
    # complements
    "complements": "complements", "combines": "complements", "integrates": "complements",
    "integrates_with": "complements", "combines_with": "complements", "synergizes_with": "complements",
    "bridges": "complements", "connects_to": "complements", "linked_to": "complements",
    "parallel_to": "complements", "equivalent_to": "complements",
    # relates_to — catch-all structural/associative (also the canonical default)
    "relates_to": "relates_to", "related_to": "relates_to", "relates": "relates_to",
    "relate_to": "relates_to", "associated_with": "relates_to", "associated": "relates_to",
    "maps_to": "relates_to", "maps": "relates_to", "connects": "relates_to",
    "corresponds_to": "relates_to", "is_related_to": "relates_to",
    # uses (additional variants)
    "involves": "uses", "involves_use_of": "uses", "utilizes_the_concept_of": "uses",
    "monitors": "uses", "monitors_via": "uses", "tracks": "uses", "observes": "uses",
    "accesses": "uses", "leverages": "uses",
    # part_of (additional variants)
    "has": "part_of", "has_property": "part_of", "has_feature": "part_of",
    "has_attribute": "part_of", "has_function": "part_of",
    # explains (additional variants)
    "evaluates": "explains", "assesses": "explains", "studies": "explains",
    "analyzes": "explains", "investigates": "explains", "tests": "explains",
    # enables (additional variants)
    "addresses": "enables", "solves": "enables", "handles": "enables",
    "mitigates": "enables", "optimizes": "enables", "improves": "enables",
    "enhances": "enables", "strengthens": "enables", "increases": "enables",
    "reduces": "enables", "decreases": "enables",
    # developed_by (additional variants)
    "develops": "developed_by", "introduces": "developed_by", "presented_by": "developed_by",
    "discovered_by": "developed_by", "invented_by": "developed_by",
    "formulated_by": "developed_by", "proposed": "developed_by",
}

ENTITY_TYPE_MAP = {
    # concept — abstract ideas, mechanisms, phenomena, conditions
    "concept": "concept", "process": "concept", "condition": "concept",
    "mechanism": "concept", "pattern": "concept", "phenomenon": "concept",
    "property": "concept", "effect": "concept", "principle": "concept",
    "constraint": "concept", "goal": "concept", "challenge": "concept",
    "problem": "concept", "anomaly": "concept", "paradox": "concept",
    "category": "concept", "context": "concept", "finding": "concept",
    "strategy": "concept", "rule": "concept", "scenario": "concept",
    "analogy": "concept", "anti-pattern": "concept", "measurement": "concept",
    "metric": "concept", "distribution": "concept", "function": "concept",
    "mathematical function": "concept", "mathematical structure": "concept",
    "pde": "concept", "interpretation": "concept", "metacognitive error": "concept",
    "rebuttal": "concept", "natural phenomenon": "concept", "biological_process": "concept",
    "cognitive_function": "concept",
    # technique — tools, methods, algorithms, architectures, frameworks, protocols
    "technique": "technique", "tool": "technique", "method": "technique",
    "framework": "technique", "protocol": "technique", "algorithm": "technique",
    "application": "technique", "architecture": "technique", "format": "technique",
    "prompting type": "technique", "hybrid architecture": "technique",
    "task": "technique", "benchmark": "technique", "study_type": "technique",
    "study type": "technique", "intervention": "technique", "treatment": "technique",
    "component": "technique", "resource": "technique", "data": "technique",
    # theory — theories, models, theorems, laws, paradigms, schools of thought
    "theory": "theory", "theorem": "theory", "conjecture": "theory",
    "law": "theory", "model": "theory", "paradigm": "theory",
    "school_of_thought": "theory", "school of thought": "theory",
    "doctrine": "theory", "movement": "theory",
    # person — individual people
    "person": "person", "people": "person", "role": "person",
    # field — academic/professional disciplines
    "field": "field", "domain": "field",
    "field/technique": "field",
    # system — platforms, organizations, companies, services, hardware, networks
    "system": "system", "platform": "system", "organization": "system",
    "company": "system", "service": "system", "hardware": "system",
    "network": "system", "database": "system", "library": "system",
    "product": "system", "project": "system", "program": "system",
    "institution": "system", "regulatory body": "system",
    "political entity": "system", "market": "system",
    # anatomy — anatomical structures and brain regions
    "anatomy": "anatomy", "anatomical_structure": "anatomy",
    "anatomical structure": "anatomy", "anatomical": "anatomy",
    "brain_region": "anatomy", "brain region": "anatomy",
    "brain_structure": "anatomy", "neural_pathway": "anatomy",
    "structure": "anatomy",
    # biology — biological entities (proteins, genes, molecules, neurotransmitters)
    "protein": "biology", "gene": "biology", "molecule": "biology",
    "neurotransmitter": "biology", "neuromodulator": "biology",
    "hormone": "biology", "receptor": "biology", "drug": "biology",
    "drug_class": "biology", "substance": "biology", "material": "biology",
    "species": "biology",
    # event — historical events, experiments, studies
    "event": "event", "experiment": "event", "study": "event",
    "clinical_trial": "event", "case": "event",
    # publication — books, papers, documents, essays
    "publication": "publication", "book": "publication", "paper": "publication",
    "document": "publication", "essay": "publication", "journal": "publication",
    "guide": "publication", "blog": "publication",
    # entity (catch-all fallback — normalize to concept)
    "entity": "concept", "type": "concept", "example": "concept",
    "research": "concept", "technology": "technique",
    "storage": "system", "standard": "technique",
    # award / prize
    "award": "event", "prize": "event",
    # location / demographic
    "location": "concept", "country": "concept", "demographic": "concept",
    "group": "concept", "language": "concept", "time period": "concept",
    # design system (normalize to technique)
    "design system": "technique",
    # risk / disorder / disease
    "risk_factor": "concept", "disorder": "concept", "disease": "concept",
    # particle / physical
    "particle": "concept",
    # constitutional provision
    "constitutional provision": "theory",
    # act (legal)
    "act": "theory",
    # source
    "source": "publication",
}


def normalize_relation(r):
    return RELATION_MAP.get(r, "relates_to")


def normalize_entity_type(t):
    return ENTITY_TYPE_MAP.get(t, "concept")


def normalize_entity_name(name: str) -> str:
    """
    Normalize entity name to prevent fragmentation.
    Converts 'decision-making', 'decision_making' → 'decision making'.
    Preserves meaningful hyphens: 'b-tree', '5-ht1a', 't-cell'.

    Mirrors the same function in inject-causal-edges.py.
    """
    import re
    name = name.lower()
    name = name.replace('_', ' ')
    prev = None
    while prev != name:
        prev = name
        name = re.sub(r'([a-z]{3,})-([a-z]{3,})', r'\1 \2', name)
        name = re.sub(r'([a-z0-9]{2,})-([a-z]{3,})', r'\1 \2', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def init_tables(conn):
    """Create entity and relation tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,  -- concept, person, theory, technique, field
            source_file TEXT NOT NULL,
            UNIQUE(name, source_file)
        );
        CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);

        CREATE TABLE IF NOT EXISTS relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_entity TEXT NOT NULL,
            relation TEXT NOT NULL,  -- relates_to, builds_on, contradicts, applies, implements, extends
            target_entity TEXT NOT NULL,
            source_file TEXT NOT NULL,
            confidence REAL DEFAULT 0.8,
            UNIQUE(source_entity, relation, target_entity, source_file)
        );
        CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity);
        CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity);
    """)


def ollama_generate(prompt, model=GRAPH_MODEL):
    """Call Ollama generate API. Returns the response text."""
    url = f"{OLLAMA_BASE}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"num_ctx": 4096, "temperature": 0.1}
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["response"]
    except Exception as e:
        print(f"  Ollama error: {e}", file=sys.stderr)
        return ""


def extract_entities_and_relations(content, filename):
    """Use Ollama to extract entities and relations from a note."""
    # Truncate to avoid context overflow
    truncated = content[:3000]

    prompt = f"""Extract entities and relationships from this knowledge note. Return ONLY valid JSON, no other text.

Note filename: {filename}
---
{truncated}
---

Return this exact JSON structure:
{{"entities": [{{"name": "entity name (lowercase)", "type": "concept|person|theory|technique|field"}}], "relations": [{{"source": "entity1", "relation": "relates_to|builds_on|contradicts|applies|implements|extends", "target": "entity2"}}]}}

Rules:
- Entity names should be canonical (e.g. "attention mechanism" not "attention mechanisms")
- Only extract entities that are substantive concepts, not generic words
- Relations should reflect the actual relationship described in the text
- Maximum 10 entities and 10 relations per note
- Return ONLY the JSON object, nothing else"""

    response = ollama_generate(prompt)

    # Parse JSON from response (handle markdown code blocks)
    response = response.strip()
    if response.startswith("```"):
        response = re.sub(r'^```\w*\n?', '', response)
        response = re.sub(r'\n?```$', '', response)

    try:
        data = json.loads(response)
        return data.get("entities", []), data.get("relations", [])
    except json.JSONDecodeError:
        # Try to find JSON in the response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return data.get("entities", []), data.get("relations", [])
            except json.JSONDecodeError:
                pass
        return [], []


def index_notes(conn, root_dir, incremental=False):
    """Extract entities and relations from markdown notes in root_dir."""
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Directory not found: {root_dir}")
        return

    notes = sorted(root_path.rglob("*.md"))
    print(f"Found {len(notes)} knowledge notes")

    if incremental:
        # Get already-processed files
        existing = set(r[0] for r in conn.execute(
            "SELECT DISTINCT source_file FROM entities"
        ).fetchall())
        notes = [n for n in notes if n.name not in existing]
        print(f"  {len(notes)} new notes to process")

    if not notes:
        print("Nothing to index.")
        return

    total_entities = 0
    total_relations = 0

    for i, note_path in enumerate(notes):
        name = note_path.name
        content = note_path.read_text(encoding="utf-8", errors="replace")

        # Skip very short notes
        if len(content) < 100:
            continue

        print(f"  [{i+1}/{len(notes)}] {name}", end="", flush=True)

        entities, relations = extract_entities_and_relations(content, name)

        for ent in entities:
            if isinstance(ent, str):
                ent_name = normalize_entity_name(ent.strip())
                ent_type = "concept"
            else:
                ent_name = normalize_entity_name(ent.get("name", "").strip())
                ent_type = ent.get("type", "concept").strip().lower()
            if ent_name and len(ent_name) > 1:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO entities (name, type, source_file) VALUES (?, ?, ?)",
                        (ent_name, ent_type, name)
                    )
                    total_entities += 1
                except sqlite3.Error:
                    pass

        for rel in relations:
            if isinstance(rel, str):
                continue  # skip malformed relation
            src = normalize_entity_name(rel.get("source", "").strip())
            tgt = normalize_entity_name(rel.get("target", "").strip())
            relation = rel.get("relation", "relates_to").strip().lower()
            if src and tgt and src != tgt:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO relations (source_entity, relation, target_entity, source_file) VALUES (?, ?, ?, ?)",
                        (src, relation, tgt, name)
                    )
                    total_relations += 1
                except sqlite3.Error:
                    pass

        conn.commit()
        print(f" → {len(entities)}e {len(relations)}r")

    print(f"\nDone. {total_entities} entities, {total_relations} relations extracted.")


def query_entity(conn, query, hops=1):
    """Find an entity and traverse its connections."""
    query_lower = query.lower()

    # Find matching entities
    entities = conn.execute(
        "SELECT DISTINCT name, type FROM entities WHERE name LIKE ?",
        (f"%{query_lower}%",)
    ).fetchall()

    if not entities:
        print(f"No entities matching '{query}'")
        return

    print(f"Found {len(entities)} matching entities:\n")

    visited = set()
    to_visit = [(e[0], 0) for e in entities]

    while to_visit:
        entity_name, depth = to_visit.pop(0)
        if entity_name in visited or depth > hops:
            continue
        visited.add(entity_name)

        indent = "  " * depth

        # Get entity info
        sources = conn.execute(
            "SELECT type, source_file FROM entities WHERE name = ?",
            (entity_name,)
        ).fetchall()

        if sources:
            types = set(s[0] for s in sources)
            files = set(s[1] for s in sources)
            print(f"{indent}[{', '.join(types)}] {entity_name}")
            print(f"{indent}  appears in: {', '.join(sorted(files)[:5])}")

        # Get outgoing relations
        outgoing = conn.execute(
            "SELECT relation, target_entity, source_file FROM relations WHERE source_entity = ?",
            (entity_name,)
        ).fetchall()

        # Get incoming relations
        incoming = conn.execute(
            "SELECT source_entity, relation, source_file FROM relations WHERE target_entity = ?",
            (entity_name,)
        ).fetchall()

        if outgoing:
            for rel, target, src in outgoing:
                print(f"{indent}  → {rel} → {target} (in {src})")
                if depth < hops and target not in visited:
                    to_visit.append((target, depth + 1))

        if incoming:
            for source, rel, src in incoming:
                print(f"{indent}  ← {source} ← {rel} (in {src})")
                if depth < hops and source not in visited:
                    to_visit.append((source, depth + 1))

        print()


def fuzzy_entity_lookup(conn, query: str, max_candidates: int = 8) -> list[str]:
    """
    Fuzzy entity lookup using LIKE substring matching, then prefix matching.
    Returns a ranked list of candidate entity names from the graph.

    Priority order:
    1. Exact match
    2. Substring match (LIKE %query%)
    3. Word-boundary prefix match (space-separated tokens)

    This is the entity normalization layer for the NL→graph-query pipeline (H-010).
    Mitigates the plurality/synonym problem (e.g. "neural network" vs "neural networks").
    """
    query_norm = normalize_entity_name(query)

    candidates = []
    seen = set()

    # 1. Exact match
    rows = conn.execute(
        "SELECT DISTINCT name FROM entities WHERE name = ? LIMIT ?",
        (query_norm, max_candidates)
    ).fetchall()
    for row in rows:
        if row[0] not in seen:
            candidates.append(row[0])
            seen.add(row[0])

    # 2. Substring match
    if len(candidates) < max_candidates:
        rows = conn.execute(
            "SELECT DISTINCT name FROM entities WHERE name LIKE ? LIMIT ?",
            (f"%{query_norm}%", max_candidates - len(candidates))
        ).fetchall()
        for row in rows:
            if row[0] not in seen:
                candidates.append(row[0])
                seen.add(row[0])

    # 3. Per-word token match (catch plurals and compound variants)
    if len(candidates) < max_candidates:
        tokens = [t for t in query_norm.split() if len(t) > 3]
        for token in tokens[:3]:
            rows = conn.execute(
                "SELECT DISTINCT name FROM entities WHERE name LIKE ? LIMIT ?",
                (f"%{token}%", max_candidates - len(candidates))
            ).fetchall()
            for row in rows:
                if row[0] not in seen:
                    candidates.append(row[0])
                    seen.add(row[0])
            if len(candidates) >= max_candidates:
                break

    return candidates[:max_candidates]


def nl_to_graph_query(natural_language_question: str, num_ctx: int = 4096) -> dict:
    """
    Translate a natural language question to a structured graph query.
    Uses Ollama (small/fast model) — maps freeform question to:
        {entity: str, relation_filter: str | None, max_hops: int, intent: str}

    Implements the neuro-symbolic integration pattern (H-010):
    LLM → formal query → symbolic graph execution.
    The LLM provides flexibility; graph execution provides exactness.

    Returns dict with keys: entity, relation_filter, max_hops, intent, raw_question
    """
    # Use a fast local model for NL parsing — haiku-style role
    NL_MODEL = os.environ.get("NL_QUERY_MODEL", GRAPH_MODEL)

    prompt = f"""You are a knowledge graph query parser. Convert the user's natural language question
into a structured graph query. Return ONLY valid JSON, nothing else.

The knowledge graph contains academic/technical vault notes with entities like:
"attention mechanism", "spaced repetition", "hormesis", "curriculum learning",
"convex", "linesheet", "dopamine", "reinforcement learning", etc.

Question: {natural_language_question}

Return this exact JSON structure:
{{
  "entity": "primary entity to look up (lowercase, 2-4 words max)",
  "relation_filter": null or one of ["builds_on", "contradicts", "applies_to", "implements", "extends", "part_of", "uses", "enables", "related_to", "causes", "measured_by", "equivalent_to", "produces", "involves"],
  "max_hops": 1 or 2,
  "intent": "one sentence describing what the user wants to know"
}}

Rules:
- entity: the most specific single concept the question is about (not a sentence)
- relation_filter: null unless the question specifically asks about a relation type
- max_hops: 2 if the question is multi-hop ("what connects X to Y?"), else 1
- Use null for relation_filter when the question asks about all connections
- Return ONLY the JSON, no preamble or explanation"""

    response = ollama_generate(prompt, model=NL_MODEL)
    response = response.strip()

    # Strip markdown code fences if present
    if response.startswith("```"):
        response = re.sub(r'^```\w*\n?', '', response)
        response = re.sub(r'\n?```$', '', response)

    try:
        parsed = json.loads(response)
        return {
            "entity":          parsed.get("entity", ""),
            "relation_filter": parsed.get("relation_filter", None),
            "max_hops":        int(parsed.get("max_hops", 1)),
            "intent":          parsed.get("intent", natural_language_question),
            "raw_question":    natural_language_question,
        }
    except (json.JSONDecodeError, ValueError):
        # Fallback: extract likely entity keywords from question (skip question words and short tokens)
        _QUESTION_WORDS = {
            "what", "which", "who", "where", "when", "why", "how", "does", "is", "are",
            "was", "were", "will", "would", "could", "should", "the", "and", "with",
            "that", "this", "from", "have", "has", "about", "between", "connect", "connects",
            "connected", "relate", "relates", "related", "show", "shows", "explain",
        }
        tokens = re.findall(r"[a-zA-Z]+", natural_language_question.lower())
        entity_tokens = [t for t in tokens if len(t) > 4 and t not in _QUESTION_WORDS]
        fallback_entity = " ".join(entity_tokens[:3]) if entity_tokens else natural_language_question[:40]
        return {
            "entity":          fallback_entity,
            "relation_filter": None,
            "max_hops":        1,
            "intent":          f"Find connections for: {natural_language_question}",
            "raw_question":    natural_language_question,
        }


def ask_graph(conn, question: str, hops: int | None = None) -> None:
    """
    Natural language interface to the knowledge graph.
    Implements H-010: NL-to-graph-query translation layer for vault-graph.py.

    Pipeline:
    1. NL question → structured query (LLM translation)
    2. Entity name → fuzzy lookup in graph DB (exact + substring + token matching)
    3. Symbolic graph traversal on resolved entity
    4. Formatted answer with provenance

    Usage: python3 vault-graph.py ask <vault_root> "What does hormesis connect to in ML?"
    """
    print(f"\nQuery: {question}")
    print("─" * 60)

    # Step 1: Parse NL → structured query
    print("Parsing question...", end=" ", flush=True)
    parsed = nl_to_graph_query(question)
    entity_hint  = parsed["entity"]
    rel_filter   = parsed["relation_filter"]
    max_hops_q   = hops if hops is not None else parsed["max_hops"]
    intent       = parsed["intent"]

    print(f"done")
    print(f"Entity hint:     {entity_hint!r}")
    print(f"Relation filter: {rel_filter!r}")
    print(f"Max hops:        {max_hops_q}")
    print(f"Intent:          {intent}")
    print()

    # Step 2: Fuzzy entity lookup
    if not entity_hint.strip():
        print("Could not extract a query entity from the question.")
        return

    candidates = fuzzy_entity_lookup(conn, entity_hint)

    if not candidates:
        print(f"No entities found matching '{entity_hint}' in the knowledge graph.")
        print("Tip: Try a more specific term, or check 'vault-graph.py stats' for entity coverage.")
        return

    print(f"Resolved entity candidates ({len(candidates)}):")
    for i, c in enumerate(candidates[:5]):
        n_rels = conn.execute(
            "SELECT COUNT(*) FROM relations WHERE source_entity=? OR target_entity=?",
            (c, c)
        ).fetchone()[0]
        print(f"  [{i+1}] {c!r} ({n_rels} relations)")
    print()

    # Step 3: Graph traversal on top candidate
    top_entity = candidates[0]
    print(f"Traversing graph for: {top_entity!r} (hops={max_hops_q})")
    if rel_filter:
        print(f"Filtering to relation type: {rel_filter!r}")
    print("─" * 60)

    visited = set()
    to_visit = [(top_entity, 0)]
    results = []

    while to_visit:
        entity_name, depth = to_visit.pop(0)
        if entity_name in visited or depth > max_hops_q:
            continue
        visited.add(entity_name)

        indent = "  " * depth

        # Entity info
        sources = conn.execute(
            "SELECT DISTINCT type, source_file FROM entities WHERE name = ?",
            (entity_name,)
        ).fetchall()

        if sources or depth == 0:
            types = sorted(set(s[0] for s in sources)) if sources else ["unknown"]
            files = sorted(set(s[1] for s in sources)) if sources else []
            short_files = [Path(f).stem for f in files[:3]]
            print(f"{indent}[{', '.join(types)}] {entity_name}")
            if short_files:
                print(f"{indent}  notes: {', '.join(short_files)}")

        # Outgoing relations (with optional filter)
        if rel_filter:
            outgoing = conn.execute(
                "SELECT relation, target_entity, source_file FROM relations "
                "WHERE source_entity = ? AND relation = ?",
                (entity_name, rel_filter)
            ).fetchall()
        else:
            outgoing = conn.execute(
                "SELECT relation, target_entity, source_file FROM relations "
                "WHERE source_entity = ?",
                (entity_name,)
            ).fetchall()

        # Incoming relations (with optional filter)
        if rel_filter:
            incoming = conn.execute(
                "SELECT source_entity, relation, source_file FROM relations "
                "WHERE target_entity = ? AND relation = ?",
                (entity_name, rel_filter)
            ).fetchall()
        else:
            incoming = conn.execute(
                "SELECT source_entity, relation, source_file FROM relations "
                "WHERE target_entity = ?",
                (entity_name,)
            ).fetchall()

        if outgoing:
            for rel, target, src in outgoing:
                print(f"{indent}  → {rel} → {target}")
                if depth < max_hops_q and target not in visited:
                    to_visit.append((target, depth + 1))
                results.append({"direction": "out", "entity": entity_name, "rel": rel, "other": target})

        if incoming:
            for source, rel, src in incoming:
                print(f"{indent}  ← {source} ← {rel}")
                if depth < max_hops_q and source not in visited:
                    to_visit.append((source, depth + 1))
                results.append({"direction": "in", "entity": entity_name, "rel": rel, "other": source})

        if not outgoing and not incoming and depth > 0:
            print(f"{indent}  (no connections at this depth)")

        print()

    # Summary
    n_nodes = len(visited)
    n_edges = len(results)
    print(f"─" * 60)
    print(f"Traversal complete: {n_nodes} entities, {n_edges} relations")
    if len(candidates) > 1:
        print(f"\nOther matching entities (not traversed): {', '.join(repr(c) for c in candidates[1:5])}")
        print("Re-run with 'query <root> <entity_name>' for a specific entity.")


def show_stats(conn):
    """Show graph statistics."""
    entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    relation_count = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
    unique_entities = conn.execute("SELECT COUNT(DISTINCT name) FROM entities").fetchone()[0]
    files_covered = conn.execute("SELECT COUNT(DISTINCT source_file) FROM entities").fetchone()[0]

    print(f"Knowledge Graph Stats:")
    print(f"  Entities:        {entity_count} ({unique_entities} unique)")
    print(f"  Relations:       {relation_count}")
    print(f"  Notes covered:   {files_covered}")

    print(f"\nEntity types:")
    for row in conn.execute("SELECT type, COUNT(*) as c FROM entities GROUP BY type ORDER BY c DESC"):
        print(f"  {row[0]}: {row[1]}")

    print(f"\nRelation types:")
    for row in conn.execute("SELECT relation, COUNT(*) as c FROM relations GROUP BY relation ORDER BY c DESC"):
        print(f"  {row[0]}: {row[1]}")

    print(f"\nMost connected entities:")
    for row in conn.execute("""
        SELECT name, COUNT(*) as connections FROM (
            SELECT source_entity as name FROM relations
            UNION ALL
            SELECT target_entity as name FROM relations
        ) GROUP BY name ORDER BY connections DESC LIMIT 10
    """):
        print(f"  {row[0]}: {row[1]} connections")


def export_graph(conn, top=300, min_connections=2):
    """Export graph as JSON for force-graph visualization."""
    # Get entities ranked by connection count
    entity_connections = {}
    for row in conn.execute("""
        SELECT name, COUNT(*) as connections FROM (
            SELECT source_entity as name FROM relations
            UNION ALL
            SELECT target_entity as name FROM relations
        ) GROUP BY name ORDER BY connections DESC
    """):
        if row[1] >= min_connections:
            entity_connections[row[0]] = row[1]

    # Take top N
    top_entities = dict(list(entity_connections.items())[:top])
    entity_set = set(top_entities.keys())

    # Build nodes with type and source info
    nodes = []
    for name in entity_set:
        rows = conn.execute(
            "SELECT type, source_file FROM entities WHERE name = ?", (name,)
        ).fetchall()
        types = [r[0] for r in rows]
        sources = sorted(set(r[1] for r in rows))
        # Pick most common type
        primary_type = max(set(types), key=types.count) if types else "concept"
        nodes.append({
            "id": name,
            "type": normalize_entity_type(primary_type),
            "connections": top_entities[name],
            "sources": sources[:5],
        })

    # Build links (only between entities in the set)
    links = []
    seen_links = set()
    for row in conn.execute("SELECT source_entity, relation, target_entity FROM relations"):
        src, rel, tgt = row
        if src in entity_set and tgt in entity_set:
            key = (src, tgt)
            if key not in seen_links:
                seen_links.add(key)
                links.append({
                    "source": src,
                    "target": tgt,
                    "relation": normalize_relation(rel),
                })

    result = {
        "metadata": {
            "total_entities": len(nodes),
            "total_relations": len(links),
            "total_notes": conn.execute("SELECT COUNT(DISTINCT source_file) FROM entities").fetchone()[0],
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "nodes": sorted(nodes, key=lambda n: n["connections"], reverse=True),
        "links": links,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def normalize_db_types(conn):
    """Migrate existing entity types and relation types in DB to canonical values."""
    # 1. Normalize entity types
    rows = conn.execute("SELECT id, type FROM entities").fetchall()
    updates = []
    changed = 0
    for eid, etype in rows:
        canonical = ENTITY_TYPE_MAP.get(etype, "concept")
        if canonical != etype:
            updates.append((canonical, eid))
            changed += 1
    if updates:
        conn.executemany("UPDATE entities SET type = ? WHERE id = ?", updates)
        conn.commit()
    # Report final entity type distribution
    type_counts = conn.execute(
        "SELECT type, COUNT(*) FROM entities GROUP BY type ORDER BY COUNT(*) DESC"
    ).fetchall()
    print(f"Normalized {changed} entity type labels.")
    print(f"Distinct entity types after normalization: {len(type_counts)}")
    for t, c in type_counts:
        print(f"  {c:5d}  {t}")

    # 2. Normalize relation types — handle UNIQUE constraint by deleting conflicts
    rel_rows = conn.execute(
        "SELECT id, source_entity, relation, target_entity, source_file FROM relations"
    ).fetchall()
    rel_changed = 0
    rel_merged = 0
    rel_unchanged_types = {}
    for rid, src, rel, tgt, sf in rel_rows:
        canonical = RELATION_MAP.get(rel, None)
        if canonical is None:
            rel_unchanged_types[rel] = rel_unchanged_types.get(rel, 0) + 1
            canonical = "relates_to"
        if canonical != rel:
            # Check if a canonical row already exists — if so, delete this one
            existing = conn.execute(
                "SELECT id FROM relations WHERE source_entity=? AND relation=? AND target_entity=? AND source_file=?",
                (src, canonical, tgt, sf)
            ).fetchone()
            if existing:
                conn.execute("DELETE FROM relations WHERE id=?", (rid,))
                rel_merged += 1
            else:
                conn.execute(
                    "UPDATE relations SET relation=? WHERE id=?", (canonical, rid)
                )
                rel_changed += 1
    conn.commit()
    rel_counts = conn.execute(
        "SELECT relation, COUNT(*) FROM relations GROUP BY relation ORDER BY COUNT(*) DESC"
    ).fetchall()
    print(f"\nNormalized {rel_changed} relation type labels, merged {rel_merged} duplicates.")
    print(f"Distinct relation types after normalization: {len(rel_counts)}")
    for r, c in rel_counts:
        print(f"  {c:5d}  {r}")
    if rel_unchanged_types:
        top_unknown = sorted(rel_unchanged_types.items(), key=lambda x: -x[1])[:10]
        print(f"\nTop unknown relations mapped to 'relates_to' ({len(rel_unchanged_types)} types):")
        for r, c in top_unknown:
            print(f"  {c:4d}  '{r}'")


def prune_orphans(conn, dry_run=False):
    """
    Remove entities with no relations and relations pointing to non-existent entities.

    Two passes:
      1. Dangling relations — source_entity or target_entity not present in entities.name
      2. Orphan entities — entities.name not referenced in any relation (source or target)

    Reports counts for both passes. With --dry-run prints counts without deleting.
    """
    # --- Pass 1: dangling relations ---
    dangling = conn.execute("""
        SELECT id FROM relations
        WHERE source_entity NOT IN (SELECT DISTINCT name FROM entities)
           OR target_entity  NOT IN (SELECT DISTINCT name FROM entities)
    """).fetchall()
    dangling_ids = [r[0] for r in dangling]

    # --- Pass 2: orphan entities (zero relations in either direction) ---
    orphans = conn.execute("""
        SELECT DISTINCT name FROM entities
        WHERE name NOT IN (SELECT DISTINCT source_entity FROM relations)
          AND name NOT IN (SELECT DISTINCT target_entity  FROM relations)
    """).fetchall()
    orphan_names = [r[0] for r in orphans]

    print(f"Dangling relations (reference missing entities): {len(dangling_ids)}")
    print(f"Orphan entities (zero relations):               {len(orphan_names)}")

    if dry_run:
        print("[dry-run] No changes written.")
        if dangling_ids:
            sample = dangling_ids[:5]
            rows = conn.execute(
                f"SELECT source_entity, relation, target_entity FROM relations WHERE id IN ({','.join('?'*len(sample))})",
                sample
            ).fetchall()
            print("  Sample dangling relations:")
            for src, rel, tgt in rows:
                print(f"    {src!r} --{rel}--> {tgt!r}")
        if orphan_names:
            print(f"  Sample orphan entities: {orphan_names[:10]}")
        return

    # Delete dangling relations
    if dangling_ids:
        conn.execute(
            f"DELETE FROM relations WHERE id IN ({','.join('?'*len(dangling_ids))})",
            dangling_ids
        )

    # Delete orphan entities (all rows for each name)
    if orphan_names:
        conn.execute(
            f"DELETE FROM entities WHERE name IN ({','.join('?'*len(orphan_names))})",
            orphan_names
        )

    conn.commit()

    total_removed = len(dangling_ids) + len(orphan_names)
    print(f"Removed {len(dangling_ids)} dangling relations and {len(orphan_names)} orphan entities.")
    print(f"Total pruned: {total_removed}")

    # Post-prune stats
    remaining_entities = conn.execute("SELECT COUNT(DISTINCT name) FROM entities").fetchone()[0]
    remaining_relations = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
    print(f"Graph after prune: {remaining_entities} unique entities, {remaining_relations} relations")


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge graph extraction for your notes. "
        "Extracts entities and relationships using a local Ollama model."
    )
    parser.add_argument(
        "command", choices=["index", "query", "ask", "stats", "export", "normalize-db", "prune"],
        help="Command to run ('ask' accepts natural language questions)"
    )
    _vault_root_default = str(Path(__file__).resolve().parent.parent)
    parser.add_argument(
        "root", nargs="?", default=_vault_root_default,
        help="Root directory of your notes (default: vault root)"
    )
    parser.add_argument(
        "entity", nargs="?", default=None,
        help="Entity name to search for (query command only)"
    )
    parser.add_argument("--incremental", action="store_true", help="Only process new notes")
    parser.add_argument("--hops", type=int, default=1, help="Traversal depth (default: 1)")
    parser.add_argument("--top", type=int, default=300, help="Top N entities for export (default: 300)")
    parser.add_argument("--min-connections", type=int, default=2, help="Min connections for export (default: 2)")
    parser.add_argument("--db", type=str, default=None, help="Database path (default: auto per root dir)")
    parser.add_argument("--dry-run", action="store_true", help="prune: show what would be removed without deleting")

    args = parser.parse_args()

    root = os.path.realpath(args.root)
    if args.db:
        db_path = args.db
    else:
        db_path = db_path_for_root(root)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    init_tables(conn)

    if args.command == "index":
        index_notes(conn, root, incremental=args.incremental)
    elif args.command == "query":
        if not args.entity:
            print("Usage: vault-graph.py query <root> <entity>")
            return
        query_entity(conn, args.entity, args.hops)
    elif args.command == "ask":
        if not args.entity:
            print("Usage: vault-graph.py ask [root] \"natural language question\"")
            print("Example: vault-graph.py ask \"What connects hormesis to curriculum learning?\"")
            return
        # For 'ask', args.entity is the natural language question
        # Reassemble if split across multiple words (shell may split it)
        ask_graph(conn, args.entity, hops=args.hops if args.hops != 1 else None)
    elif args.command == "stats":
        show_stats(conn)
    elif args.command == "export":
        export_graph(conn, top=args.top, min_connections=args.min_connections)
    elif args.command == "normalize-db":
        normalize_db_types(conn)
    elif args.command == "prune":
        prune_orphans(conn, dry_run=args.dry_run)

    conn.close()


if __name__ == "__main__":
    main()
