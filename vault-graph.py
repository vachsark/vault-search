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
    GRAPH_MODEL    Model for extraction (default: qwen3:8b)
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
GRAPH_MODEL = os.environ.get("GRAPH_MODEL", "qwen3:8b")


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
                ent_name = ent.strip().lower()
                ent_type = "concept"
            else:
                ent_name = ent.get("name", "").strip().lower()
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
            src = rel.get("source", "").strip().lower()
            tgt = rel.get("target", "").strip().lower()
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
    """Migrate existing entity types in DB to canonical types using ENTITY_TYPE_MAP."""
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
    # Report final type distribution
    type_counts = conn.execute(
        "SELECT type, COUNT(*) FROM entities GROUP BY type ORDER BY COUNT(*) DESC"
    ).fetchall()
    print(f"Normalized {changed} entity type labels.")
    print(f"Distinct types after normalization: {len(type_counts)}")
    for t, c in type_counts:
        print(f"  {c:5d}  {t}")


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge graph extraction for your notes. "
        "Extracts entities and relationships using a local Ollama model."
    )
    parser.add_argument(
        "command", choices=["index", "query", "stats", "export", "normalize-db"],
        help="Command to run"
    )
    parser.add_argument(
        "root", nargs="?", default=".",
        help="Root directory of your notes (default: current directory)"
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

    args = parser.parse_args()

    root = os.path.realpath(args.root)
    if args.db:
        db_path = args.db
    else:
        db_path = db_path_for_root(root)

    conn = sqlite3.connect(db_path)
    init_tables(conn)

    if args.command == "index":
        index_notes(conn, root, incremental=args.incremental)
    elif args.command == "query":
        if not args.entity:
            print("Usage: vault-graph.py query <root> <entity>")
            return
        query_entity(conn, args.entity, args.hops)
    elif args.command == "stats":
        show_stats(conn)
    elif args.command == "export":
        export_graph(conn, top=args.top, min_connections=args.min_connections)
    elif args.command == "normalize-db":
        normalize_db_types(conn)

    conn.close()


if __name__ == "__main__":
    main()
