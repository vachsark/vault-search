#!/usr/bin/env python3
"""
causal-trace.py — Trace causal/mechanistic chains through the knowledge graph.

Unlike knowledge-path.py (shortest path between any two entities), this tool
follows DIRECTIONAL, TYPED edges to build mechanistic explanations:

  "Why does X lead to Y?"
  → X causes A (via mechanism M1)
  → A enables B (via mechanism M2)
  → B implements Y (in project Z)

This is the causal model layer from the vault future architecture doc.

Usage:
    python3 causal-trace.py "dopamine" "habit formation"
    python3 causal-trace.py "loss aversion" "linesheet pricing" --include-code
    python3 causal-trace.py "reward prediction error" "FitnessRewards streak" --explain
"""
import argparse
import glob
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict, deque
from pathlib import Path

VAULT_DIR = os.environ.get("VAULT_DIR", "/home/veech/Documents/TestVault")
KNOWLEDGE_DIR = os.path.join(VAULT_DIR, "Knowledge")

# Causal relation types (directional — source causes/enables/leads to target)
CAUSAL_RELATIONS = {
    "causes", "enables", "leads_to", "produces", "triggers",
    "inhibits", "prevents", "blocks",
    "implements", "applies", "applies_to",
    "builds_on", "extends", "uses",
    # Added 2026-03-21 — injected via inject-causal-edges.py
    "depends_on", "explains", "predicts", "modulates", "maps_to",
}

# Non-causal (bidirectional, structural)
STRUCTURAL_RELATIONS = {
    "relates_to", "related_to", "part_of", "is_part_of",
    "includes", "is_a", "contrasts_with",
}

# All relations that indicate a directional dependency
DIRECTIONAL = CAUSAL_RELATIONS | {"builds_on", "extends", "implements", "applies_to", "depends_on", "explains", "predicts", "modulates", "maps_to"}

# Relation normalization map: raw graph strings → canonical relation types
# Handles typos and space variants written directly in notes
_RELATION_NORMALIZE: dict[str, str] = {
    "applies, to":  "applies_to",
    "applies to":   "applies_to",
    "builds on":    "builds_on",
    "build on":     "builds_on",
    "leads to":     "leads_to",
    "led to":       "leads_to",
    "maps to":      "maps_to",
    "maps onto":    "maps_to",
    "depends on":   "depends_on",
    "based on":     "depends_on",
    "part of":      "part_of",
    "is a":         "is_a",
    "relates to":   "relates_to",
    "related to":   "related_to",
    "contrasts with": "contrasts_with",
}


def _normalize_relation(rel: str) -> str:
    """Normalize a raw relation string from the graph to a canonical form."""
    r = rel.lower().strip()
    # Check explicit map first
    if r in _RELATION_NORMALIZE:
        return _RELATION_NORMALIZE[r]
    # Generic normalization: replace spaces with underscores
    return r.replace(" ", "_")


def load_graph(db_path: str) -> tuple[dict, dict]:
    """Load the graph with typed, directional edges."""
    conn = sqlite3.connect(db_path)

    # Forward graph: source → [(target, relation, source_file)]
    forward: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    # Reverse graph: target → [(source, relation, source_file)]
    reverse: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for row in conn.execute("SELECT source_entity, relation, target_entity, source_file FROM relations"):
        s, r, t, f = row[0].lower().strip(), _normalize_relation(row[1]), row[2].lower().strip(), row[3]
        if s and t and s != t:
            forward[s].append((t, r, f))
            reverse[t].append((s, r, f))

    conn.close()
    return forward, reverse


def fuzzy_match(entity: str, graph_keys: set[str],
                forward: dict | None = None, reverse: dict | None = None) -> str | None:
    """Find the best match for an entity name in the graph.

    Priority:
      1. Exact match (always preferred over substring matches)
      2. Among substring candidates, prefer the most-connected entity

    This ensures that user query "decision making" binds to the exact node
    "decision making" rather than "bayesian decision making" (which has more
    edges but is a different concept).

    Fallback: word-overlap match for multi-word queries with no substring match.
    """
    entity = entity.lower().strip()

    def _connection_count(key: str) -> int:
        """Sum of forward + reverse edges for a graph key."""
        if forward is None and reverse is None:
            return 0
        fwd = len(forward.get(key, [])) if forward else 0
        rev = len(reverse.get(key, [])) if reverse else 0
        return fwd + rev

    # Exact match takes absolute priority — no ambiguity
    if entity in graph_keys:
        return entity

    # Collect all substring candidates
    candidates = [k for k in graph_keys if entity in k]
    if candidates:
        # Prefer most-connected among substring matches
        return max(candidates, key=lambda k: _connection_count(k))

    # Word overlap match — pick highest overlap, break ties by connection count
    words = set(entity.split())
    best, best_score, best_conn = None, 0, -1
    for k in graph_keys:
        k_words = set(k.split())
        overlap = len(words & k_words)
        conn_cnt = _connection_count(k)
        if overlap > best_score or (overlap == best_score and conn_cnt > best_conn):
            best, best_score, best_conn = k, overlap, conn_cnt
    return best if best_score >= 2 else None


def trace_causal_chain(forward: dict, reverse: dict, start: str, end: str,
                       max_depth: int = 6, prefer_causal: bool = True,
                       allow_reverse: bool = False) -> list[dict] | None:
    """Find a causal chain from start to end using BFS with relation-type priority.

    allow_reverse: also traverse edges in reverse direction when the forward
    search fails. This handles cases where the edge in the graph points from
    end→start rather than start→end (e.g. "prospect theory → wholesale pricing"
    where the stored edge points the other way).
    """
    all_keys = set(forward.keys()) | set(reverse.keys())

    start_match = fuzzy_match(start, all_keys, forward, reverse)
    end_match = fuzzy_match(end, all_keys, forward, reverse)

    if not start_match or not end_match:
        return None

    def _bfs(graph: dict, src: str, dst: str, reversed_edges: bool) -> list[dict] | None:
        """BFS through `graph` from src to dst."""
        queue: deque[tuple[str, list[dict]]] = deque([(src, [])])
        visited: set[str] = {src}

        while queue:
            node, chain = queue.popleft()
            if len(chain) >= max_depth:
                continue

            edges = graph.get(node, [])
            if prefer_causal:
                edges = sorted(edges, key=lambda e: 0 if e[1] in CAUSAL_RELATIONS else 1)

            for target, relation, source_file in edges:
                if reversed_edges:
                    # Semantics: edge was (target →[relation]→ node) in original graph
                    step = {
                        "from": node,
                        "relation": f"←{relation}",  # mark as reverse traversal
                        "to": target,
                        "source": source_file.split("/")[-1].replace(".md", ""),
                    }
                else:
                    step = {
                        "from": node,
                        "relation": relation,
                        "to": target,
                        "source": source_file.split("/")[-1].replace(".md", ""),
                    }
                new_chain = chain + [step]

                if target == dst:
                    return new_chain

                if target not in visited:
                    visited.add(target)
                    queue.append((target, new_chain))

        return None

    # Primary: forward traversal
    chain = _bfs(forward, start_match, end_match, reversed_edges=False)
    if chain is not None:
        return chain

    # Fallback: reverse traversal (follow edges backward)
    if allow_reverse:
        chain = _bfs(reverse, start_match, end_match, reversed_edges=True)

    return chain


def find_implementation(entity: str, forward: dict) -> list[dict]:
    """Find if an entity has implementation edges (connects to code/projects)."""
    implementations = []
    for target, relation, source_file in forward.get(entity, []):
        if relation in ("implements", "applies", "applies_to"):
            implementations.append({
                "entity": target,
                "relation": relation,
                "source": source_file.split("/")[-1].replace(".md", ""),
            })
    return implementations


def main():
    parser = argparse.ArgumentParser(
        description="Trace causal/mechanistic chains through the knowledge graph"
    )
    parser.add_argument("start", help="Starting concept")
    parser.add_argument("end", help="Ending concept or project feature")
    parser.add_argument("--max-depth", type=int, default=6, help="Max chain length")
    parser.add_argument("--include-code", action="store_true",
                        help="Also search for code implementations at each step")
    parser.add_argument("--explain", action="store_true",
                        help="Show source notes for each step")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--allow-reverse", action="store_true",
                        help="Also follow edges in reverse direction when forward BFS "
                             "finds no path. Fixes cases where the graph edge points "
                             "target→source instead of source→target.")

    args = parser.parse_args()

    # Find DB
    dbs = sorted(
        glob.glob(str(Path.home() / ".local/share/vault-search/*.db")),
        key=lambda p: Path(p).stat().st_size,
        reverse=True,
    )
    if not dbs:
        print("No vault-search database found", file=sys.stderr)
        sys.exit(1)

    forward, reverse = load_graph(dbs[0])

    chain = trace_causal_chain(forward, reverse, args.start, args.end,
                               max_depth=args.max_depth,
                               allow_reverse=args.allow_reverse)

    if args.json:
        print(json.dumps({"start": args.start, "end": args.end, "chain": chain}, indent=2))
        return

    if not chain:
        print(f"No causal chain found: {args.start} → {args.end}")
        print(f"(Try --max-depth {args.max_depth + 2} or check entity names)")
        return

    print(f"CAUSAL TRACE: {args.start} → {args.end} ({len(chain)} steps)")
    print("=" * 60)
    print()

    for i, step in enumerate(chain, 1):
        rel = step["relation"]
        is_reverse = rel.startswith("←")
        bare_rel = rel.lstrip("←")
        if is_reverse:
            arrow = "←"
        elif bare_rel in CAUSAL_RELATIONS:
            arrow = "→"
        else:
            arrow = "~"
        print(f"  {i}. {step['from']}")
        print(f"     {arrow} [{rel}] {arrow}")

        if args.explain:
            print(f"     (source: {step['source']})")

        if args.include_code:
            impls = find_implementation(step["to"], forward)
            if impls:
                for impl in impls[:2]:
                    print(f"     CODE: {impl['entity']} ({impl['source']})")

    # Print final entity
    print(f"  {len(chain)+1}. {chain[-1]['to']}")
    print()

    # Summary
    causal_count = sum(1 for s in chain if s["relation"].lstrip("←") in CAUSAL_RELATIONS)
    structural_count = len(chain) - causal_count
    print(f"Chain: {causal_count} causal + {structural_count} structural steps")


if __name__ == "__main__":
    main()
