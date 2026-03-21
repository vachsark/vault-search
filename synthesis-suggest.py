#!/usr/bin/env python3
"""
synthesis-suggest.py — Auto-suggest synthesis candidates from the knowledge graph.

Finds concept pairs that have high neighbor overlap across different disciplines
but no existing synthesis note bridging them. These are the highest-potential
cross-domain connections the research pipeline should investigate.

Usage:
    python3 synthesis-suggest.py ~/notes                    # Top 10 candidates
    python3 synthesis-suggest.py ~/notes --top 20           # More results
    python3 synthesis-suggest.py ~/notes --min-jaccard 0.25 # Stricter overlap threshold
    python3 synthesis-suggest.py ~/notes --json             # Machine-readable output

The tool uses the vault-search knowledge graph database. Run vault-graph.py index
first to extract entities and relationships from your notes.
"""
import argparse
import glob
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# File prefix separator for discipline detection. Files named "discipline--topic.md"
# will have their prefix used as the discipline label. Customize if your notes use
# a different naming convention.
DISCIPLINE_SEPARATOR = "--"


def suggest_synthesis(notes_dir: str, top: int = 10, min_jaccard: float = 0.15,
                      min_shared: int = 3, db_path: str = None) -> list[dict]:
    """Find high-potential cross-discipline synthesis candidates."""

    # Find DB
    if db_path:
        dbs = [db_path]
    else:
        dbs = sorted(
            glob.glob(str(Path.home() / ".local/share/vault-search/*.db")),
            key=lambda p: Path(p).stat().st_size,
            reverse=True,
        )
    if not dbs:
        return []

    conn = sqlite3.connect(dbs[0])

    # Build graph + entity-to-note mapping
    graph: dict[str, set[str]] = defaultdict(set)
    entity_notes: dict[str, set[str]] = defaultdict(set)

    for row in conn.execute("SELECT source_entity, target_entity, source_file FROM relations"):
        s, t = row[0].lower().strip(), row[1].lower().strip()
        f = row[2].split("/")[-1].replace(".md", "")
        if s and t and s != t:
            graph[s].add(t)
            graph[t].add(s)
            entity_notes[s].add(f)
            entity_notes[t].add(f)

    conn.close()

    # Find high-degree entities
    high_degree = [e for e, n in graph.items() if len(n) >= 6]

    candidates = []
    seen: set[tuple[str, str]] = set()

    for i, e1 in enumerate(high_degree):
        for e2 in high_degree[i + 1:]:
            if e1 == e2:
                continue

            # Check disciplines (files with "prefix--topic.md" naming)
            notes1 = entity_notes.get(e1, set())
            notes2 = entity_notes.get(e2, set())
            discs1 = set(n.split(DISCIPLINE_SEPARATOR)[0] for n in notes1 if DISCIPLINE_SEPARATOR in n)
            discs2 = set(n.split(DISCIPLINE_SEPARATOR)[0] for n in notes2 if DISCIPLINE_SEPARATOR in n)

            if not discs1 or not discs2 or discs1 == discs2:
                continue

            # Jaccard similarity on shared neighbors
            n1, n2 = graph[e1], graph[e2]
            intersection = n1 & n2
            union = n1 | n2
            if not union:
                continue

            jaccard = len(intersection) / len(union)

            if jaccard >= min_jaccard and len(intersection) >= min_shared:
                pair = tuple(sorted([e1, e2]))
                if pair not in seen:
                    seen.add(pair)
                    candidates.append({
                        "entity_a": e1,
                        "entity_b": e2,
                        "disciplines_a": sorted(discs1)[:3],
                        "disciplines_b": sorted(discs2)[:3],
                        "jaccard": round(jaccard, 3),
                        "shared_count": len(intersection),
                        "shared_entities": sorted(intersection)[:5],
                    })

    candidates.sort(key=lambda c: c["jaccard"], reverse=True)
    return candidates[:top]


def main():
    parser = argparse.ArgumentParser(
        description="Auto-suggest synthesis candidates from the knowledge graph"
    )
    parser.add_argument(
        "root", nargs="?", default=".",
        help="Root directory of your notes (default: current directory)"
    )
    parser.add_argument("--top", type=int, default=10, help="Number of candidates (default 10)")
    parser.add_argument("--min-jaccard", type=float, default=0.15,
                        help="Minimum Jaccard similarity (default 0.15)")
    parser.add_argument("--db", type=str, default=None,
                        help="Database path (default: auto per root dir)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    candidates = suggest_synthesis(
        notes_dir=args.root,
        top=args.top,
        min_jaccard=args.min_jaccard,
        db_path=args.db,
    )

    if args.json:
        print(json.dumps(candidates, indent=2))
        return

    if not candidates:
        print("No synthesis candidates found above threshold.")
        return

    print(f"SYNTHESIS CANDIDATES ({len(candidates)} found)")
    print("=" * 60)
    print()
    for i, c in enumerate(candidates, 1):
        d_a = "/".join(c["disciplines_a"])
        d_b = "/".join(c["disciplines_b"])
        print(f"  {i}. {c['entity_a']} ({d_a}) <-> {c['entity_b']} ({d_b})")
        print(f"     Jaccard: {c['jaccard']}, shared: {c['shared_count']} entities")
        print(f"     Common: {', '.join(c['shared_entities'])}")
        print()


if __name__ == "__main__":
    main()
