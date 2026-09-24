#!/usr/bin/env python3
"""
knowledge-path.py — Find the shortest conceptual path between two ideas.

Uses the vault's knowledge graph (entity relations) to find how two concepts
connect through intermediate ideas. Like "six degrees of separation" for knowledge.

Usage:
    python3 knowledge-path.py "prospect theory" "transformer architecture"
    python3 knowledge-path.py "dopamine" "market microstructure" --max-hops 8
    python3 knowledge-path.py "ADHD" "reinforcement learning"
"""
import argparse
import glob
import hashlib
import os
import sqlite3
import sys
from collections import defaultdict, deque
from pathlib import Path


def build_graph(db_path: str, prune_leaves: bool = True) -> dict[str, set[str]]:
    """Build an undirected adjacency graph from the relations table.

    By default, prunes degree-1 entities (mentions, not concepts).
    Per practice--first-principles-audit-knowledge-engine: 58% of entities
    are degree-1 leaf nodes that inflate average path length from the <6-hop
    target to 9.42 hops. Removing them focuses the graph on conceptual hubs.
    """
    conn = sqlite3.connect(db_path)
    graph: dict[str, set[str]] = defaultdict(set)

    for row in conn.execute("SELECT source_entity, target_entity FROM relations"):
        s, t = row[0].lower().strip(), row[1].lower().strip()
        if s and t and s != t:
            graph[s].add(t)
            graph[t].add(s)

    conn.close()

    if prune_leaves:
        # Remove degree-1 entities — they are mentions, not structural concepts.
        # A single-pass prune is sufficient; iterative k-core would be too aggressive.
        leaves = {n for n, neighbors in graph.items() if len(neighbors) < 2}
        for node in leaves:
            for neighbor in list(graph[node]):
                graph[neighbor].discard(node)
            del graph[node]

    return dict(graph)


def find_path(graph: dict[str, set[str]], start: str, end: str,
              max_depth: int = 8) -> list[str] | None:
    """BFS shortest path between two entities.

    Neighbors are explored in descending degree order so that, when multiple
    shortest paths exist, the one passing through the highest-degree (hub)
    nodes is returned. Hub nodes are semantic connectors in the knowledge
    graph — preferring them produces more meaningful paths.
    """
    start, end = start.lower().strip(), end.lower().strip()

    # Exact match first
    if start not in graph:
        # Fuzzy match — find entities containing the search term
        candidates = [e for e in graph if start in e]
        if candidates:
            start = min(candidates, key=len)  # shortest match
        else:
            return None

    if end not in graph:
        candidates = [e for e in graph if end in e]
        if candidates:
            end = min(candidates, key=len)
        else:
            return None

    if start == end:
        return [start]

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        node, path = queue.popleft()
        if len(path) > max_depth:
            continue

        # Sort by degree descending — hub-biased BFS prefers paths through
        # high-connectivity concepts (knowledge hubs) over peripheral nodes.
        neighbors = sorted(graph.get(node, set()),
                           key=lambda n: len(graph.get(n, ())), reverse=True)
        for neighbor in neighbors:
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None


def find_all_paths(graph: dict[str, set[str]], start: str, end: str,
                   max_depth: int = 6, max_paths: int = 3) -> list[list[str]]:
    """Find multiple shortest paths (not just the first one)."""
    start_lower, end_lower = start.lower().strip(), end.lower().strip()

    # Fuzzy match
    if start_lower not in graph:
        candidates = [e for e in graph if start_lower in e]
        start_lower = min(candidates, key=len) if candidates else start_lower

    if end_lower not in graph:
        candidates = [e for e in graph if end_lower in e]
        end_lower = min(candidates, key=len) if candidates else end_lower

    paths = []
    queue = deque([(start_lower, [start_lower])])
    visited_at_depth: dict[str, int] = {start_lower: 0}
    shortest_found = max_depth + 1

    while queue and len(paths) < max_paths:
        node, path = queue.popleft()

        if len(path) > shortest_found + 1:
            break

        if len(path) > max_depth:
            continue

        neighbors = sorted(graph.get(node, set()),
                           key=lambda n: len(graph.get(n, ())), reverse=True)
        for neighbor in neighbors:
            new_path = path + [neighbor]

            if neighbor == end_lower:
                paths.append(new_path)
                shortest_found = min(shortest_found, len(new_path))
                continue

            depth = len(new_path)
            if neighbor not in visited_at_depth or visited_at_depth[neighbor] >= depth:
                visited_at_depth[neighbor] = depth
                queue.append((neighbor, new_path))

    return paths


def find_db(root: str | None) -> str | None:
    """Index DB for --root (same hashing as vault-index.py). Without --root,
    honor VAULT_SEARCH_DB, else fall back to the largest index on disk."""
    custom = os.environ.get("VAULT_SEARCH_DB")
    if custom:
        return custom
    base = Path.home() / ".local/share/vault-search"
    if root:
        root_hash = hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:12]
        return str(base / f"{root_hash}.db")
    dbs = sorted(glob.glob(str(base / "*.db")),
                 key=lambda p: Path(p).stat().st_size, reverse=True)
    # Prefer the largest index that actually has a knowledge graph
    for db in dbs:
        try:
            conn = sqlite3.connect(db)
            has_graph = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='relations'"
            ).fetchone()
            conn.close()
        except sqlite3.Error:
            continue
        if has_graph:
            return db
    return dbs[0] if dbs else None


def _hint_missing(graph: dict[str, set[str]], args) -> None:
    """Explain a miss caused by an endpoint that isn't in the (pruned) graph."""
    for name in (args.start, args.end):
        term = name.lower().strip()
        if not any(term in e for e in graph):  # same fuzzy rule as find_path
            if args.include_leaves:
                print(f"  '{name}' is not an entity in the graph.")
            else:
                print(f"  '{name}' is not in the graph (it may have only one connection). "
                      f"Try --include-leaves.")


def main():
    parser = argparse.ArgumentParser(
        description="Find conceptual paths between two ideas in the knowledge graph"
    )
    parser.add_argument("start", help="Starting concept")
    parser.add_argument("end", help="Ending concept")
    parser.add_argument("--max-hops", type=int, default=8,
                        help="Maximum path length (default: 8)")
    parser.add_argument("--all", action="store_true",
                        help="Show multiple paths")
    parser.add_argument("--stats", action="store_true",
                        help="Show graph statistics")
    parser.add_argument("--include-leaves", action="store_true",
                        help="Include degree-1 entities (leaf nodes / mentions). "
                             "Default: pruned to degree>=2 for shorter, more meaningful paths")
    parser.add_argument("--root", type=str, default=None,
                        help="Vault root that was indexed (default: largest index on disk)")

    args = parser.parse_args()

    # Find the database
    db = find_db(args.root)
    if not db or not Path(db).exists():
        where = f" for {args.root}" if args.root else ""
        print(f"No vault-search database found{where}. "
              f"Run: vault-index.py <vault> && vault-graph.py index <vault>", file=sys.stderr)
        sys.exit(1)

    graph = build_graph(db, prune_leaves=not args.include_leaves)

    if args.stats:
        n_edges = sum(len(v) for v in graph.values()) // 2
        leaf_note = "" if args.include_leaves else " (degree>=2 concepts only)"
        print(f"Graph: {len(graph)} entities{leaf_note}, {n_edges} edges")
        print()

    if args.all:
        paths = find_all_paths(graph, args.start, args.end,
                               max_depth=args.max_hops, max_paths=3)
        if paths:
            print(f"{args.start} → {args.end}")
            print()
            for i, path in enumerate(paths, 1):
                print(f"  Path {i} ({len(path)-1} hops):")
                print(f"    {' → '.join(path)}")
                print()
        else:
            print(f"No path found between '{args.start}' and '{args.end}' "
                  f"within {args.max_hops} hops")
            _hint_missing(graph, args)
    else:
        path = find_path(graph, args.start, args.end, max_depth=args.max_hops)
        if path:
            hops = len(path) - 1
            print(f"{args.start} → {args.end} ({hops} hops):")
            print(f"  {' → '.join(path)}")
        else:
            print(f"No path found between '{args.start}' and '{args.end}' "
                  f"within {args.max_hops} hops")
            _hint_missing(graph, args)


if __name__ == "__main__":
    main()
