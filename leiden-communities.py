#!/usr/bin/env python3
"""
leiden-communities.py — Leiden community detection for the vault knowledge graph.

Loads the knowledge graph from SQLite, builds an igraph Graph, runs the Leiden
algorithm, and stores community assignments back in the DB. Designed to integrate
with vault-search.py for community-aware retrieval.

Usage:
    python3 leiden-communities.py                      # Run with defaults (resolution=1.0)
    python3 leiden-communities.py --resolution 0.8     # Coarser communities
    python3 leiden-communities.py --resolution 1.5     # Finer communities
    python3 leiden-communities.py --sweep              # Sweep resolutions 0.5-2.0, pick best
    python3 leiden-communities.py --export             # Export communities as JSON
    python3 leiden-communities.py --query "attention"   # Show community for an entity
    python3 leiden-communities.py --top-communities 10  # Show top N communities by size

Requires: python-igraph, leidenalg (installed in _scripts/.venv)

RQ-020: Implements community detection to reduce average path length and enable
community-aware search retrieval.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import Counter

# Locate the venv and add it to path if needed
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_SITE = os.path.join(SCRIPT_DIR, ".venv", "lib")
if os.path.isdir(VENV_SITE):
    # Find the python version directory
    for d in os.listdir(VENV_SITE):
        sp = os.path.join(VENV_SITE, d, "site-packages")
        if os.path.isdir(sp) and sp not in sys.path:
            sys.path.insert(0, sp)

try:
    import igraph as ig
    import leidenalg
except ImportError:
    print("ERROR: python-igraph and leidenalg required.", file=sys.stderr)
    print("Install: pip install python-igraph leidenalg", file=sys.stderr)
    print(f"Or use the venv: {SCRIPT_DIR}/.venv/bin/python3 {__file__}", file=sys.stderr)
    sys.exit(1)


def db_path_for_root(root: str) -> str:
    """Deterministic DB path per root directory. Matches vault-graph.py / vault-search.py."""
    custom = os.environ.get("VAULT_SEARCH_DB")
    if custom:
        return custom
    root_hash = hashlib.sha256(os.path.realpath(root).encode()).hexdigest()[:12]
    return os.path.expanduser(f"~/.local/share/vault-search/{root_hash}.db")


def detect_vault_root() -> str:
    """Walk up from script dir to find the vault root (has _scripts/ dir)."""
    d = SCRIPT_DIR
    while d != "/":
        if os.path.isdir(os.path.join(d, "_scripts")):
            return d
        d = os.path.dirname(d)
    # Fallback: parent of _scripts
    return os.path.dirname(SCRIPT_DIR)


def ensure_community_table(conn: sqlite3.Connection):
    """Create the community assignments table if it doesn't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS communities (
            entity TEXT PRIMARY KEY,
            community_id INTEGER NOT NULL,
            community_size INTEGER NOT NULL,
            resolution REAL NOT NULL,
            modularity REAL NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_communities_id ON communities(community_id);
        CREATE INDEX IF NOT EXISTS idx_communities_size ON communities(community_size);

        CREATE TABLE IF NOT EXISTS community_meta (
            community_id INTEGER PRIMARY KEY,
            size INTEGER NOT NULL,
            top_entities TEXT,  -- JSON array of top-degree entities
            relation_types TEXT,  -- JSON object of relation type counts
            density REAL,
            resolution REAL NOT NULL,
            modularity REAL NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)


def load_graph(conn: sqlite3.Connection) -> tuple:
    """Load the knowledge graph from SQLite into an igraph Graph.

    Returns: (graph, node_names) where node_names maps vertex index -> entity name.
    """
    t0 = time.time()

    # Get all unique entities from relations
    rows = conn.execute("""
        SELECT source_entity, relation, target_entity, confidence
        FROM relations
    """).fetchall()

    # Build node index
    node_set = set()
    for src, _, tgt, _ in rows:
        node_set.add(src)
        node_set.add(tgt)

    node_names = sorted(node_set)  # deterministic ordering
    node_idx = {name: i for i, name in enumerate(node_names)}

    # Build edge list with weights
    edges = []
    weights = []
    edge_types = []
    for src, rel, tgt, conf in rows:
        edges.append((node_idx[src], node_idx[tgt]))
        weights.append(conf if conf else 0.8)
        edge_types.append(rel)

    # Create undirected graph (Leiden works on undirected)
    g = ig.Graph(n=len(node_names), edges=edges, directed=False)
    g.vs["name"] = node_names
    g.es["weight"] = weights
    g.es["type"] = edge_types

    # Simplify: merge parallel edges (sum weights), remove self-loops
    g = g.simplify(combine_edges={"weight": "sum", "type": "first"})

    elapsed = time.time() - t0
    print(f"Graph loaded: {g.vcount()} nodes, {g.ecount()} edges ({elapsed:.2f}s)")

    return g, node_names


def run_leiden(g: ig.Graph, resolution: float = 1.0) -> tuple:
    """Run Leiden community detection.

    Returns: (partition, modularity)
    """
    t0 = time.time()

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=-1,  # iterate until stable
        seed=42,  # reproducible
    )

    modularity = g.modularity(partition.membership, weights="weight")
    elapsed = time.time() - t0

    n_communities = len(set(partition.membership))
    sizes = Counter(partition.membership)
    largest = sizes.most_common(1)[0][1]
    singletons = sum(1 for s in sizes.values() if s == 1)

    print(f"Leiden complete ({elapsed:.2f}s):")
    print(f"  Resolution:    {resolution}")
    print(f"  Communities:   {n_communities}")
    print(f"  Modularity:    {modularity:.4f}")
    print(f"  Largest:       {largest} nodes")
    print(f"  Singletons:    {singletons}")
    print(f"  Median size:   {sorted(sizes.values())[len(sizes)//2]}")

    return partition, modularity


def resolution_sweep(g: ig.Graph, values=None) -> tuple:
    """Sweep resolution parameters to find the best modularity.

    Returns: (best_resolution, best_partition, best_modularity)
    """
    if values is None:
        values = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]

    print(f"\nResolution sweep ({len(values)} values):")
    print(f"{'Resolution':>12} {'Communities':>12} {'Modularity':>12} {'Largest':>10} {'Singletons':>12}")
    print("-" * 62)

    best_mod = -1
    best_res = 1.0
    best_part = None

    for res in values:
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=res,
            n_iterations=-1,
            seed=42,
        )
        mod = g.modularity(partition.membership, weights="weight")
        sizes = Counter(partition.membership)
        n_comm = len(sizes)
        largest = sizes.most_common(1)[0][1]
        singletons = sum(1 for s in sizes.values() if s == 1)

        marker = " <-- best" if mod > best_mod else ""
        print(f"{res:>12.2f} {n_comm:>12} {mod:>12.4f} {largest:>10} {singletons:>12}{marker}")

        if mod > best_mod:
            best_mod = mod
            best_res = res
            best_part = partition

    print(f"\nBest resolution: {best_res} (modularity={best_mod:.4f})")
    return best_res, best_part, best_mod


def store_communities(conn: sqlite3.Connection, g: ig.Graph,
                      partition, modularity: float, resolution: float):
    """Store community assignments and metadata in the DB."""
    t0 = time.time()
    ensure_community_table(conn)

    membership = partition.membership
    sizes = Counter(membership)

    # Clear old data
    conn.execute("DELETE FROM communities")
    conn.execute("DELETE FROM community_meta")

    # Insert node assignments
    rows = []
    for i, name in enumerate(g.vs["name"]):
        cid = membership[i]
        rows.append((name, cid, sizes[cid], resolution, modularity))

    conn.executemany(
        "INSERT INTO communities (entity, community_id, community_size, resolution, modularity) "
        "VALUES (?, ?, ?, ?, ?)",
        rows
    )

    # Compute and store community metadata
    # Build per-community entity lists and degree info
    community_nodes = {}
    for i, cid in enumerate(membership):
        community_nodes.setdefault(cid, []).append(i)

    meta_rows = []
    for cid, node_indices in community_nodes.items():
        subg = g.subgraph(node_indices)

        # Top entities by degree in the subgraph
        degrees = subg.degree()
        top_idx = sorted(range(len(degrees)), key=lambda x: degrees[x], reverse=True)[:10]
        top_entities = [subg.vs[i]["name"] for i in top_idx]

        # Relation type distribution within community
        rel_types = Counter()
        for e in subg.es:
            if "type" in e.attributes():
                rel_types[e["type"]] += 1

        # Density
        density = subg.density() if subg.vcount() > 1 else 0.0

        meta_rows.append((
            cid,
            len(node_indices),
            json.dumps(top_entities),
            json.dumps(dict(rel_types.most_common(10))),
            density,
            resolution,
            modularity,
        ))

    conn.executemany(
        "INSERT INTO community_meta "
        "(community_id, size, top_entities, relation_types, density, resolution, modularity) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        meta_rows
    )

    conn.commit()
    elapsed = time.time() - t0
    print(f"\nStored {len(rows)} node assignments and {len(meta_rows)} community profiles ({elapsed:.2f}s)")


def print_top_communities(conn: sqlite3.Connection, n: int = 15):
    """Print the top N communities by size with their key entities."""
    rows = conn.execute(
        "SELECT community_id, size, top_entities, relation_types, density "
        "FROM community_meta ORDER BY size DESC LIMIT ?",
        (n,)
    ).fetchall()

    if not rows:
        print("No community data found. Run detection first.")
        return

    print(f"\nTop {n} communities:")
    print(f"{'ID':>5} {'Size':>6} {'Density':>8}  Top Entities")
    print("-" * 80)

    for cid, size, top_ent, rel_types, density in rows:
        entities = json.loads(top_ent)[:5]
        ent_str = ", ".join(entities)
        print(f"{cid:>5} {size:>6} {density:>8.4f}  {ent_str}")


def query_entity(conn: sqlite3.Connection, query: str):
    """Show community info for an entity."""
    # Prefer exact match, fall back to fuzzy
    row = conn.execute(
        "SELECT entity, community_id, community_size FROM communities "
        "WHERE entity = ? LIMIT 1",
        (query.lower(),)
    ).fetchone()
    if not row:
        row = conn.execute(
            "SELECT entity, community_id, community_size FROM communities "
            "WHERE entity LIKE ? ORDER BY length(entity) ASC LIMIT 1",
            (f"%{query}%",)
        ).fetchone()

    if not row:
        print(f"No entity matching '{query}' found in communities.")
        return

    entity, cid, csize = row
    print(f"\nEntity:    {entity}")
    print(f"Community: {cid} (size={csize})")

    # Get community metadata
    meta = conn.execute(
        "SELECT top_entities, relation_types, density FROM community_meta WHERE community_id=?",
        (cid,)
    ).fetchone()

    if meta:
        top_ent = json.loads(meta[0])
        rel_types = json.loads(meta[1])
        density = meta[2]
        print(f"Density:   {density:.4f}")
        print(f"Top entities: {', '.join(top_ent[:8])}")
        print(f"Relation types: {rel_types}")

    # Show other entities in the same community
    peers = conn.execute(
        "SELECT entity FROM communities WHERE community_id=? AND entity != ? "
        "ORDER BY entity LIMIT 20",
        (cid, entity)
    ).fetchall()
    if peers:
        print(f"\nPeers in community ({min(20, len(peers))} shown):")
        for (p,) in peers:
            print(f"  - {p}")


def export_communities(conn: sqlite3.Connection, outpath: str):
    """Export community data as JSON."""
    communities = {}
    rows = conn.execute(
        "SELECT entity, community_id FROM communities ORDER BY community_id, entity"
    ).fetchall()

    for entity, cid in rows:
        communities.setdefault(cid, []).append(entity)

    meta = {}
    mrows = conn.execute(
        "SELECT community_id, size, top_entities, relation_types, density, resolution, modularity "
        "FROM community_meta ORDER BY size DESC"
    ).fetchall()

    for cid, size, top_ent, rel_types, density, resolution, modularity in mrows:
        meta[cid] = {
            "size": size,
            "top_entities": json.loads(top_ent),
            "relation_types": json.loads(rel_types),
            "density": density,
        }

    output = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resolution": mrows[0][5] if mrows else None,
        "modularity": mrows[0][6] if mrows else None,
        "num_communities": len(communities),
        "num_nodes": sum(len(v) for v in communities.values()),
        "communities": {str(k): {"entities": v, "meta": meta.get(k, {})}
                       for k, v in sorted(communities.items(),
                                          key=lambda x: len(x[1]), reverse=True)},
    }

    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Exported to {outpath} ({os.path.getsize(outpath) / 1024:.1f} KB)")


def print_graph_stats(g: ig.Graph, partition):
    """Print detailed graph statistics relevant to vault health."""
    membership = partition.membership
    sizes = Counter(membership)

    # Connected components
    components = g.connected_components()
    giant = max(components, key=len)

    print(f"\n{'='*60}")
    print(f"GRAPH STATISTICS")
    print(f"{'='*60}")
    print(f"Nodes:                {g.vcount():>8}")
    print(f"Edges:                {g.ecount():>8}")
    print(f"Density:              {g.density():>8.6f}")
    print(f"Connected components: {len(components):>8}")
    print(f"Giant component:      {len(giant):>8} ({100*len(giant)/g.vcount():.1f}%)")

    # Average path length on giant component
    if len(giant) > 1 and len(giant) <= 5000:
        subg = g.subgraph(giant)
        avg_path = subg.average_path_length()
        diameter = subg.diameter()
        print(f"Avg path length (GC): {avg_path:>8.2f}")
        print(f"Diameter (GC):        {diameter:>8}")
    elif len(giant) > 5000:
        # Sample-based estimate for large graphs
        import random
        random.seed(42)
        subg = g.subgraph(giant)
        sample_size = min(500, len(giant))
        sample = random.sample(range(subg.vcount()), sample_size)
        total = 0
        count = 0
        for v in sample:
            dists = subg.distances(source=v)[0]
            for d in dists:
                if d > 0 and d < float('inf'):
                    total += d
                    count += 1
        if count > 0:
            est_avg = total / count
            print(f"Avg path length (est):{est_avg:>8.2f} (sampled {sample_size} nodes)")

    print(f"\n{'='*60}")
    print(f"COMMUNITY STATISTICS")
    print(f"{'='*60}")
    print(f"Communities:          {len(sizes):>8}")
    print(f"Modularity:           {g.modularity(membership, weights='weight'):>8.4f}")

    size_vals = sorted(sizes.values(), reverse=True)
    print(f"Largest community:    {size_vals[0]:>8}")
    print(f"Smallest community:   {size_vals[-1]:>8}")
    print(f"Median size:          {size_vals[len(size_vals)//2]:>8}")
    print(f"Mean size:            {sum(size_vals)/len(size_vals):>8.1f}")
    print(f"Singletons:           {sum(1 for s in size_vals if s == 1):>8}")
    print(f"Size > 10:            {sum(1 for s in size_vals if s > 10):>8}")
    print(f"Size > 50:            {sum(1 for s in size_vals if s > 50):>8}")
    print(f"Size > 100:           {sum(1 for s in size_vals if s > 100):>8}")

    # Size distribution histogram
    buckets = [(1, 1), (2, 5), (6, 10), (11, 25), (26, 50), (51, 100),
               (101, 250), (251, 500), (501, float('inf'))]
    print(f"\nCommunity size distribution:")
    for lo, hi in buckets:
        c = sum(1 for s in size_vals if lo <= s <= hi)
        if c > 0:
            label = f"{lo}-{hi}" if hi != float('inf') else f"{lo}+"
            bar = "#" * min(c, 50)
            print(f"  {label:>8}: {c:>5}  {bar}")

    # -------------------------------------------------------------------
    # Node degree distribution — shows leaf node inflation
    # -------------------------------------------------------------------
    degrees = g.degree()
    n_nodes = g.vcount()
    print(f"\n{'='*60}")
    print(f"NODE DEGREE DISTRIBUTION")
    print(f"{'='*60}")

    deg_counter = Counter(degrees)
    avg_degree = sum(degrees) / n_nodes if n_nodes else 0
    median_idx = n_nodes // 2
    sorted_degrees = sorted(degrees)
    median_degree = sorted_degrees[median_idx] if n_nodes else 0
    max_degree = max(degrees) if degrees else 0

    print(f"Mean degree:          {avg_degree:>8.2f}")
    print(f"Median degree:        {median_degree:>8}")
    print(f"Max degree:           {max_degree:>8}")

    # Degree-1 (leaf) stats — the key metric for the audit
    deg1_count = deg_counter.get(1, 0)
    deg1_pct = 100 * deg1_count / n_nodes if n_nodes else 0
    print(f"Degree-1 (leaf) nodes:{deg1_count:>8} ({deg1_pct:.1f}%)")
    deg_ge2 = n_nodes - deg1_count
    deg_ge2_pct = 100 * deg_ge2 / n_nodes if n_nodes else 0
    print(f"Degree >= 2 (core):   {deg_ge2:>8} ({deg_ge2_pct:.1f}%)")
    deg_ge3 = sum(1 for d in degrees if d >= 3)
    deg_ge3_pct = 100 * deg_ge3 / n_nodes if n_nodes else 0
    print(f"Degree >= 3:          {deg_ge3:>8} ({deg_ge3_pct:.1f}%)")

    deg_buckets = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 25), (26, 50),
                   (51, 100), (101, float('inf'))]
    print(f"\nDegree distribution:")
    for lo, hi in deg_buckets:
        c = sum(1 for d in degrees if lo <= d <= hi)
        if c > 0:
            label = f"{lo}-{hi}" if hi != float('inf') else f"{lo}+"
            pct = 100 * c / n_nodes
            bar = "#" * min(int(pct), 50)
            print(f"  {label:>8}: {c:>6} ({pct:>5.1f}%)  {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="Leiden community detection for the vault knowledge graph."
    )
    parser.add_argument("--resolution", "-r", type=float, default=1.0,
                        help="Resolution parameter for Leiden (default: 1.0)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep resolution values and pick the best")
    parser.add_argument("--export", action="store_true",
                        help="Export communities as JSON")
    parser.add_argument("--export-path", type=str, default=None,
                        help="Path for JSON export (default: _collab/leiden-communities.json)")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Query community for an entity")
    parser.add_argument("--top-communities", "-t", type=int, default=None,
                        help="Show top N communities by size")
    parser.add_argument("--stats-only", action="store_true",
                        help="Print stats without re-running detection")
    parser.add_argument("--no-store", action="store_true",
                        help="Don't store results in DB (dry run)")
    parser.add_argument("--vault-root", type=str, default=None,
                        help="Vault root path (auto-detected if not given)")

    args = parser.parse_args()

    # Resolve vault root and DB
    vault_root = args.vault_root or detect_vault_root()
    db = db_path_for_root(vault_root)

    if not os.path.exists(db):
        print(f"ERROR: Database not found at {db}", file=sys.stderr)
        print(f"Run vault-graph.py index first.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db)

    # Query mode: just look up an entity
    if args.query:
        query_entity(conn, args.query)
        conn.close()
        return

    # Top communities mode
    if args.top_communities and args.stats_only:
        print_top_communities(conn, args.top_communities)
        conn.close()
        return

    # Check if we have relations
    rel_count = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
    if rel_count == 0:
        print("ERROR: No relations in the graph. Run vault-graph.py index first.", file=sys.stderr)
        conn.close()
        sys.exit(1)

    print(f"Vault root: {vault_root}")
    print(f"Database:   {db}")
    print(f"Relations:  {rel_count}")
    print()

    # Load graph
    g, node_names = load_graph(conn)

    # Run detection
    if args.sweep:
        resolution, partition, modularity = resolution_sweep(g)
    else:
        resolution = args.resolution
        partition, modularity = run_leiden(g, resolution)

    # Print detailed stats
    print_graph_stats(g, partition)

    # Print top communities
    if not args.no_store:
        store_communities(conn, g, partition, modularity, resolution)
        print_top_communities(conn, args.top_communities or 15)

    if args.top_communities and args.no_store:
        # Can't show from DB if we didn't store, so show inline
        membership = partition.membership
        sizes = Counter(membership)
        community_nodes = {}
        for i, cid in enumerate(membership):
            community_nodes.setdefault(cid, []).append(i)

        top = sorted(community_nodes.items(), key=lambda x: len(x[1]), reverse=True)[:args.top_communities]
        print(f"\nTop {args.top_communities} communities (not stored):")
        for cid, indices in top:
            names = [g.vs[i]["name"] for i in indices]
            deg = g.degree(indices)
            top_by_deg = sorted(zip(names, deg), key=lambda x: x[1], reverse=True)[:5]
            ent_str = ", ".join(f"{n}" for n, d in top_by_deg)
            print(f"  [{cid}] size={len(indices)}: {ent_str}")

    # Export
    if args.export and not args.no_store:
        export_path = args.export_path or os.path.join(vault_root, "_collab", "leiden-communities.json")
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        export_communities(conn, export_path)

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
