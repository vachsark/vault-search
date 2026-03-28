#!/usr/bin/env python3
"""
synthesis-suggest.py — Auto-suggest synthesis candidates from the knowledge graph.

Finds concept pairs that have high neighbor overlap across different disciplines
but no existing synthesis note bridging them. These are the highest-potential
cross-domain connections the research pipeline should investigate.

Usage:
    python3 synthesis-suggest.py                    # Top 10 candidates
    python3 synthesis-suggest.py --top 20           # More results
    python3 synthesis-suggest.py --min-jaccard 0.25 # Stricter overlap threshold
    python3 synthesis-suggest.py --json             # Machine-readable output
    python3 synthesis-suggest.py --ucb              # UCB exploration bonus (SYN-008)
"""
import argparse
import glob
import json
import math
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

VAULT_DIR = os.environ.get("VAULT_DIR", "/home/veech/Documents/TestVault")
KNOWLEDGE_DIR = os.path.join(VAULT_DIR, "Knowledge")


def suggest_synthesis(top: int = 10, min_jaccard: float = 0.15,
                      min_shared: int = 3, ucb: bool = False,
                      ucb_c: float = 1.0) -> list[dict]:
    """Find high-potential cross-discipline synthesis candidates.

    When ucb=True (SYN-2026-03-22-008), applies a UCB1 exploration bonus that
    up-weights entities with few existing synthesis notes. This shifts synthesis
    effort toward the frontier — under-explored entities with high cross-domain
    Jaccard potential but few existing synthesis notes.

    UCB formula per entity: bonus(e) = c * sqrt(ln(T) / max(N(e), 1))
    where T = total synthesis notes, N(e) = synthesis notes mentioning entity e.
    Pair score = jaccard * (1 + mean(bonus(e1), bonus(e2)))
    """

    # Find DB
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

    # Get existing synthesis notes
    synthesis_entities = set()
    # 2026-03-22 SYN-W9-META-001: Collect synthesis note title word sets for title dedup.
    # Each title is derived from the filename (e.g., "synthesis--foo-bar-baz.md" -> {"foo","bar","baz"}).
    # Used below to skip suggestions that duplicate an existing synthesis note.
    existing_synthesis_title_words: list[set[str]] = []
    # SYN-2026-03-22-008: Per-entity synthesis note counts for UCB exploration bonus.
    # N(entity) = number of synthesis notes mentioning that entity.
    entity_synthesis_count: dict[str, int] = defaultdict(int)
    total_synthesis_notes = 0
    for f in os.listdir(KNOWLEDGE_DIR):
        if f.startswith("synthesis--") and f.endswith(".md"):
            total_synthesis_notes += 1
            # Extract title words from filename for dedup
            slug = f.replace("synthesis--", "").replace(".md", "")
            title_words = set(w for w in slug.split("-") if len(w) > 2)
            if title_words:
                existing_synthesis_title_words.append(title_words)
            # Read the note and extract mentioned entities
            try:
                content = open(os.path.join(KNOWLEDGE_DIR, f), errors="replace").read().lower()
                for entity in graph:
                    if len(entity) > 4 and entity in content:
                        synthesis_entities.add(entity)
                        entity_synthesis_count[entity] += 1
            except Exception:
                pass

    # Find high-degree entities
    high_degree = [e for e, n in graph.items() if len(n) >= 6]

    candidates = []
    seen: set[tuple[str, str]] = set()

    for i, e1 in enumerate(high_degree):
        for e2 in high_degree[i + 1:]:
            if e1 == e2:
                continue

            # Skip if both are already in synthesis notes
            if e1 in synthesis_entities and e2 in synthesis_entities:
                continue

            # Check disciplines
            notes1 = entity_notes.get(e1, set())
            notes2 = entity_notes.get(e2, set())
            discs1 = set(n.split("--")[0] for n in notes1 if "--" in n)
            discs2 = set(n.split("--")[0] for n in notes2 if "--" in n)

            if not discs1 or not discs2 or discs1 == discs2:
                continue

            # Jaccard similarity
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

    # SYN-2026-03-22-008: UCB exploration bonus for under-explored entities.
    # Shifts synthesis effort toward the frontier — entities that have high Jaccard
    # potential but few existing synthesis notes covering them.
    # UCB1 formula: bonus(e) = c * sqrt(ln(T) / max(N(e), 1))
    # Pair score: jaccard * (1 + mean(bonus(e1), bonus(e2)))
    # When ucb=False, this block is skipped entirely (no behavioral change).
    if ucb and total_synthesis_notes > 0:
        ln_T = math.log(max(total_synthesis_notes, 1))
        for c in candidates:
            n_a = max(entity_synthesis_count.get(c["entity_a"], 0), 1)
            n_b = max(entity_synthesis_count.get(c["entity_b"], 0), 1)
            bonus_a = ucb_c * math.sqrt(ln_T / n_a)
            bonus_b = ucb_c * math.sqrt(ln_T / n_b)
            ucb_bonus = (bonus_a + bonus_b) / 2.0
            c["ucb_bonus"] = round(ucb_bonus, 3)
            # Multiplicative blending: jaccard * (1 + bonus) preserves the Jaccard
            # base signal while boosting under-explored pairs proportionally.
            c["jaccard_raw"] = c["jaccard"]
            c["jaccard"] = round(c["jaccard"] * (1.0 + ucb_bonus), 3)

    candidates.sort(key=lambda c: c["jaccard"], reverse=True)

    # Apply domain penalty: if both entities share the same domain prefix, reduce
    # jaccard by 50% to push intra-domain pairs down the ranking before dedup.
    for c in candidates:
        discs_a = set(c["disciplines_a"])
        discs_b = set(c["disciplines_b"])
        if discs_a & discs_b:  # non-empty intersection = shared domain prefix
            c["jaccard"] = round(c["jaccard"] * 0.5, 3)
            c["domain_penalized"] = True

    # Re-sort after domain penalty
    candidates.sort(key=lambda c: c["jaccard"], reverse=True)

    # Exact-variant filter (Max Run 5, 2026-03-22): Remove pairs where both entities
    # are spelling variants of each other (apostrophe, pluralization, adverb/adjective
    # swap). Normalized string comparison catches cases the 40% word-overlap threshold
    # misses because the variant words have >2 chars and appear in both names.
    def _normalize_token(tok: str) -> str:
        """Normalize one word token to canonical form."""
        tok = tok.lower().replace("'", "").replace("\u2019", "")  # apostrophes
        # adverb → adjective: evolutionarily → evolutionary
        if tok.endswith("ily") and len(tok) > 5:
            tok = tok[:-3] + "y"  # happily→happy, evolutionarily→evolutionary
        elif tok.endswith("ally") and len(tok) > 6:
            tok = tok[:-4] + "al"  # basically→basal (approx — good enough for dedup)
        elif tok.endswith("ly") and len(tok) > 4:
            tok = tok[:-2]  # quickly→quick
        # plurals
        if tok.endswith("ies") and len(tok) > 4:
            tok = tok[:-3] + "y"
        elif tok.endswith("s") and len(tok) > 4 and not tok.endswith("ss"):
            tok = tok[:-1]
        return tok

    def _normalize_entity(e: str) -> str:
        """Normalize full entity string for exact-variant comparison."""
        return " ".join(_normalize_token(w) for w in e.split())

    def _is_spelling_variant(a: str, b: str) -> bool:
        """Return True if a and b normalize to the same canonical form."""
        return _normalize_entity(a) == _normalize_entity(b)

    # Filter out pairs that are purely spelling variants (distance-0 synthesis)
    candidates = [
        c for c in candidates
        if not _is_spelling_variant(c["entity_a"], c["entity_b"])
    ]

    # Near-duplicate dedup: skip candidates whose entity names share >40% of words
    # with any already-accepted candidate entity name (lowered from 60%).
    deduped: list[dict] = []
    accepted_entity_words: list[set[str]] = []

    def _words(entity: str) -> set[str]:
        return set(w for w in entity.split() if len(w) > 2)

    def _word_overlap(a: str, b: str) -> float:
        wa, wb = _words(a), _words(b)
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / min(len(wa), len(wb))

    for c in candidates:
        ea, eb = c["entity_a"], c["entity_b"]
        is_dup = False
        for accepted_words in accepted_entity_words:
            if _word_overlap(ea, " ".join(accepted_words)) > 0.4 or \
               _word_overlap(eb, " ".join(accepted_words)) > 0.4:
                is_dup = True
                break
        if not is_dup:
            deduped.append(c)
            accepted_entity_words.append(_words(ea) | _words(eb))

    # SYN-W9-META-001 (2026-03-22): Per-batch domain diversity cap.
    # Prevents over-representation of high-degree domains (cs, neuro) in the
    # final output. If a domain already has >= MAX_DOMAIN_SLOTS suggestions in
    # the accepted batch, score further candidates from that domain by 0.7x.
    # This is applied as a greedy rerank over the already-deduped list.
    MAX_DOMAIN_SLOTS = 3
    domain_slot_counts: dict[str, int] = defaultdict(int)
    diversity_ranked: list[dict] = []

    for c in deduped:
        all_discs = set(c["disciplines_a"]) | set(c["disciplines_b"])
        overrepresented = any(domain_slot_counts[d] >= MAX_DOMAIN_SLOTS for d in all_discs)
        if overrepresented:
            c = dict(c)  # copy to avoid mutating original
            c["jaccard"] = round(c["jaccard"] * 0.7, 3)
            c["batch_diversity_penalized"] = True
        diversity_ranked.append(c)
        for d in all_discs:
            domain_slot_counts[d] += 1

    # Final sort by (possibly adjusted) jaccard after diversity penalty
    diversity_ranked.sort(key=lambda c: c["jaccard"], reverse=True)

    # 2026-03-22 SYN-W9-META-001: Title dedup filter.
    # Skip suggestions whose entity pair words overlap >80% with an existing
    # synthesis note title. This prevents re-suggesting topics already covered.
    # "Title" here = the set of words from both entity names combined, compared
    # against the word set extracted from each synthesis-- filename slug.
    def _title_overlap(entity_a: str, entity_b: str) -> bool:
        """Return True if entity pair duplicates an existing synthesis note title."""
        pair_words = set(w.lower() for w in (entity_a + " " + entity_b).split() if len(w) > 2)
        if not pair_words:
            return False
        for title_words in existing_synthesis_title_words:
            if not title_words:
                continue
            overlap = pair_words & title_words
            # Check overlap relative to the smaller set (the pair or the title)
            denom = min(len(pair_words), len(title_words))
            if denom > 0 and len(overlap) / denom > 0.8:
                return True
        return False

    title_deduped = [
        c for c in diversity_ranked
        if not _title_overlap(c["entity_a"], c["entity_b"])
    ]

    return title_deduped[:top]


def main():
    parser = argparse.ArgumentParser(
        description="Auto-suggest synthesis candidates from the knowledge graph"
    )
    parser.add_argument("--top", type=int, default=10, help="Number of candidates (default 10)")
    parser.add_argument("--min-jaccard", type=float, default=0.15,
                        help="Minimum Jaccard similarity (default 0.15)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--ucb", action="store_true",
                        help="Apply UCB exploration bonus for under-explored entities (SYN-008)")
    parser.add_argument("--ucb-c", type=float, default=1.0,
                        help="UCB exploration constant (default 1.0; higher = more exploration)")
    args = parser.parse_args()

    candidates = suggest_synthesis(top=args.top, min_jaccard=args.min_jaccard,
                                   ucb=args.ucb, ucb_c=args.ucb_c)

    if args.json:
        print(json.dumps(candidates, indent=2))
        return

    if not candidates:
        print("No synthesis candidates found above threshold.")
        return

    mode = "UCB-weighted" if args.ucb else "standard"
    print(f"SYNTHESIS CANDIDATES ({len(candidates)} found, {mode})")
    print("=" * 60)
    print()
    for i, c in enumerate(candidates, 1):
        d_a = "/".join(c["disciplines_a"])
        d_b = "/".join(c["disciplines_b"])
        print(f"  {i}. {c['entity_a']} ({d_a}) ↔ {c['entity_b']} ({d_b})")
        if "ucb_bonus" in c:
            print(f"     Score: {c['jaccard']} (raw Jaccard: {c.get('jaccard_raw', '?')}, UCB bonus: +{c['ucb_bonus']})")
        else:
            print(f"     Jaccard: {c['jaccard']}, shared: {c['shared_count']} entities")
        print(f"     Common: {', '.join(c['shared_entities'])}")
        print()


if __name__ == "__main__":
    main()
