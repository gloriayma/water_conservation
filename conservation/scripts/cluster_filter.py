"""
Parse RCSB 100%-identity cluster file and apply quality filters.

Cluster file format (clusters-by-entity-100.txt):
  One cluster per line: space-separated PDBID_entityID tokens (uppercase PDB IDs).
  Example: '1ABC_1 2DEF_1 3GHI_2 ...'

ASSUMPTIONS VERIFIED ON 2026-05-04:
  HOLDS:     Every token has exactly one '_' → PDBID_entityID format.
  HOLDS:     No exact duplicate tokens within a cluster line.
  DOES NOT HOLD: A PDB can appear twice in one cluster with different entity IDs
             (4,127 token occurrences across 3,581 unique PDBs). Verified breakdown:
               ~1,099 cases: same-sequence chains (RCSB assigned separate entity IDs
                             to chains Boltz considers identical — homodimer-like)
               ~2,124 cases: genuinely different sequences (a heterocomplex where both
                             protein chains independently fall into the same cluster)
               ~358 cases: not in npz dataset (can't verify)
             In both cases the right behavior is one entry per PDB (waters belong to
             the whole structure, not individual chains). We deduplicate by PDB ID and
             count discarded tokens as 'dups_within_cluster' in the stats.

Token accounting (2.0 Å run):
  total_tokens = dups_within_cluster + dropped_not_in_manifest
               + dropped_not_xray + dropped_bad_resolution
               + dropped_no_waters + total_qualifying_pdb_in_clusters
  1,597,265   = 4,127 + 1,142,259 + 200,377 + 141,153 + 208 + 109,141  ✓
"""

import json
from pathlib import Path

CONSERVATION_DIR = Path("/data/rbg/users/gloriama/dev/water_conservation/conservation")
CLUSTER_FILE = CONSERVATION_DIR / "clusters-by-entity-100.txt"
MANIFEST_FILE = Path("/data/rbg/shared/datasets/processed_rcsb/rcsb_solvents/manifest.json")


def load_manifest_index(manifest_path: Path = MANIFEST_FILE) -> dict:
    """
    Return dict[pdb_id_lower] -> {resolution, method, has_waters}.
    has_waters is True if any chain has mol_type==4 and num_residues > 0.
    """
    entries = json.load(open(manifest_path))
    index = {}
    for e in entries:
        pdb_id = e["id"].lower()
        has_waters = any(c["mol_type"] == 4 and c["num_residues"] > 0 for c in e["chains"])
        index[pdb_id] = {
            "resolution": e["structure"].get("resolution"),
            "method": e["structure"].get("method", ""),
            "has_waters": has_waters,
        }
    return index


def filter_clusters(
    manifest_index: dict,
    resolution_cutoff: float = 2.0,
    min_cluster_size: int = 5,
    cluster_file: Path = CLUSTER_FILE,
) -> tuple[list[list[str]], dict]:
    """
    Filter RCSB clusters to X-ray, resolution <= cutoff, with waters, >= min_cluster_size.

    Returns:
      filtered_clusters: list of lists of qualifying PDB IDs (lowercase) per cluster
      stats: dict with full token-level accounting (all tokens sum to total_tokens)
    """
    total_tokens = 0
    dups_within_cluster = 0       # same pdb_id, different entity_id in one cluster
    dropped_not_in_manifest = 0
    dropped_not_xray = 0
    dropped_bad_resolution = 0
    dropped_no_waters = 0
    qualifying_pdb_counts = []    # len of qualifying list for every cluster with >=1
    filtered_clusters = []

    with open(cluster_file) as f:
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            total_tokens += len(tokens)

            # ASSUMPTION HOLDS: every token has exactly one '_'
            # ASSUMPTION DOES NOT HOLD: each pdb_id appears at most once per cluster
            seen_pdbs = set()
            qualifying = []

            for tok in tokens:
                assert "_" in tok, f"malformed token: {tok!r}"
                pdb_id = tok.rsplit("_", 1)[0].lower()

                if pdb_id in seen_pdbs:
                    dups_within_cluster += 1
                    continue
                seen_pdbs.add(pdb_id)

                meta = manifest_index.get(pdb_id)
                if meta is None:
                    dropped_not_in_manifest += 1
                    continue
                if meta["method"] != "x-ray diffraction":
                    dropped_not_xray += 1
                    continue
                if meta["resolution"] is None or meta["resolution"] > resolution_cutoff:
                    dropped_bad_resolution += 1
                    continue
                if not meta["has_waters"]:
                    dropped_no_waters += 1
                    continue

                qualifying.append(pdb_id)

            if qualifying:
                qualifying_pdb_counts.append(len(qualifying))
            if len(qualifying) >= min_cluster_size:
                filtered_clusters.append(qualifying)

    # Every token must be accounted for in exactly one bucket.
    total_qualifying = sum(qualifying_pdb_counts)
    assert (dups_within_cluster + dropped_not_in_manifest + dropped_not_xray
            + dropped_bad_resolution + dropped_no_waters + total_qualifying
            == total_tokens), "token accounting does not balance"

    stats = {
        # Token-level (all sum to total_tokens)
        "total_tokens": total_tokens,
        "dups_within_cluster": dups_within_cluster,
        "dropped_not_in_manifest": dropped_not_in_manifest,
        "dropped_not_xray": dropped_not_xray,
        "dropped_bad_resolution": dropped_bad_resolution,
        "dropped_no_waters": dropped_no_waters,
        "total_qualifying_pdb_in_clusters": total_qualifying,
        # Cluster-level
        "clusters_with_any_qualifying": len(qualifying_pdb_counts),
        "clusters_after_size_filter": len(filtered_clusters),
        "qualifying_pdb_counts": qualifying_pdb_counts,
        "resolution_cutoff": resolution_cutoff,
        "min_cluster_size": min_cluster_size,
    }
    return filtered_clusters, stats
