"""
Parse the RCSB 100% sequence-identity cluster file and apply quality filters.

Cluster file format (clusters-by-entity-100.txt):
  One cluster per line. Each line is a space-separated list of PDBID_entityID tokens
  (PDB IDs in uppercase), e.g.:
    1ABC_1 2DEF_1 3GHI_2 ...

Filtering pipeline:
  1. Parse cluster file into lists of (pdb_id, entity_id) pairs.
  2. Build a lookup from pdb_id -> metadata from manifest.json.
  3. For each cluster, keep only members whose PDB entry passes:
       - method == "x-ray diffraction"
       - resolution <= resolution_cutoff
       - has at least one water chain (mol_type==4) with >0 residues
  4. Deduplicate to unique PDB IDs per cluster (a PDB entry can have multiple
     entities in the same cluster if it is a homo-oligomer at 100% identity,
     but we want one entry per structure).
  5. Keep clusters with >= min_cluster_size qualifying PDB entries.
"""

import json
from pathlib import Path
from collections import Counter

CONSERVATION_DIR = Path("/data/rbg/users/gloriama/dev/water_conservation/conservation")
CLUSTER_FILE = CONSERVATION_DIR / "clusters-by-entity-100.txt"
MANIFEST_FILE = Path("/data/rbg/shared/datasets/processed_rcsb/rcsb_solvents/manifest.json")


def load_manifest_index(manifest_path: Path = MANIFEST_FILE) -> dict:
    """
    Load manifest.json and return a dict keyed by lowercase PDB ID.

    Each value has:
      resolution: float or None
      method: str
      has_waters: bool  (True if any chain has mol_type==4 with res_num > 0)
    """
    with open(manifest_path) as f:
        entries = json.load(f)

    index = {}
    for e in entries:
        pdb_id = e["id"].lower()
        structure = e["structure"]
        has_waters = any(
            c["mol_type"] == 4 and c["num_residues"] > 0
            for c in e["chains"]
        )
        index[pdb_id] = {
            "resolution": structure.get("resolution"),
            "method": structure.get("method", ""),
            "has_waters": has_waters,
        }
    return index


def parse_cluster_line(line: str) -> list[tuple[str, str]]:
    """
    Parse one line of the cluster file into (pdb_id_lower, entity_id) tuples.

    Input:  '1ABC_1 2DEF_1 3GHI_2'
    Output: [('1abc', '1'), ('2def', '1'), ('3ghi', '2')]
    """
    members = []
    for token in line.strip().split():
        if "_" not in token:
            continue
        pdb, entity = token.rsplit("_", 1)
        members.append((pdb.lower(), entity))
    return members


def is_qualifying_entry(
    pdb_id: str,
    manifest_index: dict,
    resolution_cutoff: float,
) -> bool:
    """Return True if the PDB entry passes X-ray, resolution, and water filters."""
    meta = manifest_index.get(pdb_id)
    if meta is None:
        return False
    if meta["method"] != "x-ray diffraction":
        return False
    if meta["resolution"] is None or meta["resolution"] > resolution_cutoff:
        return False
    if not meta["has_waters"]:
        return False
    return True


def filter_clusters(
    cluster_file: Path = CLUSTER_FILE,
    manifest_index: dict = None,
    resolution_cutoff: float = 2.0,
    min_cluster_size: int = 5,
) -> tuple[list[list[str]], dict]:
    """
    Parse the cluster file and return filtered clusters with statistics.

    Returns:
      filtered_clusters: list of lists, each inner list is qualifying PDB IDs
                         (lowercase, deduplicated) for one cluster
      stats: dict with counts at each filtering stage
    """
    if manifest_index is None:
        manifest_index = load_manifest_index()

    total_clusters = 0
    total_members_raw = 0
    clusters_with_any_qualifying = 0
    qualifying_pdb_counts = []  # per cluster, after all filters before size cutoff
    filtered_clusters = []

    # Track how many were dropped at each stage (per member across all clusters)
    dropped_not_in_manifest = 0
    dropped_not_xray = 0
    dropped_resolution = 0
    dropped_no_waters = 0

    with open(cluster_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_clusters += 1
            members = parse_cluster_line(line)
            total_members_raw += len(members)

            # Deduplicate to unique PDB IDs (same PDB, multiple entities -> count once)
            seen_pdbs = set()
            qualifying = []

            for pdb_id, _ in members:
                if pdb_id in seen_pdbs:
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
                    dropped_resolution += 1
                    continue
                if not meta["has_waters"]:
                    dropped_no_waters += 1
                    continue

                qualifying.append(pdb_id)

            if qualifying:
                clusters_with_any_qualifying += 1
                qualifying_pdb_counts.append(len(qualifying))

            if len(qualifying) >= min_cluster_size:
                filtered_clusters.append(qualifying)

    stats = {
        "total_clusters_in_file": total_clusters,
        "total_member_tokens": total_members_raw,
        "clusters_with_any_qualifying": clusters_with_any_qualifying,
        "clusters_after_size_filter": len(filtered_clusters),
        "dropped_not_in_manifest": dropped_not_in_manifest,
        "dropped_not_xray": dropped_not_xray,
        "dropped_bad_resolution": dropped_resolution,
        "dropped_no_waters": dropped_no_waters,
        "qualifying_pdb_counts": qualifying_pdb_counts,  # per-cluster counts (before size filter)
        "resolution_cutoff": resolution_cutoff,
        "min_cluster_size": min_cluster_size,
    }

    return filtered_clusters, stats


def cluster_size_histogram(qualifying_pdb_counts: list[int], bins=None) -> dict:
    """
    Compute a histogram of cluster sizes from per-cluster counts.

    Returns dict with bin edges and counts.
    """
    import numpy as np
    counts = np.array(qualifying_pdb_counts)
    if bins is None:
        # Useful breakpoints for the write-up
        bins = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10000]
    hist, edges = np.histogram(counts, bins=bins)
    return {"bin_edges": edges.tolist(), "counts": hist.tolist()}
