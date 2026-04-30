from pathlib import Path
from collections import defaultdict

def parse_clusters(path):
    """
    Parse an RCSB cluster file.

    Returns:
        clusters: list of lists of entity IDs (e.g., ['6LU7_1', '5R7Y_1', ...]),
                  one inner list per cluster, in the file's original order
                  (largest cluster first).
    """
    clusters = []
    with open(path) as f:
        for line in f:
            entities = line.split()
            entities = [e for e in entities if is_experimental(e)]
            if entities:  # skip blank lines if any
                clusters.append(entities)
    return clusters


def is_experimental(entity_id):
    pdb_part = entity_id.split("_")[0]
    return len(pdb_part) == 4 and pdb_part.isalnum()


def entity_to_pdb(entity_id):
    """'6LU7_1' -> ('6LU7', 1)"""
    pdb_id, entity_num = entity_id.split("_")
    return pdb_id, int(entity_num)


def cluster_pdb_ids(cluster):
    """Return the unique PDB IDs in a cluster, preserving order.

    Note: a single PDB entry can appear multiple times in one cluster if it
    has multiple polymer entities with the same sequence (e.g., a homodimer
    where chains A and B are entities 1 and 2). For water analysis we'd
    typically dedupe and pick one chain per entry.
    """
    seen = set()
    pdb_ids = []
    for ent in cluster:
        pdb_id, _ = entity_to_pdb(ent)
        if pdb_id not in seen:
            seen.add(pdb_id)
            pdb_ids.append(pdb_id)
    return pdb_ids