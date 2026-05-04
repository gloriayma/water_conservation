"""
Structural alignment of representative protein chains across a cluster.

Uses gemmi for both sequence alignment (align_string_sequences, C++) and
structural superposition (superpose_positions, C++). No custom DP or Kabsch.

Strategy:
  - Representative chain: first valid polymer chain (mol_type=0) per structure.
  - Reference: highest-resolution entry in the cluster.
  - All entries aligned onto the reference.
  - Large clusters (> LARGE_CLUSTER_THRESHOLD): flagged in stats but handled identically
    (all still aligned onto the single reference).
"""

import re
import numpy as np
import gemmi

from npz_io import load_npz, get_first_valid_polymer_chain_idx

LARGE_CLUSTER_THRESHOLD = 100


def get_ca_and_seq(npz: dict, chain_idx: int) -> tuple[np.ndarray, list[str]]:
    """
    Return (ca_coords, residue_names) for the given polymer chain.
    Only includes residues where the CA atom is marked present.
    """
    chains   = npz["chains"]
    residues = npz["residues"]
    atoms    = npz["atoms"]

    chain    = chains[chain_idx]
    r_start  = chain["res_idx"]
    r_end    = r_start + chain["res_num"]
    a_start  = chain["atom_idx"]

    chain_residues = residues[r_start:r_end]
    chain_atoms    = atoms[a_start: a_start + chain["atom_num"]]

    ca_coords, names = [], []
    for res in chain_residues:
        local_start = res["atom_idx"] - a_start
        res_atoms   = chain_atoms[local_start: local_start + res["atom_num"]]
        ca_mask     = (res_atoms["name"] == "CA") & res_atoms["is_present"]
        if ca_mask.any():
            ca_coords.append(res_atoms["coords"][ca_mask][0])
            names.append(str(res["name"]))

    return np.array(ca_coords, dtype=np.float32), names


def matched_indices(cigar: str, names_q: list[str], names_t: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a gemmi cigar string and return indices of matched (identical) residue pairs.
    M = aligned pair; I = insertion in query (gap in target); D = deletion in query.
    """
    qi, ti, idx_q, idx_t = 0, 0, [], []
    for m in re.finditer(r'(\d+)([MIDSH])', cigar):
        length, op = int(m.group(1)), m.group(2)
        if op == 'M':
            for _ in range(length):
                if names_q[qi] == names_t[ti]:
                    idx_q.append(qi)
                    idx_t.append(ti)
                qi += 1; ti += 1
        elif op == 'I':
            qi += length
        elif op == 'D':
            ti += length
    return np.array(idx_q), np.array(idx_t)


def align_and_superpose(
    ca_q: np.ndarray, names_q: list[str],
    ca_r: np.ndarray, names_r: list[str],
) -> tuple[float, int, np.ndarray, np.ndarray]:
    """
    Sequence-align then structurally superpose query onto reference.
    Uses gemmi.align_string_sequences (C++) + gemmi.superpose_positions (C++).

    Returns (rmsd, n_aligned, R, t) where R is (3,3) rotation and t is (3,) translation
    such that ca_q_aligned = ca_q @ R.T + t minimises RMSD against ca_r.
    """
    aln = gemmi.align_string_sequences(names_q, names_r, [])
    idx_q, idx_r = matched_indices(aln.cigar_str(), names_q, names_r)

    if len(idx_q) < 4:
        raise ValueError(f"too few matched residues: {len(idx_q)}")

    pos_q = [gemmi.Position(*c) for c in ca_q[idx_q]]
    pos_r = [gemmi.Position(*c) for c in ca_r[idx_r]]

    sup = gemmi.superpose_positions(pos_q, pos_r)

    R = np.array(sup.transform.mat.tolist(), dtype=np.float64)   # (3, 3)
    t = np.array([sup.transform.vec.x,
                  sup.transform.vec.y,
                  sup.transform.vec.z], dtype=np.float64)         # (3,)

    return sup.rmsd, len(idx_q), R, t


def pick_reference(cluster: list[str], manifest_index: dict) -> str:
    """Return the PDB ID with the lowest (best) resolution in the cluster."""
    def res(pdb):
        r = manifest_index[pdb]["resolution"]
        return r if r is not None else 99.0
    return min(cluster, key=res)


def align_cluster(cluster: list[str], manifest_index: dict) -> dict:
    """
    Align all entries in a cluster onto the highest-resolution reference.

    Returns:
      reference_pdb:  str
      is_large:       bool (cluster size > LARGE_CLUSTER_THRESHOLD)
      alignments:     list of dicts, one per cluster member:
                        pdb_id, rmsd, n_aligned, R (3x3), t (3,), failed (bool)
                      Reference entry has rmsd=0, R=identity, t=zeros.
    """
    ref_pdb = pick_reference(cluster, manifest_index)

    try:
        ref_npz       = load_npz(ref_pdb)
        ref_chain_idx = get_first_valid_polymer_chain_idx(ref_npz)
        if ref_chain_idx is None:
            return {"reference_pdb": ref_pdb, "is_large": False, "alignments": [],
                    "error": "no polymer chain in reference"}
        ref_ca, ref_names = get_ca_and_seq(ref_npz, ref_chain_idx)
    except Exception as e:
        return {"reference_pdb": ref_pdb, "is_large": False, "alignments": [],
                "error": f"reference load failed: {e}"}

    alignments = []
    for pdb_id in cluster:
        if pdb_id == ref_pdb:
            alignments.append({
                "pdb_id":    pdb_id,
                "rmsd":      0.0,
                "n_aligned": len(ref_ca),
                "R":         np.eye(3),
                "t":         np.zeros(3),
                "failed":    False,
            })
            continue
        try:
            npz       = load_npz(pdb_id)
            chain_idx = get_first_valid_polymer_chain_idx(npz)
            if chain_idx is None:
                raise ValueError("no polymer chain")
            ca, names = get_ca_and_seq(npz, chain_idx)
            rmsd, n_aligned, R, t = align_and_superpose(ca, names, ref_ca, ref_names)
            alignments.append({
                "pdb_id":    pdb_id,
                "rmsd":      rmsd,
                "n_aligned": n_aligned,
                "R":         R,
                "t":         t,
                "failed":    False,
            })
        except Exception as e:
            alignments.append({"pdb_id": pdb_id, "failed": True, "reason": str(e)})

    return {
        "reference_pdb": ref_pdb,
        "is_large":      len(cluster) > LARGE_CLUSTER_THRESHOLD,
        "alignments":    alignments,
    }


def transform_coords(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rotation R and translation t to coordinates (N, 3)."""
    return coords @ R.T + t
