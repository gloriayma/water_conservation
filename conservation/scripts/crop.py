"""
Greedy sphere-covering crop of a structure into spatial patches.

Crops define a canonical set of local regions on a reference structure.
Downstream local alignment within each crop avoids the error-propagation
problem of global superposition for large proteins.

Only non-water residues seed crops. Each seed captures all residues
(including water) within `radius` A of the seed center atom. Crops may
overlap: a residue within radius of two seeds appears in both crops.

To crop cluster members to match a reference: use crop_sample(), which
matches non-solvent residues by (entity_id, sym_id, local chain position)
and includes waters independently by proximity to the corresponding seed.
"""

import numpy as np
from scipy.spatial import cKDTree
import sys
from pathlib import Path

MOL_TYPE_POLYMER = 0
MOL_TYPE_LIGAND = 3
MOL_TYPE_WATER = 4

_FOLDEVERYTHING_SRC = "/data/rbg/users/gloriama/dev/foldeverything/src"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _arrays(struct_or_npz):
    """Unpack (atoms, residues, chains) from a Structure object or npz/dict."""
    if hasattr(struct_or_npz, "atoms"):
        return struct_or_npz.atoms, struct_or_npz.residues, struct_or_npz.chains
    return struct_or_npz["atoms"], struct_or_npz["residues"], struct_or_npz["chains"]


def _build_res_chain_map(struct_or_npz):
    """
    Build per-residue chain membership arrays.

    Returns:
      chain_idx:  (n_res,) int32 -- which chain each residue belongs to
      local_pos:  (n_res,) int32 -- 0-based position within that chain
    """
    _, residues, chains = _arrays(struct_or_npz)
    n_res = len(residues)
    chain_idx = np.empty(n_res, dtype=np.int32)
    local_pos = np.empty(n_res, dtype=np.int32)
    for ci, chain in enumerate(chains):
        r_start = int(chain["res_idx"])
        r_num = int(chain["res_num"])
        chain_idx[r_start : r_start + r_num] = ci
        local_pos[r_start : r_start + r_num] = np.arange(r_num, dtype=np.int32)
    return chain_idx, local_pos


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_nonsolvent_residues(struct_or_npz) -> int:
    """Count residues belonging to polymer (mol_type==0) or ligand (mol_type==3) chains."""
    _, _, chains = _arrays(struct_or_npz)
    mask = (chains["mol_type"] == MOL_TYPE_POLYMER) | (chains["mol_type"] == MOL_TYPE_LIGAND)
    return int(chains["res_num"][mask].sum())


def get_residue_centers(struct_or_npz):
    """
    Return the canonical center coordinate for each present residue.

    Uses residues["atom_center"] -- the precomputed representative atom index
    stored in the npz schema (CA for polymers, O for waters). Only residues
    where both the residue and its center atom are marked present are included.

    Returns:
      centers:        (N, 3) float32
      is_water:       (N,) bool -- True for mol_type==4 residues
      res_global_idx: (N,) int  -- index into the residues array
    """
    atoms, residues, chains = _arrays(struct_or_npz)

    n_res = len(residues)
    res_is_water = np.zeros(n_res, dtype=bool)
    for chain in chains:
        if chain["mol_type"] == MOL_TYPE_WATER:
            r_start = int(chain["res_idx"])
            r_end = r_start + int(chain["res_num"])
            res_is_water[r_start:r_end] = True

    centers, is_water_out, res_idx_out = [], [], []
    for ri, res in enumerate(residues):
        if not res["is_present"]:
            continue
        center_atom = atoms[int(res["atom_center"])]
        if not center_atom["is_present"]:
            continue
        centers.append(center_atom["coords"].copy())
        is_water_out.append(bool(res_is_water[ri]))
        res_idx_out.append(ri)

    if not centers:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=np.int32),
        )
    return (
        np.array(centers, dtype=np.float32),
        np.array(is_water_out, dtype=bool),
        np.array(res_idx_out, dtype=np.int32),
    )


def greedy_crop(struct_or_npz, radius: float = 20.0) -> list:
    """
    Greedy sphere-covering of non-water residue centers.

    Only non-water residues (polymer + ligand) can seed crops. Each seed
    captures all residues within `radius` A, including nearby waters. Once
    a residue is covered it cannot become a seed; crops may overlap.

    Returns list of dicts:
      seed_res_idx:       global residue index of the seed
      res_global_indices: all global residue indices in this crop (incl. water)
      n_nonsolvent:       count of polymer/ligand residues in crop
      n_water:            count of water residues in crop
    """
    centers, is_water, res_global_idx = get_residue_centers(struct_or_npz)
    N = len(centers)
    if N == 0:
        return []

    tree = cKDTree(centers)
    covered = np.zeros(N, dtype=bool)
    crops = []

    for i in np.where(~is_water)[0]:  # only non-water residues can be seeds
        if covered[i]:
            continue

        neighbors = sorted(tree.query_ball_point(centers[i], r=radius))
        covered[neighbors] = True

        neighbor_is_water = is_water[neighbors]
        crops.append({
            "seed_res_idx":       int(res_global_idx[i]),
            "res_global_indices": [int(res_global_idx[j]) for j in neighbors],
            "n_nonsolvent":       int((~neighbor_is_water).sum()),
            "n_water":            int(neighbor_is_water.sum()),
        })

    return crops


def crop_sample(ref_struct, ref_crop: dict, sample_struct, water_radius: float = 20.0) -> dict:
    """
    Extract a crop from sample_struct corresponding to a reference crop.

    Non-solvent correspondence: residues matched by (entity_id, sym_id, local
    position within chain). This works for same-sequence clusters because 100%
    identity guarantees the same chain layout. Residues absent in the sample
    (missing density or shorter chain) are recorded as None.

    Water inclusion: all present waters in sample within water_radius of the
    sample's corresponding seed center atom. Falls back to the mean center of
    all matched non-solvent residues if the seed is absent in the sample.

    TODO: explore alternative water inclusion -- all waters within a smaller
    radius (e.g. 4-5 A) of ANY polymer/ligand atom in the sample crop, rather
    than a single sphere around the seed center. This would give denser local
    hydration shells and could be more relevant for conservation scoring.

    Returns dict:
      ref_ns_indices:       list[int]       non-solvent residue idx in reference (ordered)
      sample_ns_indices:    list[int|None]  corresponding idx in sample (None = absent)
      sample_water_indices: list[int]       water residue idx in sample within radius
      n_matched:            int             non-None entries in sample_ns_indices
      n_missing:            int             None entries
      n_water:              int             len(sample_water_indices)
      used_seed_fallback:   bool            True if seed was absent; used mean center
    """
    _, _, ref_chains = _arrays(ref_struct)
    sample_atoms, sample_residues, sample_chains = _arrays(sample_struct)

    ref_chain_idx, ref_local_pos = _build_res_chain_map(ref_struct)

    # Map (entity_id, sym_id) -> chain_idx in sample (first match wins)
    sample_entity_to_chain = {}
    for ci, chain in enumerate(sample_chains):
        key = (int(chain["entity_id"]), int(chain["sym_id"]))
        if key not in sample_entity_to_chain:
            sample_entity_to_chain[key] = ci

    # Match each non-water reference residue to sample
    ref_ns_indices = []
    sample_ns_indices = []

    for r_ref in ref_crop["res_global_indices"]:
        ci_ref = int(ref_chain_idx[r_ref])
        if int(ref_chains[ci_ref]["mol_type"]) == MOL_TYPE_WATER:
            continue

        ref_ns_indices.append(r_ref)
        lpos = int(ref_local_pos[r_ref])
        key = (int(ref_chains[ci_ref]["entity_id"]), int(ref_chains[ci_ref]["sym_id"]))

        r_sample = None
        if key in sample_entity_to_chain:
            ci_s = sample_entity_to_chain[key]
            if lpos < int(sample_chains[ci_s]["res_num"]):
                r_sample = int(sample_chains[ci_s]["res_idx"]) + lpos
        sample_ns_indices.append(r_sample)

    # Locate the seed center in the sample's coordinate frame
    seed_ref = ref_crop["seed_res_idx"]
    ci_ref_seed = int(ref_chain_idx[seed_ref])
    seed_key = (int(ref_chains[ci_ref_seed]["entity_id"]), int(ref_chains[ci_ref_seed]["sym_id"]))
    seed_lpos = int(ref_local_pos[seed_ref])

    seed_center = None
    used_fallback = False

    if seed_key in sample_entity_to_chain:
        ci_s = sample_entity_to_chain[seed_key]
        if seed_lpos < int(sample_chains[ci_s]["res_num"]):
            r_s = int(sample_chains[ci_s]["res_idx"]) + seed_lpos
            res_s = sample_residues[r_s]
            if res_s["is_present"]:
                ca = sample_atoms[int(res_s["atom_center"])]
                if ca["is_present"]:
                    seed_center = ca["coords"].astype(np.float64)

    if seed_center is None:
        # Fallback: mean of all matched non-solvent center atoms
        coords = []
        for r_s in sample_ns_indices:
            if r_s is None:
                continue
            res_s = sample_residues[r_s]
            if not res_s["is_present"]:
                continue
            ca = sample_atoms[int(res_s["atom_center"])]
            if ca["is_present"]:
                coords.append(ca["coords"])
        if coords:
            seed_center = np.mean(coords, axis=0)
            used_fallback = True

    # Collect present water centers from sample
    water_centers = []
    water_res_idx = []
    for chain in sample_chains:
        if int(chain["mol_type"]) != MOL_TYPE_WATER:
            continue
        r_start = int(chain["res_idx"])
        r_end = r_start + int(chain["res_num"])
        for ri in range(r_start, r_end):
            res = sample_residues[ri]
            if not res["is_present"]:
                continue
            ca = sample_atoms[int(res["atom_center"])]
            if not ca["is_present"]:
                continue
            water_centers.append(ca["coords"])
            water_res_idx.append(ri)

    sample_water_indices = []
    if seed_center is not None and water_centers:
        wc = np.array(water_centers, dtype=np.float64)
        dists = np.linalg.norm(wc - seed_center, axis=1)
        sample_water_indices = [water_res_idx[i] for i in range(len(water_res_idx)) if dists[i] <= water_radius]

    n_matched = sum(1 for r in sample_ns_indices if r is not None)
    n_missing = sum(1 for r in sample_ns_indices if r is None)

    return {
        "ref_ns_indices":       ref_ns_indices,
        "sample_ns_indices":    sample_ns_indices,
        "sample_water_indices": sample_water_indices,
        "n_matched":            n_matched,
        "n_missing":            n_missing,
        "n_water":              len(sample_water_indices),
        "used_seed_fallback":   used_fallback,
    }


def save_crop_as_mmcif(struct, res_global_indices, output_path) -> None:
    """
    Write a spatial crop (subset of residues) to an mmCIF file.

    Uses BoltzGen's Structure.extract_residues for correct reindexing and
    to_mmcif for format conversion. struct must be a Structure object.

    Args:
      struct:             boltzgen.data.data.Structure
      res_global_indices: list or array of global residue indices to include
      output_path:        Path or str for the output .cif file
    """
    if _FOLDEVERYTHING_SRC not in sys.path:
        sys.path.insert(0, _FOLDEVERYTHING_SRC)
    from boltzgen.data.data import Structure
    from boltzgen.data.write.mmcif import to_mmcif

    indices = np.array(res_global_indices, dtype=np.int32)
    cropped = Structure.extract_residues(struct, indices)
    cif_str = to_mmcif(cropped)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(cif_str)
