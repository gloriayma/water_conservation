from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

from boltzgen.data import const
from boltzgen.data.data import Structure

from basic_helpers import resolve_npz_path

from impute_solvents_from_triples import *
from gloria_hbond_helpers import *


NPZ_ROOT = Path("/data/rbg/shared/datasets/processed_rcsb/rcsb_solvents/structures")
TRIPLE_HBOND_PAIR_MAX_DIST = 5.6
COLLISION_MIN_DIST = 1.0


@dataclass(frozen=True, slots=True)
class WaterRecallResult:
    num_waters_a: int
    num_waters_b: int
    num_matching_waters: int
    recall: float
    match_mask: np.ndarray
    nearest_indices_in_b: np.ndarray
    nearest_distances: np.ndarray


def _as_coordinate_array(coords: np.ndarray, name: str) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3), got {coords.shape}.")
    return coords


def count_matching_waters_fast(
    water_A_coords: np.ndarray,
    water_B_coords: np.ndarray,
    cutoff: float,
) -> int:
    water_A_coords = _as_coordinate_array(water_A_coords, "water_A_coords")
    water_B_coords = _as_coordinate_array(water_B_coords, "water_B_coords")
    if cutoff < 0:
        raise ValueError("cutoff must be non-negative.")
    if len(water_A_coords) == 0 or len(water_B_coords) == 0:
        return 0

    tree_B = cKDTree(water_B_coords)
    dists, _ = tree_B.query(
        water_A_coords,
        k=1,
        distance_upper_bound=cutoff,
        workers=-1,
    )
    return int(np.isfinite(dists).sum())


def compare_water_coords(
    water_A_coords: np.ndarray,
    water_B_coords: np.ndarray,
    cutoff: float,
) -> WaterRecallResult:
    """Compare water coordinates in A against waters in B."""
    water_A_coords = _as_coordinate_array(water_A_coords, "water_A_coords")
    water_B_coords = _as_coordinate_array(water_B_coords, "water_B_coords")
    if cutoff < 0:
        raise ValueError("cutoff must be non-negative.")

    num_waters_a = len(water_A_coords)
    num_waters_b = len(water_B_coords)
    match_mask = np.zeros(num_waters_a, dtype=bool)
    nearest_indices_in_b = np.full(num_waters_a, -1, dtype=np.int64)
    nearest_distances = np.full(num_waters_a, np.inf, dtype=float)

    if num_waters_a > 0 and num_waters_b > 0:
        tree_B = cKDTree(water_B_coords)
        dists, indices = tree_B.query(
            water_A_coords,
            k=1,
            distance_upper_bound=cutoff,
            workers=-1,
        )
        match_mask = np.isfinite(dists)
        nearest_indices_in_b[match_mask] = indices[match_mask].astype(np.int64)
        nearest_distances[match_mask] = dists[match_mask]

    num_matching_waters = int(match_mask.sum())
    recall = num_matching_waters / num_waters_a if num_waters_a > 0 else 0.0
    return WaterRecallResult(
        num_waters_a=num_waters_a,
        num_waters_b=num_waters_b,
        num_matching_waters=num_matching_waters,
        recall=recall,
        match_mask=match_mask,
        nearest_indices_in_b=nearest_indices_in_b,
        nearest_distances=nearest_distances,
    )


def _extract_solvent_coords(structure: Structure) -> np.ndarray:
    """Extract one coordinate per solvent chain using the first present atom."""
    atom_coords = structure.coords["coords"]
    atom_present = structure.atoms["is_present"]
    solvent_atom_indices = []

    for chain in structure.chains:
        if chain["mol_type"] not in (
            const.chain_type_ids["SOLVENT"],
            const.chain_type_ids["iSOLVENT"],
        ):
            continue

        atom_start = chain["atom_idx"]
        atom_end = atom_start + chain["atom_num"]
        present_atom_offsets = np.flatnonzero(atom_present[atom_start:atom_end])
        if len(present_atom_offsets) == 0:
            continue
        solvent_atom_indices.append(atom_start + present_atom_offsets[0])

    if not solvent_atom_indices:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(atom_coords[np.asarray(solvent_atom_indices, dtype=np.int64)], dtype=float)


def compare_structure_waters(
    structure_A: Structure,
    structure_B: Structure,
    cutoff: float,
) -> WaterRecallResult:
    """Compare waters in structure A against waters in structure B."""
    water_A_coords = _extract_solvent_coords(structure_A)
    water_B_coords = _extract_solvent_coords(structure_B)
    return compare_water_coords(water_A_coords, water_B_coords, cutoff)




def recall_result(PDB_ID, recall_threshold):
    npz_path = resolve_npz_path(PDB_ID, NPZ_ROOT)
    gt_structure = Structure.load(npz_path)
    gt_structure = gt_structure.to_one_solvent_per_chain(gt_structure)
    gt_3hbonds_structure = gloria_remove_weak_solvents(
        gt_structure,
        min_hbonds=3,
    )

    gt_stripped_structure = gt_structure.remove_solvents()
    imputed = impute_solvents_from_atom_triples(
        gt_stripped_structure,
        one_solvent_per_chain=True,
        max_pair_dist=TRIPLE_HBOND_PAIR_MAX_DIST,
    )
    no_collisions = filter_solvent_clashes(
        imputed,
        min_dist=COLLISION_MIN_DIST,
    )

    result = compare_structure_waters(
        gt_3hbonds_structure,
        no_collisions,
        cutoff=recall_threshold,
    )
    return result
