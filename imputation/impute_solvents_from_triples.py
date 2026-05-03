from __future__ import annotations
from time import perf_counter

import numpy as np
from scipy.spatial import KDTree

from boltzgen.data import const
from boltzgen.data.data import Chain, Structure

from gloria_hbond_helpers import *

# from new_impute_solvent import get_circumcenter


HBOND_CANDIDATE_PREFIXES = ("N", "O", "F", "S")
HBOND_PAIR_MIN_DIST = 0 #3.13
HBOND_PAIR_MAX_DIST = 5.6 #8# 5.09


def get_atom_to_residue_index(structure: Structure) -> np.ndarray:
    atom_to_residue = np.full(len(structure.atoms), -1, dtype=np.int64)
    for res_idx, residue in enumerate(structure.residues):
        atom_start = residue["atom_idx"]
        atom_end = atom_start + residue["atom_num"]
        atom_to_residue[atom_start:atom_end] = res_idx
    return atom_to_residue


def get_circumcenter(a, b, c, epsilon=1e-4):
    """
    Calculates the circumcenter of three points a, b, and c, and returns
    - the circumcenter,
    - the circumradius,
    - the normal vector of the plane containing the three points

    Returns none if the points are collinear.
    """
    u = b - a
    v = c - a
    uxv = np.cross(u, v)

    # Collinearity check
    if np.linalg.norm(uxv) < epsilon:
        return None, None, None

    numerator = np.cross(np.linalg.norm(u) ** 2 * v - np.linalg.norm(v) ** 2 * u, uxv)
    denominator = 2 * np.linalg.norm(uxv) ** 2
    circumcenter = a + numerator / denominator
    circumradius = np.linalg.norm(numerator / denominator)
    normal = uxv / np.linalg.norm(uxv)
    return circumcenter, circumradius, normal


def get_hbond_candidate_atom_data(
    structure: Structure,
) -> tuple[np.ndarray, np.ndarray]:
    atoms = structure.atoms
    coords = structure.coords["coords"]
    hbond_candidate_mask = np.zeros(len(atoms), dtype=bool)
    for prefix in HBOND_CANDIDATE_PREFIXES:
        hbond_candidate_mask |= np.char.startswith(atoms["name"], prefix)
    hbond_candidate_mask &= atoms["is_present"]
    candidate_atom_indices = np.flatnonzero(hbond_candidate_mask)
    candidate_atom_coords = coords[hbond_candidate_mask]
    print(f"Number of candidate atoms: {len(candidate_atom_indices)}")
    return candidate_atom_indices, candidate_atom_coords


def place_water_from_atom_triple(
    # atom_coords: np.ndarray,
    coords_i, coords_j, coords_k,
    max_hbond_length: float,
    hbond_length: float = 2.8,
    epsilon: float = 1e-4,
) -> np.ndarray | None:
    """
    Return an array of placed water oxygen coordinates with shape (n, 3).

    Find the circumcenter, and follow the normal vector. 

    This yields two solutions. We return both. 

    Returns none if the atoms are collinear (TODO). If the circumradius is greater than hbond_length, but less
    than max_hbond_length, place at circumcenter. Otherwise, raise an error.
    """

    circumcenter, circumradius, normal = get_circumcenter(
        coords_i,
        coords_j,
        coords_k,
        epsilon=epsilon,
    )

    if circumcenter is None:
        print("Collinear atoms. TODO: handle this")
        return None
    if circumradius > max_hbond_length:
        # raise ValueError(f"Circumradius {circumradius} is greater than max_hbond_length {max_hbond_length}")
        # print(f"Circumradius {circumradius} is greater than max_hbond_length {max_hbond_length}. TODO: handle this")
        return None

    if circumradius > hbond_length:
        # Keep the return shape consistent with the two-solution case.
        return circumcenter[None, :]

    height = np.sqrt(max(hbond_length**2 - circumradius**2, 0.0))
    return np.stack(
        (circumcenter + height * normal, circumcenter - height * normal),
        axis=0,
    )



def kdtree_find_water_coords_for_three_hbonds(
    structure: Structure,
    max_hbond_length: float,
    hbond_length: float = 2.8,
    # min_pair_dist: float = HBOND_PAIR_MIN_DIST,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """
    New version: returns the coordinates of the placed water(s) for each triple. 
    (Either 0, 1, or 2 coordinates per triple.)
        Find atom triples for three-H-bond water placement using a k-d tree.
        Pairwise distances are all less than max_pair_dist.


    """
    num_triples = 0
    candidate_atom_indices, candidate_atom_coords = get_hbond_candidate_atom_data(
        structure
    )
    atom_to_residue = get_atom_to_residue_index(structure)
    candidate_residue_indices = atom_to_residue[candidate_atom_indices]
    kdtree_find_start = perf_counter()


    if len(candidate_atom_indices) < 3:
        return np.zeros((0, 3), dtype=np.int64)

    kdtree = KDTree(candidate_atom_coords)
    max_pair_dist = 2*max_hbond_length
    neighbor_lists = kdtree.query_ball_tree(kdtree, r=max_pair_dist)
    print(f"Number of neighbor lists: {len(neighbor_lists)}")
    print(f"Neighbor lists: {neighbor_lists[:10]}")
    
    neighbors = []
    for i, neighbor in enumerate(neighbor_lists):
        neighbors.append(set(j for j in neighbor if j > i))  # j > i avoids duplicates

    neighbor_list_time = perf_counter() - kdtree_find_start
    print(f"{neighbor_list_time=:.2f}s")
    place_water_start = perf_counter()

    placed_water_coords = [] # speed up by allocating array ahead of time? 

    for i in range(len(candidate_atom_coords)):
        for j in neighbors[i]:
            # k must be in neighbor list of both i and j
            common = neighbors[i].intersection(neighbors[j])
            for k in common:
                num_triples += 1
                if len(
                    {
                        candidate_residue_indices[i],
                        candidate_residue_indices[j],
                        candidate_residue_indices[k],
                    }
                ) < 3:
                    continue

                water_coords = place_water_from_atom_triple(
                    candidate_atom_coords[i],
                    candidate_atom_coords[j],
                    candidate_atom_coords[k],
                    max_hbond_length=max_hbond_length,
                    hbond_length=hbond_length,
                    epsilon=epsilon,
                )
                if water_coords is not None:
                    placed_water_coords.extend(water_coords)
    
    place_water_time = perf_counter() - place_water_start
    print(f"{place_water_time=:.2f}s")
    print(f"Number of triples: {num_triples}")
    print(f"Number of placed waters: {len(placed_water_coords)}")

    return np.array(placed_water_coords, dtype=candidate_atom_coords.dtype)


def impute_solvents_from_atom_triples(
    structure: Structure,
    max_hbond_length: float, 
    one_solvent_per_chain: bool = True,
    hbond_length: float = 2.8,
    # min_hbond_length: float,
    # max_pair_dist: float = HBOND_PAIR_MAX_DIST,
    epsilon: float = 1e-4,
) -> Structure:
    """
    Place one water for every valid 3-atom triple in the input structure.

    This function assumes the caller has already prepared the input structure
    (for example, by stripping waters beforehand if desired).
    """
  
    placed_water_coords = kdtree_find_water_coords_for_three_hbonds(
        structure,
        max_hbond_length=max_hbond_length,
        hbond_length=hbond_length,
        # min_pair_dist=min_pair_dist,
        epsilon=epsilon,
    )

    return _append_imputed_solvents(
        structure,
        placed_water_coords,
        one_solvent_per_chain=one_solvent_per_chain,
    )


def _append_imputed_solvents(
    structure: Structure,
    imputed_solvent_coords: np.ndarray,
    one_solvent_per_chain: bool = False,
) -> Structure:
    append_imputed_solvents_start = perf_counter()
    if len(imputed_solvent_coords) == 0:
        return structure

    placed_water_coords_array = np.zeros(len(imputed_solvent_coords), dtype=structure.coords.dtype)
    placed_water_coords_array["coords"] = np.asarray(imputed_solvent_coords)
    imputed_solvent_coords = placed_water_coords_array

    num_solvents = 0
    entities = set()
    solvent_entities = set()
    solvent_chain_mask = (
        (structure.chains["mol_type"] == const.chain_type_ids["SOLVENT"])
        | (structure.chains["mol_type"] == const.chain_type_ids["iSOLVENT"])
    )
    for chain in structure.chains:
        if chain["mol_type"] in (
            const.chain_type_ids["SOLVENT"],
            const.chain_type_ids["iSOLVENT"],
        ):
            num_solvents += chain["res_num"]
            solvent_entities.add(chain["entity_id"])
        entities.add(chain["entity_id"])
    assert len(solvent_entities) <= 1, (
        "Currently only one solvent entity id is supported for imputation."
    )

    chains = structure.chains.copy()
    atoms = structure.atoms.copy()
    residues = structure.residues.copy()
    mask = structure.mask.copy()

    if len(solvent_entities) == 0:
        solvent_entity_id = max(entities) + 1
        next_sym_id = 0
    else:
        solvent_entity_id = solvent_entities.pop()
        next_sym_id = int(structure.chains[solvent_chain_mask]["sym_id"].max()) + 1

    num_to_impute = len(imputed_solvent_coords)
    imputed_solvent_atoms = np.zeros(num_to_impute, dtype=atoms.dtype)
    imputed_solvent_residues = np.zeros(num_to_impute, dtype=residues.dtype)
    imputed_solvent_chains = None
    

    if not one_solvent_per_chain:
        atom_idx = chains[-1]["atom_idx"] + chains[-1]["atom_num"]
        res_idx = chains[-1]["res_idx"] + chains[-1]["res_num"]
        imaginary_solvent_chain = np.array(
            [
                (
                    "iH2O",
                    const.chain_type_ids["iSOLVENT"],
                    solvent_entity_id,
                    next_sym_id,
                    len(chains),
                    atom_idx,
                    num_to_impute,
                    res_idx,
                    num_to_impute,
                    0,
                )
            ],
            dtype=Chain,
        )
        chains = np.append(chains, imaginary_solvent_chain)
    else:
        base_atom_idx = chains[-1]["atom_idx"] + chains[-1]["atom_num"]
        base_chain_idx = len(chains)
        base_res_idx = len(residues)
        imputed_solvent_chains = np.zeros(num_to_impute, dtype=Chain)

    for impute_idx, new_coords in enumerate(imputed_solvent_coords["coords"]):
        imputed_solvent_atoms[impute_idx] = ("O", new_coords, True, 50.0, 1.0)
        if one_solvent_per_chain:
            solvent_atom_idx = base_atom_idx + impute_idx
            imputed_solvent_chains[impute_idx] = (
                f"Wa{num_solvents + impute_idx}",
                const.chain_type_ids["iSOLVENT"],
                solvent_entity_id,
                next_sym_id + impute_idx,
                base_chain_idx + impute_idx,
                solvent_atom_idx,
                1,
                base_res_idx + impute_idx,
                1,
                0,
            )
        else:
            solvent_atom_idx = imaginary_solvent_chain[0]["atom_idx"] + impute_idx

        imputed_solvent_residues[impute_idx] = (
            "HOH",
            const.token_ids["HOH"],
            len(residues) + impute_idx,
            solvent_atom_idx,
            1,
            solvent_atom_idx,
            solvent_atom_idx,
            False,
            True,
        )

    coords = np.append(structure.coords, imputed_solvent_coords)
    atoms = np.append(atoms, imputed_solvent_atoms)
    residues = np.append(residues, imputed_solvent_residues)
    if one_solvent_per_chain:
        chains = np.append(chains, imputed_solvent_chains)
        mask = np.append(mask, np.full(num_to_impute, True))
    else:
        mask = np.append(mask, True)
    append_imputed_solvents_time = perf_counter() - append_imputed_solvents_start

    print(f"{append_imputed_solvents_time=:.2f}s")

    return Structure(
        atoms=atoms,
        bonds=structure.bonds,
        residues=residues,
        chains=chains,
        interfaces=structure.interfaces,
        mask=mask,
        coords=coords,
        ensemble=structure.ensemble,
    )




def filter_solvent_clashes(
    structure: Structure,
    protein_clash_rad: float,
    solvent_clash_rad: float,
) -> Structure:
    """
    Remove solvent molecules whose oxygen clashes with a molecule within protein_clash_rad, 
    or wiht another solvent molecule within solvent_clash_rad.

    This is a two-stage post-processing filter:
    1. Drop solvent molecules whose oxygen clashes with any non-solvent atom.
    2. Among the remaining solvents, greedily keep a non-clashing subset in
       chain order.
    """
    structure = structure.to_one_solvent_per_chain(structure)
    mask = structure.mask.copy()
    atom_coords = structure.coords["coords"]
    atom_present = structure.atoms["is_present"]

    solvent_chain_indices = []
    solvent_atom_indices = []
    solvent_atom_mask = np.zeros(len(structure.atoms), dtype=bool)
    for chain_idx, chain in enumerate(structure.chains):
        if chain["mol_type"] not in (
            const.chain_type_ids["SOLVENT"],
            const.chain_type_ids["iSOLVENT"],
        ):
            continue

        atom_start = chain["atom_idx"]
        atom_end = atom_start + chain["atom_num"]
        solvent_atom_mask[atom_start:atom_end] = True

        present_atom_offsets = np.flatnonzero(atom_present[atom_start:atom_end])
        if len(present_atom_offsets) == 0:
            mask[chain_idx] = False
            continue

        solvent_chain_indices.append(chain_idx)
        solvent_atom_indices.append(atom_start + present_atom_offsets[0])

    if not solvent_chain_indices:
        return structure

    solvent_chain_indices = np.array(solvent_chain_indices, dtype=np.int64)
    solvent_atom_indices = np.array(solvent_atom_indices, dtype=np.int64)
    solvent_coords = atom_coords[solvent_atom_indices]
    nonsolvent_atom_mask = (~solvent_atom_mask) & atom_present
    nonsolvent_coords = atom_coords[nonsolvent_atom_mask]
    # clash_radius = np.nextafter(min_dist, 0.0)
    protein_clash_radius = np.nextafter(protein_clash_rad, 0.0)
    solvent_clash_radius = np.nextafter(solvent_clash_rad, 0.0)

    if len(nonsolvent_coords) > 0:
        fixed_atom_kdtree = KDTree(nonsolvent_coords)
        fixed_atom_hits = fixed_atom_kdtree.query_ball_point(solvent_coords, r=protein_clash_radius)
        clashes_with_nonsolvent = np.array(
            [len(hit_indices) > 0 for hit_indices in fixed_atom_hits],
            dtype=bool,
        )
        mask[solvent_chain_indices[clashes_with_nonsolvent]] = False
    else:
        clashes_with_nonsolvent = np.zeros(len(solvent_chain_indices), dtype=bool)

    surviving_chain_indices = solvent_chain_indices[~clashes_with_nonsolvent]
    surviving_coords = solvent_coords[~clashes_with_nonsolvent]
    print(f"Number of surviving waters after clash with protein: {len(surviving_chain_indices)}")
    if len(surviving_chain_indices) == 0:
        return rebuild_structure_with_mask(structure, mask)

    proposed_water_kdtree = KDTree(surviving_coords)
    clash_neighbors = proposed_water_kdtree.query_ball_tree(
        proposed_water_kdtree,
        r=solvent_clash_radius,
    )

    blocked = np.zeros(len(surviving_chain_indices), dtype=bool)
    for i, neighbor_indices in enumerate(clash_neighbors):
        if blocked[i]:
            mask[surviving_chain_indices[i]] = False
            continue

        for j in neighbor_indices:
            if j <= i:
                continue
            blocked[j] = True
            mask[surviving_chain_indices[j]] = False

    print(f"Number of surviving waters after clash with other waters: {np.sum(mask[surviving_chain_indices])}")
    return rebuild_structure_with_mask(structure, mask)
