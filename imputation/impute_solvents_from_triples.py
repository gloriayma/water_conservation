from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree

from boltzgen.data import const
from boltzgen.data.data import Chain, Structure

from new_impute_solvent import get_circumcenter


HBOND_CANDIDATE_PREFIXES = ("N", "O", "F", "S")
HBOND_PAIR_MIN_DIST = 3.13
HBOND_PAIR_MAX_DIST = 8# 5.09


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
    return candidate_atom_indices, candidate_atom_coords


def place_water_from_atom_triple(
    atom_coords: np.ndarray,
    hbond_length: float = 2.8,
    epsilon: float = 1e-4,
) -> np.ndarray | None:
    """
    Return a tuple of coordinates for two water oxygen atoms 
    equidistant from the three atoms in the triple, hbond_length away from each. 

    Find the circumcenter, and follow the normal vector. 

    This yields two solutions. We return both. 

    Returns none if the atoms are collinear, or if the circumradius is greater than hbond_length.
    """
    if atom_coords.shape != (3, 3):
        raise ValueError("atom_coords must have shape (3, 3)")

    circumcenter, circumradius, normal = get_circumcenter(
        atom_coords[0],
        atom_coords[1],
        atom_coords[2],
        epsilon=epsilon,
    )
    if circumcenter is None or circumradius > hbond_length:
        return None

    height = np.sqrt(max(hbond_length**2 - circumradius**2, 0.0))
    return (circumcenter + height * normal, circumcenter - height * normal)


def kdtree_find_atom_triples_for_three_hbonds(
    structure: Structure,
    hbond_length: float = 2.8,
    min_pair_dist: float = HBOND_PAIR_MIN_DIST,
    max_pair_dist: float = HBOND_PAIR_MAX_DIST,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """
    Find atom triples for three-H-bond water placement using a k-d tree.
    Returns atom indices into the input structure, with one row per triple.
    Pairwise distances are all less than max_pair_dist.
    """
    candidate_atom_indices, candidate_atom_coords = get_hbond_candidate_atom_data(
        structure
    )
    if len(candidate_atom_indices) < 3:
        return np.zeros((0, 3), dtype=np.int64)

    kdtree = KDTree(candidate_atom_coords)
    neighbor_lists = kdtree.query_ball_tree(kdtree, r=max_pair_dist)
    print(f"Number of neighbor lists: {len(neighbor_lists)}")
    print(f"Neighbor lists: {neighbor_lists[:10]}")
    
    neighbors = []
    for i, neighbor in enumerate(neighbor_lists):
        neighbors.append(set(j for j in neighbor if j > i))  # j > i avoids duplicates

    triples = []

    for i in range(len(candidate_atom_coords)):
        for j in neighbors[i]:
            # k must be in neighbor list of both i and j
            common = neighbors[i].intersection(neighbors[j])
            for k in common:
                triples.append(
                    (
                        candidate_atom_indices[i],
                        candidate_atom_indices[j],
                        candidate_atom_indices[k],
                    )
                )

    if not triples:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array(triples, dtype=np.int64)

def find_atom_triples_for_three_hbonds(
    structure: Structure,
    hbond_length: float = 2.8,
    min_pair_dist: float = HBOND_PAIR_MIN_DIST,
    max_pair_dist: float = HBOND_PAIR_MAX_DIST,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """
    Find unique atom triples that can support a 3-H-bond water placement.

    Returns atom indices into the input structure, with one row per triple.
    """
    candidate_atom_indices, candidate_atom_coords = get_hbond_candidate_atom_data(
        structure
    )
    if len(candidate_atom_indices) < 3:
        return np.zeros((0, 3), dtype=np.int64)

    pair_distances = squareform(pdist(candidate_atom_coords))
    adjacency = (pair_distances > min_pair_dist) & (pair_distances < max_pair_dist)
    np.fill_diagonal(adjacency, False)

    triples = []
    num_candidates = len(candidate_atom_indices)
    for i in range(num_candidates - 2):
        for j in range(i + 1, num_candidates - 1):
            if not adjacency[i, j]:
                continue
            for k in range(j + 1, num_candidates):
                if not (adjacency[i, k] and adjacency[j, k]):
                    continue
                placed_water = place_water_from_atom_triple(
                    candidate_atom_coords[[i, j, k]],
                    hbond_length=hbond_length,
                    epsilon=epsilon,
                )
                if placed_water is None:
                    continue
                triples.append(
                    (
                        candidate_atom_indices[i],
                        candidate_atom_indices[j],
                        candidate_atom_indices[k],
                    )
                )

    if not triples:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array(triples, dtype=np.int64)



def impute_solvents_from_atom_triples(
    structure: Structure,
    one_solvent_per_chain: bool = True,
    hbond_length: float = 2.8,
    min_pair_dist: float = HBOND_PAIR_MIN_DIST,
    max_pair_dist: float = HBOND_PAIR_MAX_DIST,
    epsilon: float = 1e-4,
) -> Structure:
    """
    Place one water for every valid 3-atom triple in the input structure.

    This function assumes the caller has already prepared the input structure
    (for example, by stripping waters beforehand if desired).
    """
    # atom_triples = find_atom_triples_for_three_hbonds(
    #     structure,
    #     hbond_length=hbond_length,
    #     min_pair_dist=min_pair_dist,
    #     max_pair_dist=max_pair_dist,
    #     epsilon=epsilon,
    # )

    atom_triples = kdtree_find_atom_triples_for_three_hbonds(
        structure,
        hbond_length=hbond_length,
        min_pair_dist=min_pair_dist,
        max_pair_dist=max_pair_dist,
        epsilon=epsilon,
    )

    if len(atom_triples) == 0:
        return structure

    coords = structure.coords.copy()
    placed_water_coords = np.zeros(2 * len(atom_triples), dtype=coords.dtype)
    num_placed_waters = 0
    for atom_triple in atom_triples:
        new_coords = place_water_from_atom_triple(
            structure.coords["coords"][atom_triple],
            hbond_length=hbond_length,
            epsilon=epsilon,
        )
        if new_coords is None:
            continue
        for water_coords in new_coords:
            placed_water_coords[num_placed_waters]["coords"] = water_coords
            num_placed_waters += 1

    if num_placed_waters == 0:
        return structure

    return _append_imputed_solvents(
        structure,
        placed_water_coords[:num_placed_waters],
        one_solvent_per_chain=one_solvent_per_chain,
    )


def _append_imputed_solvents(
    structure: Structure,
    imputed_solvent_coords: np.ndarray,
    one_solvent_per_chain: bool = False,
) -> Structure:
    if len(imputed_solvent_coords) == 0:
        return structure

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
