from __future__ import annotations

import numpy as np

from boltzgen.data import const
from boltzgen.data.data import Structure




def gloria_get_solvent_hbond_counts_and_mask(
    structure: Structure,
) -> tuple[np.ndarray, np.ndarray]:
    """Count solvent H-bonds and track which solvent chains are present."""
    solvent_chain_mask = np.zeros(len(structure.chains), dtype=bool)
    num_hbonds = np.zeros(len(structure.chains), dtype=np.int64)
    solvent_atom_mask = np.zeros(len(structure.atoms), dtype=bool)
    solvent_atom_indices = []

    for chain_idx, chain in enumerate(structure.chains):
        if chain["mol_type"] in (
            const.chain_type_ids["SOLVENT"],
            const.chain_type_ids["iSOLVENT"],
        ):
            atom_start = chain["atom_idx"]
            atom_end = atom_start + chain["atom_num"]
            solvent_atom_mask[atom_start:atom_end] = True

            present_atom_offsets = np.flatnonzero(
                structure.atoms["is_present"][atom_start:atom_end]
            )
            # In some cases solvent molecules are not present.
            if len(present_atom_offsets) > 0:
                solvent_chain_mask[chain_idx] = True
                solvent_atom_indices.append(atom_start + present_atom_offsets[0])

    if solvent_chain_mask.sum() == 0:
        return num_hbonds, solvent_chain_mask

    other_atom_mask = ~solvent_atom_mask
    atom_to_res = np.zeros((len(structure.atoms), len(structure.residues)), dtype=bool)
    for res_idx, res in enumerate(structure.residues):
        atom_start = res["atom_idx"]
        atom_end = atom_start + res["atom_num"]
        atom_to_res[atom_start:atom_end, res_idx] = True
    solvent_coords = structure.atoms["coords"][solvent_atom_indices]

    # None of the solvent chains are actually present.
    if len(solvent_coords) == 0:
        return num_hbonds, solvent_chain_mask

    hbond_candidate_mask = (
        (
            np.char.startswith(structure.atoms["name"], ("N"))
            | np.char.startswith(structure.atoms["name"], ("O"))
            | np.char.startswith(structure.atoms["name"], ("F"))
        )
        & structure.atoms["is_present"]
        & other_atom_mask
    )

    hbond_candidate_atom_coords = structure.atoms["coords"][hbond_candidate_mask]
    hbond_candidate_atom_to_res = atom_to_res[hbond_candidate_mask]

    diff = solvent_coords[:, None, ...] - hbond_candidate_atom_coords[None, ...]
    distances = np.linalg.norm(diff, axis=-1)
    is_hbonds = (distances < const.hbond_max_dist).astype(float) * (
        distances > const.hbond_min_dist
    ).astype(float)
    is_hbonds_res = (
        (is_hbonds @ hbond_candidate_atom_to_res.astype(float)).astype(bool).astype(float)
    )
    num_hbonds[solvent_chain_mask] = is_hbonds_res.sum(axis=1).astype(np.int64)

    return num_hbonds, solvent_chain_mask


def gloria_get_solvent_hbond_counts(structure: Structure) -> np.ndarray:
    """Count H-bonds for each present solvent chain."""
    num_hbonds, _ = gloria_get_solvent_hbond_counts_and_mask(structure)
    return num_hbonds


def gloria_get_solvent_hbond_mask(
    structure: Structure,
    min_hbonds: int = 2,
) -> np.ndarray:
    """Return a chain-level mask for present solvent chains above a threshold."""
    num_hbonds, solvent_chain_mask = gloria_get_solvent_hbond_counts_and_mask(structure)
    keep_mask = np.zeros(len(structure.chains), dtype=bool)
    keep_mask[solvent_chain_mask] = num_hbonds[solvent_chain_mask] >= min_hbonds
    return keep_mask



def gloria_remove_weak_solvents(
    structure: Structure,
    min_hbonds: int = 2,
) -> Structure:
    """Remove solvents that have fewer than ``min_hbonds`` H-bonds."""
    mask = structure.mask.copy()
    num_hbonds, solvent_chain_mask = gloria_get_solvent_hbond_counts_and_mask(structure)
    mask[solvent_chain_mask] = num_hbonds[solvent_chain_mask] >= min_hbonds
    return rebuild_structure_with_mask(structure, mask)


def rebuild_structure_with_mask(
    structure: Structure,
    mask: np.ndarray,
) -> Structure:
    filtered_structure = Structure(
        atoms=structure.atoms,
        bonds=structure.bonds,
        residues=structure.residues,
        chains=structure.chains,
        interfaces=structure.interfaces,
        mask=mask,
        coords=structure.coords,
        ensemble=structure.ensemble,
    )
    return filtered_structure.remove_invalid_chains()

