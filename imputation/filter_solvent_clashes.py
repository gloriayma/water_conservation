from __future__ import annotations

import numpy as np

from boltzgen.data import const
from boltzgen.data.data import Structure


def _rebuild_structure_with_mask(
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


def filter_solvent_clashes(
    structure: Structure,
    min_dist: float = 2.5,
) -> Structure:
    """
    Remove solvent molecules whose oxygen clashes within ``min_dist``.

    This is a greedy post-processing filter: solvent molecules are visited in
    chain order, and later solvents are dropped if they clash with any
    non-solvent atom or with an already-kept solvent oxygen.
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

    nonsolvent_atom_mask = (~solvent_atom_mask) & atom_present
    nonsolvent_coords = atom_coords[nonsolvent_atom_mask]

    kept_solvent_coords = []
    for chain_idx, atom_idx in zip(solvent_chain_indices, solvent_atom_indices):
        solvent_coord = atom_coords[atom_idx]

        clashes_with_nonsolvent = (
            len(nonsolvent_coords) > 0
            and np.any(np.linalg.norm(nonsolvent_coords - solvent_coord, axis=1) < min_dist)
        )
        clashes_with_kept_solvent = (
            len(kept_solvent_coords) > 0
            and np.any(
                np.linalg.norm(np.asarray(kept_solvent_coords) - solvent_coord, axis=1)
                < min_dist
            )
        )

        if clashes_with_nonsolvent or clashes_with_kept_solvent:
            mask[chain_idx] = False
            continue

        kept_solvent_coords.append(solvent_coord)

    return _rebuild_structure_with_mask(structure, mask)
