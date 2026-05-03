from __future__ import annotations

import numpy as np

from boltzgen.data import const
from boltzgen.data.data import Structure
from gloria_hbond_helpers import gloria_get_solvent_hbond_counts_and_mask

def gloria_remove_low_b_factor_solvents(
    structure: Structure,
    threshold=None,
    quantile=None,
    n_keep=None,
) -> Structure:
    """Remove solvents with low B-factors using the main-codebase logic."""
    assert sum(x is not None for x in [threshold, quantile, n_keep]) == 1, (
        "Exactly one of threshold, quantile, or n_keep should be set"
    )

    mask = structure.mask.copy()

    solvent_chain_indices = []
    solvent_bfactors = []
    for chain_idx, chain in enumerate(structure.chains):
        if chain["mol_type"] in (
            const.chain_type_ids["SOLVENT"],
            const.chain_type_ids["iSOLVENT"],
        ):
            solvent_chain_indices.append(chain_idx)
            solvent_bfactors.append(structure.atoms[chain["atom_idx"]]["bfactor"])

    solvent_bfactors = np.array(solvent_bfactors)
    solvent_chain_indices = np.array(solvent_chain_indices)

    if quantile is not None:
        threshold = np.quantile(solvent_bfactors, quantile)
    elif n_keep is not None:
        if n_keep >= len(solvent_bfactors):
            threshold = -np.inf
        else:
            threshold = np.partition(solvent_bfactors, -n_keep)[-n_keep]

    for chain_idx, bfactor in zip(solvent_chain_indices, solvent_bfactors):
        if bfactor < threshold:
            mask[chain_idx] = False

    return rebuild_structure_with_mask(structure, mask)

