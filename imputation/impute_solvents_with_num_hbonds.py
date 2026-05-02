from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from boltzgen.data import const
from boltzgen.data.data import Chain, Structure
from new_impute_solvent import new_impute_solvent


def impute_solvents_with_num_hbonds(
    structure: Structure,
    one_solvent_per_chain: bool = False,
    min_solvents: int = 0,
    interaction_min_dist: float = 2.5,
    max_sample_attempts: int = 100,
    allow_overlap_with_real_solvents: bool = False,
) -> Structure:
    """Editable copy of `Structure.impute_solvents()` for notebook experiments."""
    num_solvents = 0
    entities = set()
    solvent_entities = set()
    solvent_chain_mask = (
        (structure.chains["mol_type"] == const.chain_type_ids["SOLVENT"])
        | (structure.chains["mol_type"] == const.chain_type_ids["iSOLVENT"])
    )
    for chain in structure.chains:
        if (
            chain["mol_type"] == const.chain_type_ids["SOLVENT"]
            or chain["mol_type"] == const.chain_type_ids["iSOLVENT"]
        ):
            num_solvents += chain["res_num"]
            solvent_entities.add(chain["entity_id"])
        entities.add(chain["entity_id"])
    assert len(solvent_entities) <= 1, (
        "Currently only one solvent entity id is supported for imputation."
    )

    if num_solvents >= min_solvents:
        return structure

    chains = structure.chains.copy()
    coords = structure.coords.copy()
    atoms = structure.atoms.copy()
    residues = structure.residues.copy()
    mask = structure.mask.copy()

    num_to_impute = min_solvents - num_solvents
    solvent_chain_mask = (
        (structure.chains["mol_type"] == const.chain_type_ids["SOLVENT"])
        | (structure.chains["mol_type"] == const.chain_type_ids["iSOLVENT"])
    )

    if len(solvent_entities) == 0:
        solvent_entity_id = max(entities) + 1
        next_sym_id = 0
    else:
        solvent_entity_id = solvent_entities.pop()
        next_sym_id = int(structure.chains[solvent_chain_mask]["sym_id"].max()) + 1

    imputed_solvent_coords = np.zeros(num_to_impute, dtype=coords.dtype)
    imputed_solvent_atoms = np.zeros(num_to_impute, dtype=atoms.dtype)
    imputed_solvent_residues = np.zeros(num_to_impute, dtype=residues.dtype)

    hbond_candidate_mask = (
        np.char.startswith(atoms["name"], ("N"))
        | np.char.startswith(atoms["name"], ("O"))
        | np.char.startswith(atoms["name"], ("F"))
        | np.char.startswith(atoms["name"], ("S"))
    )
    hbond_candidate_mask &= atoms["is_present"]
    hbond_candidate_atoms = atoms[hbond_candidate_mask]
    hbond_candidate_pair_distances = squareform(
        pdist(coords[hbond_candidate_mask]["coords"])
    )
    hbond_candidate_pairs = np.where(
        (hbond_candidate_pair_distances > 3.13)
        & (hbond_candidate_pair_distances < 5.09)
    )
    n_function = lambda s: max((60 - s) // 20, 1)

    no_gt_solvent_coords = coords["coords"]
    if allow_overlap_with_real_solvents:
        real_solvent_atom_mask = np.zeros(len(atoms), dtype=bool)
        for chain in structure.chains:
            if chain["mol_type"] == const.chain_type_ids["SOLVENT"]:
                atom_start = chain["atom_idx"]
                atom_end = atom_start + chain["atom_num"]
                real_solvent_atom_mask[atom_start:atom_end] = True
        no_gt_solvent_coords = no_gt_solvent_coords[~real_solvent_atom_mask]

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

    for impute_idx in range(num_to_impute):
        sample_attempts = 0
        while True:
            sample_attempts += 1

            try:
                new_coords = new_impute_solvent(
                    hbond_candidate_atoms,
                    hbond_candidate_pairs,
                    # n_function(sample_attempts),
                    3,
                )
                # print(n_function(sample_attempts))
            except Exception:
                new_coords = None
            if new_coords is None:
                continue

            current_coords = np.concatenate(
                [no_gt_solvent_coords, imputed_solvent_coords[:impute_idx]["coords"]]
            )
            if len(current_coords) == 0:
                closest_contact = np.inf
            else:
                distances = np.linalg.norm(current_coords - new_coords, axis=1)
                closest_contact = np.min(distances)
            if (
                closest_contact > interaction_min_dist
                or sample_attempts > max_sample_attempts
            ):
                break
        print(f"{sample_attempts=} until a {n_function(sample_attempts)=} h-bond thing was found, for the {impute_idx=}th water.")
        imputed_solvent_coords[impute_idx]["coords"] = new_coords
        imputed_solvent_atoms[impute_idx] = ("O", new_coords, True, 50.0, 1.0)
        if one_solvent_per_chain:
            atom_idx = chains[-1]["atom_idx"] + chains[-1]["atom_num"]
            res_idx = len(residues) + impute_idx
            imaginary_solvent_chain = np.array(
                [
                    (
                        f"Wa{num_solvents + impute_idx}",
                        const.chain_type_ids["iSOLVENT"],
                        solvent_entity_id,
                        next_sym_id + impute_idx,
                        len(chains),
                        atom_idx,
                        1,
                        res_idx,
                        1,
                        0,
                    )
                ],
                dtype=Chain,
            )
            chains = np.append(chains, imaginary_solvent_chain)
            solvent_atom_idx = atom_idx
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
    coords = np.append(coords, imputed_solvent_coords)
    atoms = np.append(atoms, imputed_solvent_atoms)
    residues = np.append(residues, imputed_solvent_residues)
    if one_solvent_per_chain:
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
