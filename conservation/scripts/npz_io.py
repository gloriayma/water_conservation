"""
Utilities for loading and extracting data from the processed RCSB npz files.

Each npz file contains:
  atoms:    structured array with fields (name, coords, is_present, bfactor, plddt)
  residues: structured array with fields (name, res_type, res_idx, atom_idx, atom_num, ...)
  chains:   structured array with fields (name, mol_type, entity_id, sym_id, asym_id,
                                          atom_idx, atom_num, res_idx, res_num, ...)
  coords:   alternate coords array (same shape as atoms)
  bonds, interfaces, mask, ensemble: not used here

mol_type codes:
  0 = polymer (protein/nucleic acid)
  3 = small molecule ligand
  4 = water

Missing from schema (noted for downstream decisions):
  - occupancy: not stored
  - alt_loc: not stored
  - unit cell / space group: not stored in npz or json records
"""

import numpy as np
from pathlib import Path

DATA_DIR = Path("/data/rbg/shared/datasets/processed_rcsb/rcsb_solvents")
STRUCTURES_DIR = DATA_DIR / "structures"

MOL_TYPE_POLYMER = 0
MOL_TYPE_LIGAND = 3
MOL_TYPE_WATER = 4


def load_npz(pdb_id: str) -> dict:
    """Load a structure npz by lowercase PDB ID. Returns dict of arrays."""
    path = STRUCTURES_DIR / f"{pdb_id.lower()}.npz"
    return np.load(path, allow_pickle=True)


def get_water_coords(npz: dict) -> np.ndarray:
    """
    Extract Cartesian coordinates of all water oxygen atoms.

    Returns shape (N, 3) float32 array. Returns empty array if no waters.
    Waters are stored as a single chain with mol_type=4; each residue is one HOH.
    The atom named 'O' in each water residue is the oxygen.
    """
    chains = npz["chains"]
    residues = npz["residues"]
    atoms = npz["atoms"]

    water_chain_mask = chains["mol_type"] == MOL_TYPE_WATER
    if not water_chain_mask.any():
        return np.empty((0, 3), dtype=np.float32)

    water_chains = chains[water_chain_mask]
    coords = []
    for chain in water_chains:
        a_start = chain["atom_idx"]
        a_end = a_start + chain["atom_num"]
        chain_atoms = atoms[a_start:a_end]
        # keep only O atoms that are marked present
        oxy_mask = (chain_atoms["name"] == "O") & chain_atoms["is_present"]
        for atom in chain_atoms[oxy_mask]:
            coords.append(atom["coords"])

    return np.array(coords, dtype=np.float32) if coords else np.empty((0, 3), dtype=np.float32)


def get_water_bfactors(npz: dict) -> np.ndarray:
    """
    Extract B-factors for water oxygen atoms (same ordering as get_water_coords).

    Returns shape (N,) float32 array.
    """
    chains = npz["chains"]
    atoms = npz["atoms"]

    water_chain_mask = chains["mol_type"] == MOL_TYPE_WATER
    if not water_chain_mask.any():
        return np.empty((0,), dtype=np.float32)

    water_chains = chains[water_chain_mask]
    bfactors = []
    for chain in water_chains:
        a_start = chain["atom_idx"]
        a_end = a_start + chain["atom_num"]
        chain_atoms = atoms[a_start:a_end]
        oxy_mask = (chain_atoms["name"] == "O") & chain_atoms["is_present"]
        bfactors.extend(chain_atoms["bfactor"][oxy_mask].tolist())

    return np.array(bfactors, dtype=np.float32)


def get_polymer_ca_coords(npz: dict, chain_idx: int = 0) -> np.ndarray:
    """
    Extract C-alpha coordinates for a given polymer chain (by chain array index).

    Returns shape (N, 3) float32 array. Used for structural alignment.
    """
    chains = npz["chains"]
    atoms = npz["atoms"]

    chain = chains[chain_idx]
    assert chain["mol_type"] == MOL_TYPE_POLYMER, f"Chain {chain_idx} is not a polymer"

    a_start = chain["atom_idx"]
    a_end = a_start + chain["atom_num"]
    chain_atoms = atoms[a_start:a_end]

    ca_mask = (chain_atoms["name"] == "CA") & chain_atoms["is_present"]
    return chain_atoms["coords"][ca_mask].astype(np.float32)


def get_first_valid_polymer_chain_idx(npz: dict) -> int | None:
    """
    Return the index (into chains array) of the first valid polymer chain.
    Returns None if no valid polymer chain exists.
    """
    chains = npz["chains"]
    for i, chain in enumerate(chains):
        if chain["mol_type"] == MOL_TYPE_POLYMER:
            return i
    return None


def count_waters(npz: dict) -> int:
    """Count the number of water residues in a structure."""
    chains = npz["chains"]
    water_chain_mask = chains["mol_type"] == MOL_TYPE_WATER
    if not water_chain_mask.any():
        return 0
    return int(chains["res_num"][water_chain_mask].sum())