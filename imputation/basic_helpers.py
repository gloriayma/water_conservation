from pathlib import Path

from boltzgen.data.data import Structure
from gloria_hbond_helpers import gloria_remove_weak_solvents

NPZ_ROOT = Path("/data/rbg/shared/datasets/processed_rcsb/rcsb_solvents/structures")
NPZ_PATH = None  # Set to an explicit file path to override PDB_ID + NPZ_ROOT.
NOTEBOOK_ROOT = Path("/data/rbg/users/gloriama/dev/water_conservation")

def resolve_npz_path(pdb_id: str, npz_root: Path, npz_path: str | None = None) -> Path:
    if npz_path:
        return Path(npz_path)
    return Path(npz_root) / f"{pdb_id.lower()}.npz"


def raw_gt_structure(PDB_ID):
    npz_path = resolve_npz_path(PDB_ID, NPZ_ROOT, NPZ_PATH)
    gt_structure = Structure.load(npz_path)
    # print(f"Raw GT structure number of solvents: {gt_structure.count_solvents()}")
    return gt_structure

def stripped_gt_structure(PDB_ID):
    gt_structure = raw_gt_structure(PDB_ID)
    gt_structure = gt_structure.to_one_solvent_per_chain(gt_structure)
    stripped_gt = gt_structure.remove_solvents()
    assert stripped_gt.count_solvents() == 0
    return stripped_gt


def count_residues(structure: Structure) -> int:
    return len(structure.residues)


def count_stripped_gt_residues(PDB_ID) -> int:
    return count_residues(stripped_gt_structure(PDB_ID))

def filtered_gt_structure(PDB_ID, min_hbonds):
    gt_structure = raw_gt_structure(PDB_ID)
    gt_structure = gt_structure.to_one_solvent_per_chain(gt_structure)
    filtered_gt = gloria_remove_weak_solvents(gt_structure, min_hbonds=min_hbonds)
    print(f"Filtered (>={min_hbonds} H-bonds) GT structure number of solvents: {filtered_gt.count_solvents()}")
    return filtered_gt