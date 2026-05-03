
from pathlib import Path
import sys
import numpy as np
from boltzgen.data.data import Structure
from boltzgen.data.write.mmcif import to_mmcif
from imputation.recall import compare_structure_waters
from imputation.basic_helpers import resolve_npz_path
from imputation.impute_solvents_from_triples import *
from imputation.unused_solvent_filters import *
from imputation.filter_solvent_clashes import *



# NPZ_ROOT = Path("/data/rbg/shared/datasets/processed_rcsb/rcsb_solvents/structures")
# TRIPLE_HBOND_PAIR_MAX_DIST = 5.6
# COLLISION_MIN_DIST = 1.0


# def recall_result(PDB_ID):
#     npz_path = resolve_npz_path(PDB_ID, NPZ_ROOT)
#     gt_structure = Structure.load(npz_path)
#     gt_structure = gt_structure.to_one_solvent_per_chain(gt_structure)
#     gt_3hbonds_structure = gloria_remove_weak_solvents(
#         gt_structure,
#         min_hbonds=3,
#     )

#     gt_stripped_structure = gt_structure.remove_solvents()
#     imputed = impute_solvents_from_atom_triples(
#         gt_stripped_structure,
#         one_solvent_per_chain=True,
#         max_pair_dist=TRIPLE_HBOND_PAIR_MAX_DIST,
#     )
#     no_collisions = filter_solvent_clashes(
#         imputed,
#         min_dist=COLLISION_MIN_DIST,
#     )

#     result = compare_structure_waters(
#         gt_3hbonds_structure,
#         no_collisions,
#     )
#     return result


# print(recall_result("7zzh"))
