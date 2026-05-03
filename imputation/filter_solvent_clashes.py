from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from boltzgen.data import const
from boltzgen.data.data import Structure


# def _rebuild_structure_with_mask(
#     structure: Structure,
#     mask: np.ndarray,
# ) -> Structure:
#     filtered_structure = Structure(
#         atoms=structure.atoms,
#         bonds=structure.bonds,
#         residues=structure.residues,
#         chains=structure.chains,
#         interfaces=structure.interfaces,
#         mask=mask,
#         coords=structure.coords,
#         ensemble=structure.ensemble,
#     )
#     return filtered_structure.remove_invalid_chains()


# def filter_solvent_clashes(
#     structure: Structure,
#     min_dist: float = 2.5,
# ) -> Structure:
#     """
#     Remove solvent molecules whose oxygen clashes within ``min_dist``.

#     This is a two-stage post-processing filter:
#     1. Drop solvent molecules whose oxygen clashes with any non-solvent atom.
#     2. Among the remaining solvents, greedily keep a non-clashing subset in
#        chain order.
#     """
#     structure = structure.to_one_solvent_per_chain(structure)
#     mask = structure.mask.copy()
#     atom_coords = structure.coords["coords"]
#     atom_present = structure.atoms["is_present"]

#     solvent_chain_indices = []
#     solvent_atom_indices = []
#     solvent_atom_mask = np.zeros(len(structure.atoms), dtype=bool)
#     for chain_idx, chain in enumerate(structure.chains):
#         if chain["mol_type"] not in (
#             const.chain_type_ids["SOLVENT"],
#             const.chain_type_ids["iSOLVENT"],
#         ):
#             continue

#         atom_start = chain["atom_idx"]
#         atom_end = atom_start + chain["atom_num"]
#         solvent_atom_mask[atom_start:atom_end] = True

#         present_atom_offsets = np.flatnonzero(atom_present[atom_start:atom_end])
#         if len(present_atom_offsets) == 0:
#             mask[chain_idx] = False
#             continue

#         solvent_chain_indices.append(chain_idx)
#         solvent_atom_indices.append(atom_start + present_atom_offsets[0])

#     if not solvent_chain_indices:
#         return structure

#     solvent_chain_indices = np.array(solvent_chain_indices, dtype=np.int64)
#     solvent_atom_indices = np.array(solvent_atom_indices, dtype=np.int64)
#     solvent_coords = atom_coords[solvent_atom_indices]
#     nonsolvent_atom_mask = (~solvent_atom_mask) & atom_present
#     nonsolvent_coords = atom_coords[nonsolvent_atom_mask]
#     clash_radius = np.nextafter(min_dist, 0.0)

#     if len(nonsolvent_coords) > 0:
#         fixed_atom_kdtree = KDTree(nonsolvent_coords)
#         fixed_atom_hits = fixed_atom_kdtree.query_ball_point(solvent_coords, r=clash_radius)
#         clashes_with_nonsolvent = np.array(
#             [len(hit_indices) > 0 for hit_indices in fixed_atom_hits],
#             dtype=bool,
#         )
#         mask[solvent_chain_indices[clashes_with_nonsolvent]] = False
#     else:
#         clashes_with_nonsolvent = np.zeros(len(solvent_chain_indices), dtype=bool)

#     surviving_chain_indices = solvent_chain_indices[~clashes_with_nonsolvent]
#     surviving_coords = solvent_coords[~clashes_with_nonsolvent]
#     if len(surviving_chain_indices) == 0:
#         return _rebuild_structure_with_mask(structure, mask)

#     proposed_water_kdtree = KDTree(surviving_coords)
#     clash_neighbors = proposed_water_kdtree.query_ball_tree(
#         proposed_water_kdtree,
#         r=clash_radius,
#     )

#     blocked = np.zeros(len(surviving_chain_indices), dtype=bool)
#     for i, neighbor_indices in enumerate(clash_neighbors):
#         if blocked[i]:
#             mask[surviving_chain_indices[i]] = False
#             continue

#         for j in neighbor_indices:
#             if j <= i:
#                 continue
#             blocked[j] = True
#             mask[surviving_chain_indices[j]] = False

#     return _rebuild_structure_with_mask(structure, mask)
