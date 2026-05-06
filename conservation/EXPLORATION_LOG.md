# Exploration Log — Water Conservation Pipeline

Extensive notes on what was tried, what was found, and what didn't work.
Separate from RESULTS.md which only records clean final numbers.

---

## Session 1 — 2026-05-04

### Data inspection

**NPZ schema (processed RCSB files at `/data/rbg/shared/datasets/processed_rcsb/rcsb_solvents/`):**

These are Boltz/BoltzGen-style preprocessed structures. The config (`config.yaml`) confirms
`keep_waters: 1000` and `one_solvent_per_chain: false`, so all waters are preserved.

The manifest (`manifest.json`) is a list of 238,857 entries. Each entry has:
- `id` (lowercase PDB ID)
- `structure.resolution` (float or null)
- `structure.method` (string, e.g. "x-ray diffraction")
- `chains` list with `mol_type` (0=polymer, 3=ligand, 4=water), `num_residues`, `cluster_id`

The `records/*.json` files mirror this metadata but don't add crystal info.

**NPZ arrays per structure:**
```
atoms:     (N_atoms,) structured: name, coords (3,), is_present, bfactor, plddt
residues:  (N_res,)   structured: name, res_type, res_idx, atom_idx, atom_num, ...
chains:    (N_chains,) structured: name, mol_type, entity_id, sym_id, asym_id,
                                   atom_idx, atom_num, res_idx, res_num, ...
coords:    (N_atoms,) structured: coords (3,)   [alternate coord store, same content]
bonds, interfaces, mask, ensemble: not needed for this pipeline
```

**What's present:**
- ✅ Water oxygen coordinates — HOH residues in mol_type=4 chains, atom name 'O'
- ✅ B-factors — `atoms['bfactor']` for every atom including waters
- ✅ Method and resolution metadata — in manifest.json

**What's missing (important for later steps):**
- ❌ Occupancy — not stored in npz. Alt conformations were collapsed during preprocessing.
- ❌ Alt-loc identifiers — not stored. Same consequence.
- ❌ Unit cell / space group (CRYST1) — not in npz OR in JSON records.

**Implication for crystal-contact filtering (step 7):** We cannot do crystal-contact
filtering from the npz files alone. This step will require loading original mmCIF/PDB files
from the source PDB mirror (`/data/rbg/shared/datasets/pdb` per config.yaml). This is noted
as a future dependency — not blocking steps 1–6.

**Implication for alt-loc handling:** Since alt-locs are collapsed at preprocessing time,
we cannot explicitly filter them. Any multi-conformer waters would have been reduced to a
single position (presumably the highest occupancy one). This is an approximation.

---

### Manifest method breakdown (238,857 entries)
| Method | Count |
|--------|-------|
| X-ray diffraction | 195,210 |
| Electron microscopy | 28,417 |
| Solution NMR | 14,273 |
| Electron crystallography | 273 |
| Others | <500 total |

---

### Resolution distribution for X-ray entries
- Total X-ray: 195,210
- Resolution range: 0.00 – 15.00 Å (some clearly erroneous/placeholder zeros)
- Median: 2.00 Å
- ≤2.0 Å: 98,063 entries
- ≤2.5 Å: 153,286 entries

---

### Water availability
- Entries with water chains in dataset: 183,217 out of 238,857 (77%)
- X-ray + ≤2.0 Å + waters: **97,838** entries
- X-ray + ≤2.5 Å + waters: **152,340** entries
- This is consistent with Chen et al. 2024 (~22% deposit no waters)

---

### Cluster file

File: `clusters-by-entity-100.txt` (RCSB 100% sequence identity clusters)
- 1,049,376 cluster lines
- 1,597,265 total `PDBID_entityID` tokens
- Note: the cluster file uses uppercase PDB IDs with entity numbers (e.g., `1ABC_1`).
  The npz files use lowercase. The mapping is straightforward (lowercase the PDB ID).
- Note: entity IDs in RCSB = one entry per unique molecular entity in a structure.
  A homo-oligomer has one entity for both chains, so `1ABC_1` appears once per cluster,
  not once per chain. Deduplication to unique PDB IDs is needed but trivial.
- Note: ~72% of cluster tokens (1,142,259) are not in the processed npz dataset.
  This is because the npz dataset is filtered (MinimumLengthFilter, UnknownFilter, etc.),
  and because the RCSB cluster file covers all of PDB while the npz dataset is a subset.

---

### Cluster filtering results

See `filter_stats_2.0A.json` and `filter_stats_2.5A.json` for exact numbers.

**Filtering funnel at 2.0 Å:**
```
Total clusters:                 1,049,376
  Not in npz dataset:          -1,142,259 (entity-tokens, not unique PDBs)
  Not X-ray:                     -200,377
  Resolution > 2.0 Å:            -141,153
  No waters:                         -208
Clusters with >=1 qualifying PDB:  46,646
Clusters with >=5 qualifying PDbs:  2,894
```

**Filtering funnel at 2.5 Å:**
```
Clusters with >=1 qualifying PDB:  71,378
Clusters with >=5 qualifying PDBs:  5,230
```

---

### Decisions and unknowns noted

- The 1,142,259 "not in manifest" drops are expected — the npz pipeline applies several
  additional filters (chain length, clash detection, etc.) on top of basic PDB coverage.
- Only 208 (2.0A) / 1,135 (2.5A) entries were dropped for having no waters after passing
  the X-ray + resolution filter. This is very low; the `has_waters` flag in the manifest
  is reliable.
- The resolution cutoff decision (2.0 vs 2.5 Å) has a large effect on the number of
  usable clusters: 2,894 vs 5,230. Both are substantial enough to proceed.
- The large "not in npz dataset" drop rate suggests we may want to eventually go back to
  raw PDB files for broader coverage, but for now the npz files are sufficient.

---

### Scripts written

- `scripts/npz_io.py` — loading npz files, extracting water coords and B-factors,
  getting C-alpha coords for alignment
- `scripts/cluster_filter.py` — parsing cluster file, manifest lookup, filtering

### Notebooks written

- `notebooks/01_schema_exploration.ipynb` — npz schema survey
- `notebooks/02_cluster_statistics.ipynb` — cluster filtering and statistics

---

### What was NOT tried / future work

- Did not try loading original PDB/mmCIF files (needed for crystal contact filtering)
- Did not investigate the `plddt` field in atoms — this is a per-atom pLDDT score from
  AlphaFold, not a B-factor; its meaning for crystal structures is unclear (likely 1.0 for
  all real crystal atoms)
- Did not check whether waters with `is_present=False` exist (could indicate density-absent
  atoms in multi-model ensembles)

---

## Session 2 — 2026-05-05

### Motivation: local cropping for alignment

Global CA superposition (session 1, notebook 03) aligns the entire chain as one rigid body.
For large proteins this is misleading: even with a low global RMSD, residues far from the
alignment core can have substantial positional error due to accumulated rotation/translation
uncertainty. The solution is to define canonical **local crops** on a reference structure and
align each crop independently across cluster members.

### Key discovery: `residues["atom_center"]`

The `residues` structured array has two fields not captured in the session 1 schema summary:
- `atom_center` (int32): global index into `atoms` for the canonical representative atom of
  this residue. For amino acids this is CA, for waters it is the O atom. For ligands it
  appears to be the first heavy atom.
- `atom_disto` (int32): index into `atoms` for the "distal" atom (CB for amino acids). Not
  used in this work yet.

Using `atoms[res["atom_center"]]["coords"]` is therefore the correct way to get a residue
center — no centroid computation needed. This is consistent with how Boltz/BoltzGen uses
these fields internally for distance-matrix features.

Both `is_present` fields need to be checked: `residue["is_present"]` guards the residue
itself, and `atoms[res["atom_center"]]["is_present"]` guards the center atom.

### Greedy sphere-covering algorithm

Implemented in `scripts/crop.py`. Key design choices:

- **Candidate points**: all residue canonical centers (polymer CA + water O + ligand center)
  that are marked present.
- **Covering radius**: 20 Å (configurable). Large enough that a typical 150-residue protein
  gets 2–6 crops; a 1,000-residue protein gets ~20–30.
- **Seed ordering**: array order (no optimization for minimum number of crops). This is
  deterministic and reproducible, which matters for crop identity across cluster members.
- **Coverage marking**: once a residue is covered (added to any crop), it cannot become
  a seed. Crops may overlap — a residue within 20 Å of two seeds appears in both crops.
  This is intentional (overlap at boundaries provides context for local alignment).
- **Water-only warning**: any crop with no polymer/ligand residues prints a warning. This
  flags isolated water clusters far from the protein core (e.g., disordered solvent patches
  or crystal-contact waters not near any protein atom). In practice seen rarely.

### Demo results (6crk — first 5-member cluster at 2.0 Å, 1030-residue representative)

- Structure: 982 non-water residues, 627 waters, 28 crops at radius=20 Å
- Crop sizes range from 53 to 298 total residues; median ~150
- No water-only or isolated warnings (all isolated waters are within 20 Å of at least one
  polymer atom in this structure)
- Sums across crops: 2,665 non-water (2.7× the 982 in the structure), 1,642 waters (2.6×)
  — confirms overlapping crops

### mmCIF export

Used `Structure.extract_residues(struct, indices)` + `to_mmcif(cropped)` from BoltzGen.
`extract_residues` is a method on the `Structure` class (called as
`Structure.extract_residues(struct, indices)`, not as a standalone function). It handles
all reindexing of `atom_idx`, `res_idx`, chain atom/residue ranges correctly.

### Scripts/notebooks written

- `scripts/crop.py` — `count_nonsolvent_residues`, `get_residue_centers`, `greedy_crop`,
  `save_crop_as_mmcif`
- `notebooks/04_crop_investigation.ipynb` — demo on cluster `['5kdo','6crk','6rmv','8qeg','8qeh']`

### What was NOT tried

- Did not try varying the radius (20 Å is untested; should validate that the local
  alignment within each crop is actually better than global).
- Did not implement the local alignment step (using the crop definitions). That is the
  next step.
- Did not check whether crop identity is stable across cluster members (i.e., whether
  the same seed ordering produces spatially consistent crops in structurally similar
  proteins). This needs validation before water pooling can use crops.

---

## Session 3 — 2026-05-05

### Changes to greedy_crop: water residues excluded as seeds

Changed `greedy_crop` so that only non-water (polymer + ligand) residues can serve as
seeds. Water residues can still be captured by a nearby non-water seed sphere, but they
will never anchor a crop on their own. This removes the previous "water-only crop"
warning entirely (all crops now guaranteed to have at least one non-solvent residue).
Crop count for 6crk dropped from 28 to 26 as a result.

### crop_sample: matching non-solvent residues across cluster members

New function `crop_sample(ref_struct, ref_crop, sample_struct, water_radius)`.

**Non-solvent matching**: by `(entity_id, sym_id, local 0-based position in chain)`.
For same-sequence 100%-identity clusters this is exact. Residues absent in the
sample (missing density, shorter chain) are returned as `None` in `sample_ns_indices`.
The ref/sample index lists are parallel — `ref_ns_indices[i]` corresponds to
`sample_ns_indices[i]` — which is the correspondence needed for Kabsch alignment.

**Water inclusion**: all present waters in the sample within `water_radius` of the
sample's seed center atom (the CA of the residue corresponding to the reference seed).
Falls back to the mean of all matched non-solvent centers if the seed residue is absent.

**TODO to explore**: alternative water inclusion — all waters within a smaller radius
(e.g. 4-5 A) of ANY polymer/ligand atom in the crop. Would give denser local
hydration shells and might be more biologically meaningful.

### Demo results on cluster ['5kdo','6crk','6rmv','8qeg','8qeh']

Reference: 6crk (most non-solvent residues). 26 crops x 5 samples = 130 total;
118 saved, 12 skipped (0 residues after matching).

Key observations from the stats table:
- Crops 0-19: most samples have good coverage (matched >= 80% of ref non-solvent res)
- Crops 20-25: unique to 6crk or nearly so. Crops 21, 22, 24 have 0 matched residues
  in 4/5 samples — likely regions of 6crk absent from the shorter structures.
  6rmv (429 non-solvent res) consistently has fewer matches, as expected.
- Fallbacks: 4-6 crops per sample use the mean-center fallback for water inclusion,
  meaning the reference seed residue is absent in those samples.

### Scripts/notebooks written

- `scripts/crop.py` updated: `_build_res_chain_map`, `crop_sample`, revised `greedy_crop`
- `notebooks/05_cluster_crops.ipynb`: full cluster-level crop demo and mmCIF export
