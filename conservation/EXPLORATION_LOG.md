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
