# Water Conservation Annotation Project

## Goal

Build a PDB-wide statistical resource that answers two questions:

1. For each polymer entity in the PDB, how many independent structural samples
   exist (i.e. how many PDB entries deposit the same sequence, differently
   solvated)?
2. After clustering same-sequence entries and pooling their water positions,
   what fraction of the waters in any given structure recur across the
   ensemble — i.e. what is each water's "conservation score"?

The deliverable is a flat annotation file:
`(pdb_id, chain, water_serial) → (conservation_score, cluster_size, contributing_entries)`,
plus the cluster-size and conservation-score distributions.

The motivation is to put a number on the question "how meaningful are
crystallographic waters?" — currently only answered per-family in the
literature, never PDB-wide.

## Status

In setup. No pipeline code written yet. Validation case not yet run.

## Data

Input: locally-mirrored `.npz` files, one per PDB entry, on this cluster.
These are a custom Boltz/BoltzGen-style preprocessing that — unlike the
upstream Boltz pipeline — preserves waters. **Schema not yet confirmed.**
First task in the first real session is to inspect one `.npz` and verify
what's actually in it. They are housed at /data/rbg/shared/datasets/processed_rcsb/rcsb_solvents. 

Critical fields to check for in the schema:
- water atom records (coordinates, residue type HOH/WAT)
- B-factors and occupancies on every atom (needed for quality filtering)
- alt loc identifiers (needed to deduplicate alternate conformations)
- unit cell + space group information from CRYST1
  (needed to identify crystal-contact waters via symmetry mate expansion)

If any of these are missing from the `.npz`, we fall back to the raw mmCIFs
on disk. Do not proceed past schema confirmation without resolving this.

## Method

Per same-sequence cluster, the recipe is:

1. (Completed) Pull RCSB's precomputed sequence-cluster file at **100% identity**
   (`clusters-by-entity-100.txt`). 95% is too loose — point mutations in the
   binding pocket would invalidate the conservation comparison.
2. For each cluster: filter entries to X-ray ≤ 2.0 Å (or ≤ 2.5 Å, try that too and record statistics) and
   drop entries with no deposited waters. 
2. Filter clusters to ≥5 members.
4. Take one representative chain per entry to avoid NCS double-counting.
5. Structurally align all entries onto the highest-resolution member.
6. Filter waters by quality (Carugo mobility ≥ 2.0 OR normalized B ≥ 1.0;
   PyWATER convention).
7. Filter waters near crystal contacts (within ~5 Å of a symmetry mate but
   not within ~5 Å of any "self" protein atom).
8. Pool aligned water oxygens, KD-tree → DBSCAN with eps ≈ 1.2 Å,
   min_samples = 2.
9. Conservation score per cluster = `(# distinct PDB entries contributing) /
   (# entries in the cluster)`.
10. Annotate every input water with its parent cluster's score.

## Validation case

Before any PDB-scale run, the pipeline must reproduce known results on a
small set:
- **Thrombin set** (Sanschagrin & Kuhn 1998, 10 structures) — vanddraabe
  ships this as its canonical example. Cluster positions and conservation
  rates are published; we should match them within reason.
- **Optionally lysozyme** for stress-testing scale (~1500 entries at 100%
  identity).

Do not scale up until the thrombin numbers look right.

## Conventions and constraints

- All paths live on NFS at `/data/rbg/users/gloriama/`. Never use AFS for
  outputs or intermediate state — AFS tokens expire and break long jobs.
- Compute: NEVER run heavy work on the login node. Submit via `sbatch`.
  Claude Code's job is to orchestrate — write SLURM array job scripts, watch
  the queue, aggregate results.
- Inspect data before writing pipeline code. "Show me the schema" comes
  before "write the loader."
- Validate against published numbers before scaling. The thrombin set is the
  gate; do not pass it without matching.
- Prefer gemmi for structure I/O (fast, modern, handles symmetry properly).
  Fall back to biotite if gemmi is missing something. Avoid biopython where
  the others suffice.
- Numpy + scikit-learn for the clustering math. Don't reach for pandas
  unless the data really is tabular.
- Code in `src/`, scripts in `scripts/`, tests/validation in `tests/`,
  outputs in `results/`. One-off exploration in `notebooks/`.
- Git commits at meaningful checkpoints, not after every edit. Always commit
  before risky multi-file edits.

## Known unknowns / decisions still to make

- `.npz` schema details (see Data section above).
- Resolution cutoff: 2.0 Å vs 2.5 Å. Tighter = cleaner waters, fewer
  clusters with enough members. Decide after seeing the cluster-size
  distribution.
- Minimum cluster size for inclusion (5? 10? 20?).
- How to handle conformational state heterogeneity at 100% identity (apo vs
  holo, open vs closed). Options: split clusters by structural similarity QC
  step, report scores stratified, or merge and accept noise. TBD after
  looking at a few clusters.
- Whether to use raw PDB or PDB-REDO as input. PDB-REDO has more consistent
  water modeling but may not be locally mirrored.
- Crystal-contact filter parameters (the 5 Å cutoff is a starting heuristic).
- DBSCAN eps and min_samples — the 1.2 Å, 2 starting values are reasonable
  but should be tuned on the thrombin set.

## Things explicitly NOT in scope (yet)

- MD-derived waters or hydration site prediction. Crystal waters only.
- Hydrogen bond network analysis on conserved waters. Could come later;
  WatCon does this and is worth referencing if we go there.
- Modeling waters into structures that lack them. We're scoring existing
  deposited waters, not predicting new ones.

## References worth remembering

- Sanschagrin & Kuhn 1998 — original WatCH paper, the methodological
  ancestor.
- PyWATER (Patel et al. 2014) — practical tool, has the mobility/B-factor
  filtering logic.
- vanddraabe (Esposito, CRAN) — most carefully validated reimplementation;
  ships the thrombin reference set.
- ProBiS H₂O (Jukič et al.) — uses ProBiS local alignment + DBSCAN.
- WatCon (Brownless, Harrison-Rawn & Kamerlin 2025, JACS Au) — newest, has
  per-cluster conservation scores; reference for the formal scoring.
- Bottoms, White & Tanner 2006 — six-family conservation analysis, closest
  prior work to what we're doing.
- Carugo & Bordo 1999 — water count vs. resolution, the confound to be
  aware of.
- Chen et al. 2024 (Acta Cryst D) — ~22% of PDB entries deposit no waters;
  filter accordingly.
- HomolWat (Mayol et al. NAR 2020) — adjacent tool for GPCRs, transposes
  rather than annotates.
- RCSB sequence-cluster files: see
  https://www.rcsb.org/docs/grouping-structures/sequence-based-clustering