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
First task in the first real session is to inspect a sample of `.npz` files and verify
what's actually in it. They are housed at /data/rbg/shared/datasets/processed_rcsb/rcsb_solvents. 

Critical fields to check for in the schema:
- water atom records (coordinates, residue type HOH/WAT)
- B-factors and occupancies on every atom (needed for quality filtering)
- alt loc identifiers (needed to deduplicate alternate conformations)
- unit cell + space group information from CRYST1
  (needed to identify crystal-contact waters via symmetry mate expansion)

If the latter two are missing, note this. (whether they are missing for waters, or for other atoms, etc.)

## Method

Per same-sequence cluster, the recipe is:

1. (Completed) Pull RCSB's precomputed sequence-cluster file at **100% identity**
   (`clusters-by-entity-100.txt`). 

2. Filter entries to X-ray ≤ 2.0 Å (or ≤ 2.5 Å, try that too and record statistics) and
   drop entries with no deposited waters. Record statistics for how many are dropped. 
3. Filter clusters to ≥5 members. Record statistics for number of clusters, and how large clusters are (histogram). 

BEFORE any of the next steps, please report these statistics because I need to see if this is feasible (if clusters are large enough and there are enough clusters to run this on.)


4. Take one representative chain per entry to avoid NCS double-counting.
5. Structurally align all entries pairwise onto each other within the cluster. UNLESS the cluster is too big. (record such clusters). In that case, align onto the highest resolution entry. If alignments are all bad, pick another entry to try to align it to. 
At this stage, I want you to investigate how good the alignments are. Report RMSDs and also local alignment metrics. Use your judgment and pick some RMSD threshold to consider to be "close enough", and also prepare a few ids for me that represent certain thresholds of RMSDs to visually inspect so I can decide a RMSD threshold.


7. Filter waters near crystal contacts (within ~5 Å of a symmetry mate but
   not within ~5 Å of any "self" protein atom).
8. Pool aligned water oxygens, KD-tree → DBSCAN with eps ≈ 1.2 Å,
   min_samples = 2.
9. Conservation score per cluster = `(# distinct PDB entries contributing) /
   (# entries in the cluster)`.
10. Annotate every input water with its parent cluster's score.


TODO: try this entire thing with a B factor filter (only keep B factors less than something, and normalize this.) Or alternatively try filtering waters by quality (Carugo mobility ≥ 2.0). 

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
- If you need to run anything in python, use the boltzgen_env environment. 
- Inspect data before writing pipeline code. "Show me the schema" comes
  before "write the loader."
- Validate against published numbers before scaling. The thrombin set is the
  gate; do not pass it without matching.
- Prefer gemmi for structure I/O (fast, modern, handles symmetry properly).
  Fall back to biotite if gemmi is missing something. Avoid biopython where
  the others suffice.
- Numpy + scikit-learn for the clustering math. Don't reach for pandas
  unless the data really is tabular.
- Code in `scripts/`, I want validation / proof of things to be in ipynbs - make the organization very clear with markdown cells for section headers and descriptions. Have the ipynbs call the helper functions in scripts, and make the scripts functions very readable, atomic, and reusable and meaningful. I need to be able to run the cells and reproduce your work and inspect your code by looking in the helper function files.
- Every notebook must include `%load_ext autoreload` and `%autoreload 2` in the first code cell (before any imports).

Please also create a new md file to document, extensively, what you've explored and tried - including things that did not work, as well as the reason they didn't work- and small scripts you run, and statistics you've found. Also create a separate md file to just record results. 
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