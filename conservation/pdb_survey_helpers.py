from __future__ import annotations

import glob
import random
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

BOLTZGEN_SRC = Path("/data/rbg/users/gloriama/dev/foldeverything/src")
if str(BOLTZGEN_SRC) not in sys.path:
    sys.path.insert(0, str(BOLTZGEN_SRC))

from boltzgen.data.data import Structure  # noqa: E402

NPZ_ROOT = Path("/data/rbg/shared/datasets/processed_rcsb/rcsb_solvents/structures")
CLUSTER_FILE = Path("/data/rbg/users/gloriama/dev/water_conservation/data/clusters-by-entity-100.txt")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def all_npz_paths() -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(NPZ_ROOT / "*.npz")))


def sample_npz_paths(n: int, seed: int = 42) -> list[Path]:
    paths = all_npz_paths()
    rng = random.Random(seed)
    return rng.sample(paths, min(n, len(paths)))


def cluster_representative_paths(cluster_file: Path = CLUSTER_FILE) -> list[Path]:
    """One path per sequence cluster (first member of each line that has an NPZ)."""
    reps = []
    with open(cluster_file) as f:
        for line in f:
            for token in line.split():
                pdb_id = token.split("_")[0].lower()
                p = NPZ_ROOT / f"{pdb_id}.npz"
                if p.exists():
                    reps.append(p)
                    break
    return reps


# ---------------------------------------------------------------------------
# Per-structure stats
# ---------------------------------------------------------------------------

def structure_stats(path: Path) -> dict | None:
    """Return basic shape/solvent stats for one NPZ, or None on load error."""
    try:
        s = Structure.load(path)
    except Exception:
        return None
    return {
        "pdb_id": path.stem,
        "n_atoms": len(s.atoms),
        "n_residues": len(s.residues),
        "n_chains": len(s.chains),
        "n_solvents": s.count_solvents(),
    }


# ---------------------------------------------------------------------------
# Batch survey
# ---------------------------------------------------------------------------

def survey_paths(
    paths: list[Path],
    *,
    verbose: bool = True,
    log_every: int = 500,
) -> list[dict]:
    """Run structure_stats over a list of paths; skip and warn on failures."""
    records = []
    errors = 0
    for i, p in enumerate(paths):
        if verbose and i > 0 and i % log_every == 0:
            print(f"  {i}/{len(paths)}  errors={errors}")
        rec = structure_stats(p)
        if rec is None:
            errors += 1
        else:
            records.append(rec)
    if verbose:
        print(f"Done. {len(records)} ok, {errors} errors out of {len(paths)} paths.")
    return records


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def percentile_summary(records: list[dict], key: str, pcts=(25, 50, 75, 95, 99)) -> dict:
    vals = np.array([r[key] for r in records])
    result = {"mean": float(vals.mean()), "max": int(vals.max())}
    for p in pcts:
        result[f"p{p}"] = float(np.percentile(vals, p))
    return result


def print_survey_summary(records: list[dict]) -> None:
    with_solvents = [r for r in records if r["n_solvents"] > 0]
    print(f"Structures surveyed : {len(records)}")
    print(f"With solvents       : {len(with_solvents)}  ({100*len(with_solvents)/len(records):.1f}%)")
    print()
    for key in ("n_atoms", "n_residues", "n_chains", "n_solvents"):
        s = percentile_summary(records, key)
        print(
            f"{key:15s}  mean={s['mean']:8.0f}  "
            f"p25={s['p25']:6.0f}  p50={s['p50']:6.0f}  "
            f"p75={s['p75']:6.0f}  p95={s['p95']:6.0f}  "
            f"p99={s['p99']:6.0f}  max={s['max']:8d}"
        )
    print()
    print("(solvent stats over all structures including those with 0 solvents)")
    s = percentile_summary(with_solvents, "n_solvents")
    print(
        f"{'n_solvents|>0':15s}  mean={s['mean']:8.0f}  "
        f"p25={s['p25']:6.0f}  p50={s['p50']:6.0f}  "
        f"p75={s['p75']:6.0f}  p95={s['p95']:6.0f}  "
        f"p99={s['p99']:6.0f}  max={s['max']:8d}"
    )
