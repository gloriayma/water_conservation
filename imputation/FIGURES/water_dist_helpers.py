"""
Helper functions for building the water-count distribution figure.

All filtering and counting is done from the manifest.json alone — no npz reads needed.

Manifest mol_type codes:
    0 = protein polymer
    1 = nucleic acid
    2 = carbohydrate / modified residue
    3 = small molecule (ligand)
    4 = water
"""

import json
from datetime import datetime
from pathlib import Path

MANIFEST_PATH = Path("/data/rbg/shared/datasets/processed_rcsb/rcsb_solvents/manifest.json")

CUTOFF_DATE = datetime(2023, 6, 1)
MAX_RESOLUTION = 9.0
MAX_RESIDUES = 1024
MIN_WATERS_FOR_INCLUSION = 10

MOL_TYPE_WATER = 4


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        return None


def count_waters_from_chains(chains: list[dict]) -> int:
    return sum(c["num_residues"] for c in chains if c["mol_type"] == MOL_TYPE_WATER)


def count_nonwater_residues(chains: list[dict]) -> int:
    return sum(c["num_residues"] for c in chains if c["mol_type"] != MOL_TYPE_WATER)


def filter_and_count(
    manifest: list[dict],
    cutoff_date: datetime = CUTOFF_DATE,
    max_resolution: float = MAX_RESOLUTION,
    max_residues: int = MAX_RESIDUES,
) -> tuple[list[dict], dict]:
    """
    Apply filters to manifest entries and return passing entries with their water counts.

    Returns:
        filtered: list of dicts with keys 'id', 'n_waters', 'n_residues'
        stats: counts at each filter stage
    """
    total = len(manifest)
    after_date = after_method = after_resolution = after_residues = 0

    filtered = []
    for entry in manifest:
        s = entry["structure"]

        dep = _parse_date(s.get("deposited"))
        if dep is None or dep >= cutoff_date:
            continue
        after_date += 1

        if s.get("method") != "x-ray diffraction":
            continue
        after_method += 1

        res = s.get("resolution")
        if res is None or res > max_resolution:
            continue
        after_resolution += 1

        n_res = count_nonwater_residues(entry["chains"])
        if n_res > max_residues:
            continue
        after_residues += 1

        n_waters = count_waters_from_chains(entry["chains"])
        filtered.append({"id": entry["id"], "n_waters": n_waters, "n_residues": n_res})

    stats = {
        "total_in_manifest": total,
        "after_deposit_date_filter": after_date,
        "after_method_filter": after_method,
        "after_resolution_filter": after_resolution,
        "after_residues_filter": after_residues,
        "n_geq10_waters": sum(1 for e in filtered if e["n_waters"] >= MIN_WATERS_FOR_INCLUSION),
    }
    return filtered, stats


def make_water_count_figure(
    filtered: list[dict],
    stats: dict,
    min_waters: int = MIN_WATERS_FOR_INCLUSION,
    save_path: Path | None = None,
):
    """
    Plot the distribution of water counts per structure (ICML single-column style).
    Vertical dashed line at min_waters threshold.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    # ICML single-column figure style
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    counts = np.array([e["n_waters"] for e in filtered])

    fig, ax = plt.subplots(figsize=(3.25, 2.4))

    # Log-spaced bins to handle the heavy tail
    max_count = int(counts.max())
    bins = np.concatenate([
        np.arange(0, 50, 1),
        np.arange(50, 200, 5),
        np.arange(200, min(max_count + 50, 1500), 20),
    ])
    bins = bins[bins <= max_count + 20]
    bins = np.unique(bins)

    ax.hist(counts, bins=bins, color="#4878d0", edgecolor="none", alpha=0.85)
    ax.axvline(min_waters, color="#e24a33", linestyle="--", linewidth=1.0,
               label=f"$\\geq${min_waters} waters\n(n={stats['n_geq10_waters']:,})")

    ax.set_xlabel("Number of solvents per structure")
    ax.set_ylabel("Count")
    ax.set_title("Solvent count distribution\n(filtered PDB, X-ray, ≤9 Å, ≤1024 res, pre-2023)")
    ax.legend(frameon=False)

    n_total = len(filtered)
    ax.annotate(f"N = {n_total:,}", xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=7)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig, ax
