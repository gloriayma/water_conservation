from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import KDTree

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))  # imputation/ directory

from boltzgen.data.data import Structure
from boltzgen.data import const

from basic_helpers import resolve_npz_path
from impute_solvents_from_triples import impute_solvents_from_atom_triples, filter_solvent_clashes
from gloria_hbond_helpers import gloria_remove_weak_solvents

# ─── Constants ──────────────────────────────────────────────────────────────────

NPZ_ROOT = Path("/data/rbg/shared/datasets/processed_rcsb/rcsb_solvents/structures")

SOLVENT_MOL_TYPES = frozenset({
    const.chain_type_ids["SOLVENT"],
    const.chain_type_ids["iSOLVENT"],
})
POLYMER_MOL_TYPES = frozenset({
    const.chain_type_ids["PROTEIN"],
    const.chain_type_ids["DNA"],
    const.chain_type_ids["RNA"],
})

REAL_COLOR = "#2166ac"
IMPUTED_COLOR = "#d6604d"

# ─── Tier 1: Structure accessors ────────────────────────────────────────────────

def get_residue_coords(
    structure: Structure,
    mol_types: frozenset[int] | None = None,
) -> np.ndarray:
    """
    Return representative atom (atom_center) coordinates for residues.

    atom_center is an absolute atom index storing the Cα for protein residues
    and the oxygen for water/HOH residues. If mol_types is given, only chains
    whose mol_type is in the set are included. Only present residues with a
    present representative atom are returned.

    Returns float32 array of shape (n, 3).
    """
    if mol_types is not None:
        chain_indices = np.flatnonzero(
            np.isin(structure.chains["mol_type"], list(mol_types))
        )
    else:
        chain_indices = np.arange(len(structure.chains))

    atom_coords = structure.atoms["coords"]
    atom_present = structure.atoms["is_present"]
    coords_list = []

    for ci in chain_indices:
        chain = structure.chains[ci]
        res_start = chain["res_idx"]
        res_end = res_start + chain["res_num"]
        for res in structure.residues[res_start:res_end]:
            if not res["is_present"]:
                continue
            center_idx = int(res["atom_center"])
            if center_idx >= len(atom_coords) or not atom_present[center_idx]:
                continue
            coords_list.append(atom_coords[center_idx])

    if not coords_list:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(coords_list).astype(np.float32)


# ─── Tier 2: Distance computations ──────────────────────────────────────────────

def knn_distances(
    query_coords: np.ndarray,
    ref_coords: np.ndarray,
    ks: Sequence[int],
) -> np.ndarray:
    """
    For each query point, return distance to its k-th nearest neighbor in ref_coords.

    Returns float64 array of shape (n_query, len(ks)).
    Entries are NaN where k > len(ref_coords).
    """
    n_q = len(query_coords)
    result = np.full((n_q, len(ks)), np.nan)
    if n_q == 0 or len(ref_coords) == 0:
        return result

    k_max = min(max(ks), len(ref_coords))
    tree = KDTree(ref_coords)
    dists, _ = tree.query(query_coords, k=k_max, workers=-1)
    if dists.ndim == 1:
        dists = dists[:, np.newaxis]

    for j, k in enumerate(ks):
        if k <= k_max:
            result[:, j] = dists[:, k - 1]
    return result


def radius_neighbor_counts(
    query_coords: np.ndarray,
    ref_coords: np.ndarray,
    radii: Sequence[float],
) -> np.ndarray:
    """
    For each query point, count ref_coords within each radius.

    Returns int32 array of shape (n_query, len(radii)).
    """
    n_q = len(query_coords)
    result = np.zeros((n_q, len(radii)), dtype=np.int32)
    if n_q == 0 or len(ref_coords) == 0:
        return result

    tree = KDTree(ref_coords)
    for j, r in enumerate(radii):
        result[:, j] = tree.query_ball_point(query_coords, r=r, workers=-1, return_length=True)
    return result


def centroid_distances(query_coords: np.ndarray, ref_coords: np.ndarray) -> np.ndarray:
    """
    Distance from each query point to the centroid of ref_coords.

    Returns float64 array of shape (n_query,).
    """
    if len(query_coords) == 0 or len(ref_coords) == 0:
        return np.full(len(query_coords), np.nan)
    centroid = ref_coords.mean(axis=0)
    return np.linalg.norm(query_coords.astype(np.float64) - centroid.astype(np.float64), axis=1)


def self_nearest_neighbor_distances(coords: np.ndarray) -> np.ndarray:
    """
    For each point, distance to its nearest other point within the same set.

    Returns float64 array of shape (n,). NaN for sets with fewer than 2 points.
    """
    if len(coords) < 2:
        return np.full(len(coords), np.nan)
    tree = KDTree(coords)
    dists, _ = tree.query(coords, k=2, workers=-1)
    return dists[:, 1].astype(np.float64)


# ─── Tier 3: Structure loading ───────────────────────────────────────────────────

def load_gt_and_imputed(
    pdb_id: str,
    npz_root: Path = NPZ_ROOT,
    max_hbond_length: float = 3.5,
    protein_clash_rad: float = 2.2,
    solvent_clash_rad: float = 3.0,
    min_hbonds_gt: int = 3,
) -> tuple[Structure, Structure, Structure]:
    """
    Load and prepare three structures for a PDB ID:
      - gt_real: ground-truth waters with >= min_hbonds_gt H-bonds
      - imputed: imputed waters placed by triple geometry, clash-filtered
      - gt_stripped: protein-only structure (no solvents), used as reference

    Imputed waters have mol_type=iSOLVENT (5); real waters have mol_type=SOLVENT (4).
    """
    npz_path = resolve_npz_path(pdb_id, npz_root)
    gt_raw = Structure.load(npz_path)
    gt_raw = gt_raw.to_one_solvent_per_chain(gt_raw)

    gt_real = gloria_remove_weak_solvents(gt_raw, min_hbonds=min_hbonds_gt)
    gt_stripped = gt_raw.remove_solvents()

    imputed = impute_solvents_from_atom_triples(gt_stripped, max_hbond_length=max_hbond_length)
    imputed_filtered = filter_solvent_clashes(
        imputed,
        protein_clash_rad=protein_clash_rad,
        solvent_clash_rad=solvent_clash_rad,
    )
    return gt_real, imputed_filtered, gt_stripped


def analyze_pdb(
    pdb_id: str,
    k_list: Sequence[int],
    radius_list: Sequence[float],
    npz_root: Path = NPZ_ROOT,
    max_hbond_length: float = 3.5,
    protein_clash_rad: float = 2.2,
    solvent_clash_rad: float = 3.0,
    min_hbonds_gt: int = 3,
) -> dict:
    """
    Run all distance analyses for one PDB ID against its polymer residue centers.

    Returns a dict with:
      pdb_id, n_real, n_imputed,
      real_knn / imputed_knn      : shape (n_waters, len(k_list))
      real_radius / imputed_radius : shape (n_waters, len(radius_list))
      real_centroid / imputed_centroid : shape (n_waters,)
      real_ww / imputed_ww        : shape (n_waters,)  [water-water nearest-neighbor]
    """
    gt_real, imputed, gt_stripped = load_gt_and_imputed(
        pdb_id, npz_root, max_hbond_length,
        protein_clash_rad, solvent_clash_rad, min_hbonds_gt,
    )
    ref_coords = get_residue_coords(gt_stripped, mol_types=POLYMER_MOL_TYPES)
    real_coords = get_residue_coords(gt_real, mol_types=SOLVENT_MOL_TYPES)
    imp_coords = get_residue_coords(imputed, mol_types=SOLVENT_MOL_TYPES)

    return {
        "pdb_id": pdb_id,
        "n_real": len(real_coords),
        "n_imputed": len(imp_coords),
        "real_knn":      knn_distances(real_coords, ref_coords, k_list),
        "imputed_knn":   knn_distances(imp_coords,  ref_coords, k_list),
        "real_radius":   radius_neighbor_counts(real_coords, ref_coords, radius_list),
        "imputed_radius": radius_neighbor_counts(imp_coords, ref_coords, radius_list),
        "real_centroid":    centroid_distances(real_coords, ref_coords),
        "imputed_centroid": centroid_distances(imp_coords,  ref_coords),
        "real_ww":    self_nearest_neighbor_distances(real_coords),
        "imputed_ww": self_nearest_neighbor_distances(imp_coords),
    }


def collect_results(
    pdb_ids: list[str],
    k_list: Sequence[int],
    radius_list: Sequence[float],
    **kwargs,
) -> list[dict]:
    """Run analyze_pdb for each PDB ID and return a list of result dicts."""
    results = []
    for i, pdb_id in enumerate(pdb_ids):
        print(f"[{i+1}/{len(pdb_ids)}] {pdb_id}", end="  ", flush=True)
        try:
            r = analyze_pdb(pdb_id, k_list, radius_list, **kwargs)
            print(f"real={r['n_real']}  imputed={r['n_imputed']}")
            results.append(r)
        except Exception as e:
            print(f"ERROR: {e}")
    return results


# ─── Tier 4: Aggregation ─────────────────────────────────────────────────────────

def concat_field(results: list[dict], field: str) -> np.ndarray:
    """Concatenate one result field across all PDB entries along axis 0."""
    arrays = [r[field] for r in results if np.asarray(r[field]).size > 0]
    if not arrays:
        return np.array([])
    return np.concatenate(arrays, axis=0)


# ─── Tier 5: Plotting — ICML paper style ─────────────────────────────────────────

def set_paper_style() -> None:
    """
    Apply ICML-appropriate matplotlib rcParams.

    Targets 9 pt serif font, no top/right spines, 300 dpi for saving.
    Call once at the top of a notebook to affect all subsequent figures.
    """
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def ecdf_xy(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) step-function arrays for an ECDF plot, ignoring NaN."""
    data = np.sort(np.asarray(data, float))
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return np.array([]), np.array([])
    n = len(data)
    x = np.concatenate([[data[0]], data])
    y = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    return x, y


def plot_knn_violins(
    real_knn: np.ndarray,
    imputed_knn: np.ndarray,
    k_list: Sequence[int],
    figsize: tuple[float, float] = (6.5, 3.0),
) -> plt.Figure:
    """
    Violin plot of kNN-distance distributions, one violin pair per k.

    x-axis: k value; y-axis: distance (Å) to the k-th nearest residue center.
    Real waters (blue) and imputed waters (red) shown side by side per k.
    """
    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(k_list))
    width = 0.30

    for j, k in enumerate(k_list):
        for vals, color, offset in [
            (real_knn[:, j],    REAL_COLOR,    -width / 2),
            (imputed_knn[:, j], IMPUTED_COLOR, +width / 2),
        ]:
            vals = vals[np.isfinite(vals)]
            if len(vals) < 2:
                continue
            vp = ax.violinplot(
                [vals],
                positions=[positions[j] + offset],
                widths=width,
                showmedians=True,
                showextrema=False,
            )
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.7)
            vp["cmedians"].set_color(color)
            vp["cmedians"].set_linewidth(1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"k = {k}" for k in k_list])
    ax.set_xlabel("k-th nearest residue center")
    ax.set_ylabel("distance (Å)")
    ax.set_title("Nearest-neighbor distances to residue centers")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=REAL_COLOR,    alpha=0.7, label=f"real ≥3 H-bonds (n = {len(real_knn)})"),
        Patch(facecolor=IMPUTED_COLOR, alpha=0.7, label=f"imputed (n = {len(imputed_knn)})"),
    ])
    fig.tight_layout()
    return fig


def plot_radius_count_ecdfs(
    real_radius: np.ndarray,
    imputed_radius: np.ndarray,
    radius_list: Sequence[float],
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Panel of ECDFs for N(r) = #{residue centers within r Å}, real vs imputed.

    One subplot per radius value, 4 per row.
    """
    n_r = len(radius_list)
    ncols = min(4, n_r)
    nrows = (n_r + ncols - 1) // ncols
    if figsize is None:
        figsize = (ncols * 2.5, nrows * 2.2)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for j, r in enumerate(radius_list):
        ax = axes[j // ncols][j % ncols]
        for vals, color, label in [
            (real_radius[:, j].astype(float),    REAL_COLOR,    "real"),
            (imputed_radius[:, j].astype(float), IMPUTED_COLOR, "imputed"),
        ]:
            x, y = ecdf_xy(vals)
            if len(x) > 0:
                ax.plot(x, y, color=color, label=label)
        ax.set_xlabel(f"N(r = {r:.0f} Å)")
        ax.set_ylabel("ECDF")
        ax.set_title(f"r = {r:.0f} Å")
        if j == 0:
            ax.legend()

    for j in range(n_r, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Residue centers within radius r: real vs imputed waters", y=1.02)
    fig.tight_layout()
    return fig


def plot_ecdf_comparison(
    real_vals: np.ndarray,
    imputed_vals: np.ndarray,
    xlabel: str,
    title: str,
    figsize: tuple[float, float] = (3.5, 2.8),
) -> plt.Figure:
    """
    ECDF comparison of one scalar distribution for real vs imputed waters.

    General-purpose; used for centroid distances, water-water distances, etc.
    """
    n_r = int(np.isfinite(np.asarray(real_vals, float)).sum())
    n_i = int(np.isfinite(np.asarray(imputed_vals, float)).sum())

    fig, ax = plt.subplots(figsize=figsize)
    for vals, color, label in [
        (real_vals,    REAL_COLOR,    f"real ≥3 H-bonds (n = {n_r})"),
        (imputed_vals, IMPUTED_COLOR, f"imputed (n = {n_i})"),
    ]:
        x, y = ecdf_xy(np.asarray(vals, float))
        if len(x) > 0:
            ax.plot(x, y, color=color, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("ECDF")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
