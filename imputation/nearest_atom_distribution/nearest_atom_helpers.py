from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))  # imputation/

from boltzgen.data import const
from boltzgen.data.data import Structure

from basic_helpers import resolve_npz_path
from gloria_hbond_helpers import gloria_remove_weak_solvents

# ─── Constants ───────────────────────────────────────────────────────────────────

NPZ_ROOT = Path("/data/rbg/shared/datasets/processed_rcsb/rcsb_solvents/structures")

SOLVENT_MOL_TYPE = const.chain_type_ids["SOLVENT"]

HBOND_MIN_DIST: float = const.hbond_min_dist   # 2.5 Å
HBOND_MAX_DIST: float = const.hbond_max_dist   # 3.5 Å

_MOL_CATEGORY_MAP: dict[int, str] = {
    const.chain_type_ids["PROTEIN"]:    "PROTEIN",
    const.chain_type_ids["DNA"]:        "DNA",
    const.chain_type_ids["RNA"]:        "RNA",
    const.chain_type_ids["NONPOLYMER"]: "NONPOLYMER",
    const.chain_type_ids["SOLVENT"]:    "SOLVENT",
    const.chain_type_ids["iSOLVENT"]:   "SOLVENT",
}

_PROTEIN_BACKBONE_ATOMS = set(const.protein_backbone_atom_names)
_NUCLEIC_BACKBONE_ATOMS = set(const.nucleic_backbone_atom_names)

# Element colors matching CPK / PyMOL conventions
ELEMENT_COLORS: dict[str, str] = {
    "C": "#808080",
    "N": "#4472C4",
    "O": "#C0504D",
    "S": "#E6B800",
    "P": "#FF8000",
    "F": "#70AD47",
    "X": "#CCCCCC",
}

# Categorical colors for mol_category + backbone/sidechain labels
_MOLCAT_COLORS: dict[str, str] = {
    "PROTEIN_backbone":  "#1F497D",
    "PROTEIN_sidechain": "#4472C4",
    "DNA_backbone":      "#375623",
    "DNA_base":          "#70AD47",
    "RNA_backbone":      "#4F6228",
    "RNA_base":          "#9BBB59",
    "NONPOLYMER_N/A":    "#F79646",
    "UNKNOWN_N/A":       "#AAAAAA",
}


# ─── Tier 1: atom-level classification ──────────────────────────────────────────

def atom_name_to_element(name: str) -> str:
    """Infer element symbol from PDB atom name (works for common heavy atoms)."""
    stripped = name.strip().lstrip("0123456789")
    return stripped[0].upper() if stripped else "X"


def get_mol_category(mol_type_int: int) -> str:
    return _MOL_CATEGORY_MAP.get(int(mol_type_int), "UNKNOWN")


def classify_backbone_vs_sidechain(atom_name: str, mol_category: str) -> str:
    if mol_category == "PROTEIN":
        return "backbone" if atom_name in _PROTEIN_BACKBONE_ATOMS else "sidechain"
    elif mol_category in ("DNA", "RNA"):
        return "backbone" if atom_name in _NUCLEIC_BACKBONE_ATOMS else "base"
    return "N/A"


def molcat_label(mol_category: str, bb_sc: str) -> str:
    """Compound label combining mol_category and backbone/sidechain classification."""
    return f"{mol_category}_{bb_sc}"


# ─── Tier 2: atom array building ────────────────────────────────────────────────

def _build_residue_and_chain_maps(structure: Structure) -> tuple[np.ndarray, np.ndarray]:
    atom_to_residue = np.full(len(structure.atoms), -1, dtype=np.int64)
    for res_idx, res in enumerate(structure.residues):
        atom_to_residue[res["atom_idx"]: res["atom_idx"] + res["atom_num"]] = res_idx

    atom_to_chain = np.full(len(structure.atoms), -1, dtype=np.int64)
    for chain_idx, chain in enumerate(structure.chains):
        atom_to_chain[chain["atom_idx"]: chain["atom_idx"] + chain["atom_num"]] = chain_idx

    return atom_to_residue, atom_to_chain


def build_all_atom_info(structure: Structure) -> dict:
    """
    Build arrays for ALL present atoms in the structure.
    Used for the k-nearest and per-atom-type distance analyses (Q1, Q2).

    Returns a dict with parallel arrays:
      atom_indices, coords, atom_names, elements,
      res_names, mol_categories, bb_sc, molcat_labels
    """
    atom_to_residue, atom_to_chain = _build_residue_and_chain_maps(structure)

    atom_indices = np.flatnonzero(structure.atoms["is_present"])
    if len(atom_indices) == 0:
        empty = np.array([], dtype=object)
        return {
            "atom_indices": atom_indices,
            "coords": np.zeros((0, 3), dtype=np.float64),
            "atom_names": empty, "elements": empty,
            "res_names": empty, "mol_categories": empty,
            "bb_sc": empty, "molcat_labels": empty,
        }

    atom_names = structure.atoms["name"][atom_indices]
    coords = np.asarray(structure.coords["coords"][atom_indices], dtype=np.float64)
    elements = np.array([atom_name_to_element(n) for n in atom_names])

    res_indices = atom_to_residue[atom_indices]
    chain_indices = atom_to_chain[atom_indices]

    res_names = np.array([
        structure.residues["name"][ri] if ri >= 0 else "UNK"
        for ri in res_indices
    ])
    mol_types = np.array([
        structure.chains["mol_type"][ci] if ci >= 0 else -1
        for ci in chain_indices
    ])
    mol_categories = np.array([get_mol_category(mt) for mt in mol_types])
    bb_sc = np.array([
        classify_backbone_vs_sidechain(an, mc)
        for an, mc in zip(atom_names, mol_categories)
    ])
    molcat_labels = np.array([
        molcat_label(mc, bs) for mc, bs in zip(mol_categories, bb_sc)
    ])

    return {
        "atom_indices": atom_indices,
        "coords": coords,
        "atom_names": atom_names,
        "elements": elements,
        "res_names": res_names,
        "mol_categories": mol_categories,
        "bb_sc": bb_sc,
        "molcat_labels": molcat_labels,
    }


def build_hbond_candidate_info(structure: Structure) -> dict:
    """
    Build arrays for N/O/F atoms only (H-bond candidates).
    Used for H-bond partner distance analysis (Q3).

    Same field layout as build_all_atom_info.
    """
    atom_to_residue, atom_to_chain = _build_residue_and_chain_maps(structure)

    hbond_mask = (
        (
            np.char.startswith(structure.atoms["name"], "N")
            | np.char.startswith(structure.atoms["name"], "O")
            | np.char.startswith(structure.atoms["name"], "F")
        )
        & structure.atoms["is_present"]
    )

    atom_indices = np.flatnonzero(hbond_mask)
    if len(atom_indices) == 0:
        empty = np.array([], dtype=object)
        return {
            "atom_indices": atom_indices,
            "coords": np.zeros((0, 3), dtype=np.float64),
            "atom_names": empty, "elements": empty,
            "res_names": empty, "mol_categories": empty,
            "bb_sc": empty, "molcat_labels": empty,
        }

    atom_names = structure.atoms["name"][atom_indices]
    coords = np.asarray(structure.coords["coords"][atom_indices], dtype=np.float64)
    elements = np.array([atom_name_to_element(n) for n in atom_names])

    res_indices = atom_to_residue[atom_indices]
    chain_indices = atom_to_chain[atom_indices]

    res_names = np.array([
        structure.residues["name"][ri] if ri >= 0 else "UNK"
        for ri in res_indices
    ])
    mol_types = np.array([
        structure.chains["mol_type"][ci] if ci >= 0 else -1
        for ci in chain_indices
    ])
    mol_categories = np.array([get_mol_category(mt) for mt in mol_types])
    bb_sc = np.array([
        classify_backbone_vs_sidechain(an, mc)
        for an, mc in zip(atom_names, mol_categories)
    ])
    molcat_labels = np.array([
        molcat_label(mc, bs) for mc, bs in zip(mol_categories, bb_sc)
    ])

    return {
        "atom_indices": atom_indices,
        "coords": coords,
        "atom_names": atom_names,
        "elements": elements,
        "res_names": res_names,
        "mol_categories": mol_categories,
        "bb_sc": bb_sc,
        "molcat_labels": molcat_labels,
    }


# ─── Tier 3: k-nearest atom identity analysis (Q1) ──────────────────────────────

def get_k_nearest_identities(
    water_coords: np.ndarray,
    all_atom_info: dict,
    k_max: int = 10,
) -> dict:
    """
    For each water, find the k_max nearest atoms (all heavy atoms from stripped structure).

    Returns a dict with parallel arrays of shape (n_waters, k_max):
      distances     : float64  — Å distance to k-th nearest atom
      elements      : object   — element symbol (C, N, O, …)
      mol_categories: object   — PROTEIN / DNA / RNA / NONPOLYMER / UNKNOWN
      atom_names    : object   — raw atom name (CA, OG, N, …)
      bb_sc         : object   — backbone / sidechain / base / N/A
      molcat_labels : object   — e.g. "PROTEIN_backbone"

    Entries are NaN / "X" where k > number of atoms in the structure.
    """
    n_w = len(water_coords)
    n_a = len(all_atom_info["coords"])
    k_actual = min(k_max, n_a)

    distances    = np.full((n_w, k_max), np.nan)
    elements     = np.full((n_w, k_max), "X",       dtype=object)
    mol_cats     = np.full((n_w, k_max), "UNKNOWN",  dtype=object)
    atom_names   = np.full((n_w, k_max), "?",        dtype=object)
    bb_sc        = np.full((n_w, k_max), "N/A",      dtype=object)
    molcat_labs  = np.full((n_w, k_max), "UNKNOWN_N/A", dtype=object)

    if n_w == 0 or n_a == 0:
        return dict(
            distances=distances, elements=elements, mol_categories=mol_cats,
            atom_names=atom_names, bb_sc=bb_sc, molcat_labels=molcat_labs,
        )

    tree = cKDTree(all_atom_info["coords"])
    dists, idxs = tree.query(water_coords, k=k_actual, workers=-1)
    if dists.ndim == 1:
        dists = dists[:, np.newaxis]
        idxs  = idxs[:, np.newaxis]

    distances[:, :k_actual]   = dists
    elements[:, :k_actual]    = all_atom_info["elements"][idxs]
    mol_cats[:, :k_actual]    = all_atom_info["mol_categories"][idxs]
    atom_names[:, :k_actual]  = all_atom_info["atom_names"][idxs]
    bb_sc[:, :k_actual]       = all_atom_info["bb_sc"][idxs]
    molcat_labs[:, :k_actual] = all_atom_info["molcat_labels"][idxs]

    return dict(
        distances=distances, elements=elements, mol_categories=mol_cats,
        atom_names=atom_names, bb_sc=bb_sc, molcat_labels=molcat_labs,
    )


# ─── Tier 4: per-atom-type nearest-distance analysis (Q2) ───────────────────────

def compute_atomtype_nearest_dists(
    water_coords: np.ndarray,
    all_atom_info: dict,
    label_field: str = "elements",
) -> dict[str, np.ndarray]:
    """
    For each unique atom-type label, compute the per-water minimum distance
    to any atom of that type.

    label_field: which identity array to group by.
      "elements"     → C, N, O, S, P, F, …
      "molcat_labels" → PROTEIN_backbone, PROTEIN_sidechain, DNA_base, …

    Returns: {label: float64 array of shape (n_waters,)}
    Each entry is the distance from that water to the nearest atom of that type.
    """
    n_w = len(water_coords)
    if n_w == 0 or len(all_atom_info["coords"]) == 0:
        return {}

    labels = all_atom_info[label_field]
    unique_labels = np.unique(labels)
    result: dict[str, np.ndarray] = {}

    for lbl in unique_labels:
        atom_coords = all_atom_info["coords"][labels == lbl]
        if len(atom_coords) == 0:
            continue
        tree = cKDTree(atom_coords)
        dists, _ = tree.query(water_coords, k=1, workers=-1)
        result[str(lbl)] = dists.astype(np.float64)

    return result


# ─── Tier 5: H-bond partner distance analysis (Q3) ──────────────────────────────

def compute_hbond_partner_distances(
    water_coords: np.ndarray,
    candidate_info: dict,
    min_dist: float = HBOND_MIN_DIST,
    max_dist: float = HBOND_MAX_DIST,
) -> list[dict]:
    """
    For every H-bond partner of every water, return a flat list of contact dicts.

    Each dict has: distance, element, mol_category, atom_name, bb_sc, molcat_label.
    This lets callers filter/group by partner identity downstream.
    """
    if len(water_coords) == 0 or len(candidate_info["coords"]) == 0:
        return []

    tree = cKDTree(candidate_info["coords"])
    within_max = tree.query_ball_point(water_coords, r=max_dist)

    contacts: list[dict] = []
    for water_pos, neighbors in zip(water_coords, within_max):
        for n in neighbors:
            d = float(np.linalg.norm(water_pos - candidate_info["coords"][n]))
            if d >= min_dist:
                contacts.append({
                    "distance":      d,
                    "element":       str(candidate_info["elements"][n]),
                    "mol_category":  str(candidate_info["mol_categories"][n]),
                    "atom_name":     str(candidate_info["atom_names"][n]),
                    "bb_sc":         str(candidate_info["bb_sc"][n]),
                    "molcat_label":  str(candidate_info["molcat_labels"][n]),
                })
    return contacts


# ─── Tier 6: water coordinate extraction ────────────────────────────────────────

def extract_solvent_coords(structure: Structure) -> np.ndarray:
    """Extract one oxygen coordinate per SOLVENT chain from structure."""
    atom_coords = structure.coords["coords"]
    atom_present = structure.atoms["is_present"]
    coords_list = []

    for chain in structure.chains:
        if chain["mol_type"] != SOLVENT_MOL_TYPE:
            continue
        atom_start = chain["atom_idx"]
        atom_end = atom_start + chain["atom_num"]
        present_offsets = np.flatnonzero(atom_present[atom_start:atom_end])
        if len(present_offsets) > 0:
            coords_list.append(atom_coords[atom_start + present_offsets[0]])

    if not coords_list:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(coords_list, dtype=np.float64)


# ─── Tier 7: per-PDB runner and collection ──────────────────────────────────────

def analyze_one_pdb(
    pdb_id: str,
    min_hbonds: int = 3,
    k_max: int = 10,
    npz_root: Path = NPZ_ROOT,
) -> dict:
    """
    Run all three analyses for one PDB entry (real ≥min_hbonds waters only).

    Returns:
      pdb_id, n_waters
      k_nearest       : dict from get_k_nearest_identities
      atomtype_dists  : dict {label: per-water min-dist array} (element grouping)
      molcat_dists    : dict {label: per-water min-dist array} (molcat_label grouping)
      hbond_contacts  : flat list[dict] from compute_hbond_partner_distances
    """
    npz_path = resolve_npz_path(pdb_id, npz_root)
    gt_raw = Structure.load(npz_path)
    gt_raw = gt_raw.to_one_solvent_per_chain(gt_raw)

    gt_filtered = gloria_remove_weak_solvents(gt_raw, min_hbonds=min_hbonds)
    water_coords = extract_solvent_coords(gt_filtered)

    stripped = gt_raw.remove_solvents()
    all_atom_info   = build_all_atom_info(stripped)
    hbond_cand_info = build_hbond_candidate_info(stripped)

    k_nearest      = get_k_nearest_identities(water_coords, all_atom_info, k_max=k_max)
    atomtype_dists = compute_atomtype_nearest_dists(water_coords, all_atom_info, label_field="elements")
    molcat_dists   = compute_atomtype_nearest_dists(water_coords, all_atom_info, label_field="molcat_labels")
    hbond_contacts = compute_hbond_partner_distances(water_coords, hbond_cand_info)

    return {
        "pdb_id":        pdb_id,
        "n_waters":      len(water_coords),
        "k_nearest":     k_nearest,
        "atomtype_dists": atomtype_dists,
        "molcat_dists":  molcat_dists,
        "hbond_contacts": hbond_contacts,
    }


def collect_results(
    pdb_ids: list[str],
    min_hbonds: int = 3,
    k_max: int = 10,
    npz_root: Path = NPZ_ROOT,
) -> list[dict]:
    """Run analyze_one_pdb for each PDB ID and return per-PDB result dicts."""
    results = []
    for i, pdb_id in enumerate(pdb_ids):
        print(f"[{i+1}/{len(pdb_ids)}] {pdb_id}", end="  ", flush=True)
        try:
            r = analyze_one_pdb(pdb_id, min_hbonds=min_hbonds, k_max=k_max, npz_root=npz_root)
            print(f"n_waters={r['n_waters']}")
            results.append(r)
        except Exception as e:
            print(f"ERROR: {e}")
    return results


# ─── Tier 8: aggregation ────────────────────────────────────────────────────────

def aggregate_k_nearest(results: list[dict], field: str) -> np.ndarray:
    """
    Concatenate a (n_waters, k_max) array field from k_nearest across all results.

    field: one of 'distances', 'elements', 'mol_categories', 'atom_names',
           'bb_sc', 'molcat_labels'.
    """
    arrays = [r["k_nearest"][field] for r in results if r["n_waters"] > 0]
    if not arrays:
        return np.array([])
    return np.concatenate(arrays, axis=0)


def aggregate_dist_dict(
    results: list[dict],
    result_key: str,
) -> dict[str, np.ndarray]:
    """
    Pool per-water distance arrays across all PDB results for a given result_key
    ('atomtype_dists' or 'molcat_dists').

    Returns: {label: concatenated float64 array of per-water min distances}
    """
    merged: dict[str, list[np.ndarray]] = {}
    for r in results:
        for lbl, dists in r[result_key].items():
            merged.setdefault(lbl, []).append(dists)
    return {lbl: np.concatenate(arrs) for lbl, arrs in merged.items()}


def aggregate_hbond_contacts(results: list[dict]) -> list[dict]:
    """Flatten all H-bond contact dicts across all PDB results."""
    return [c for r in results for c in r["hbond_contacts"]]


# ─── Tier 9: plotting ────────────────────────────────────────────────────────────

def set_paper_style() -> None:
    """Apply ICML-appropriate matplotlib rcParams."""
    mpl.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["STIXGeneral", "DejaVu Serif"],
        "font.size":         9,
        "axes.labelsize":    9,
        "axes.titlesize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "legend.frameon":    False,
        "lines.linewidth":   1.0,
        "axes.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
    })


def _get_color(label: str, label_type: str) -> str:
    if label_type == "elements":
        return ELEMENT_COLORS.get(label, "#CCCCCC")
    return _MOLCAT_COLORS.get(label, "#CCCCCC")


def plot_k_nearest_stacked_bar(
    all_identities: np.ndarray,
    all_distances: np.ndarray,
    identity_label_type: str = "elements",
    k_max: int | None = None,
    top_n_categories: int = 8,
    figsize: tuple[float, float] = (9, 4),
) -> plt.Figure:
    """
    Stacked bar chart: for each rank k (x-axis), show the fraction of waters
    whose k-th nearest atom belongs to each identity category (stacked bars).

    all_identities: object array (n_waters, k_max) — e.g. element symbols
    all_distances:  float array  (n_waters, k_max) — used to detect NaN (missing k)
    identity_label_type: 'elements' or 'molcat_labels' (controls colors)
    top_n_categories: pool the rest into 'other' to keep the chart readable
    """
    if k_max is None:
        k_max = all_identities.shape[1]

    valid_mask = np.isfinite(all_distances[:, :k_max].astype(float))

    # Determine top categories by total count across all ranks
    flat_valid = all_identities[:, :k_max][valid_mask]
    cats, cnts = np.unique(flat_valid, return_counts=True)
    top_cats = cats[np.argsort(-cnts)][:top_n_categories]
    top_cats_set = set(top_cats)

    all_cats = list(top_cats) + (["other"] if len(cats) > top_n_categories else [])

    fracs = np.zeros((len(all_cats), k_max))
    for k in range(k_max):
        col = all_identities[:, k]
        vm  = valid_mask[:, k]
        n   = vm.sum()
        if n == 0:
            continue
        col_valid = col[vm]
        for ci, cat in enumerate(all_cats):
            if cat == "other":
                fracs[ci, k] = np.isin(col_valid, list(top_cats_set), invert=True).sum() / n
            else:
                fracs[ci, k] = (col_valid == cat).sum() / n

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(1, k_max + 1)
    bottom = np.zeros(k_max)

    for ci, cat in enumerate(all_cats):
        color = "lightgray" if cat == "other" else _get_color(cat, identity_label_type)
        ax.bar(x, fracs[ci], bottom=bottom, label=cat, color=color, edgecolor="none")
        bottom += fracs[ci]

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in x])
    ax.set_xlabel("rank k (k-th nearest atom to water O)")
    ax.set_ylabel("fraction of waters")
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_atomtype_min_dist_distributions(
    dist_dict: dict[str, np.ndarray],
    label_type: str = "elements",
    max_dist: float = 6.0,
    n_bins: int = 40,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    One subplot per atom-type label: histogram of per-water minimum distances.

    Subplots are sorted by global minimum distance (smallest first).
    Title of each subplot shows the global min and sample count.
    """
    if not dist_dict:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig

    mins   = {lbl: float(np.nanmin(d)) for lbl, d in dist_dict.items()}
    labels = sorted(mins, key=mins.get)

    n = len(labels)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (ncols * 3.0, nrows * 2.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i, lbl in enumerate(labels):
        ax     = axes[i // ncols][i % ncols]
        dists  = dist_dict[lbl]
        finite = dists[np.isfinite(dists) & (dists <= max_dist)]
        color  = _get_color(lbl, label_type)
        ax.hist(finite, bins=n_bins, density=True, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(f"{lbl}\nmin={mins[lbl]:.2f} Å  n={len(finite)}", fontsize=8)
        ax.set_xlabel("min dist to water O (Å)", fontsize=7)
        ax.set_ylabel("density", fontsize=7)
        ax.set_xlim(0, max_dist)
        ax.axvline(mins[lbl], color="k", linewidth=0.8, linestyle="--")

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle("Distribution of per-water minimum distance to each atom type", y=1.02)
    fig.tight_layout()
    return fig


def plot_hbond_distance_histogram(
    contacts: list[dict],
    split_by: str | None = "element",
    n_bins: int = 40,
    figsize: tuple[float, float] = (6, 3.5),
) -> plt.Figure:
    """
    Histogram of H-bond partner distances.

    contacts: flat list of dicts from compute_hbond_partner_distances / aggregate_hbond_contacts.
    split_by: None → single histogram; 'element' → one curve per partner element.
    """
    if not contacts:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "no contacts", ha="center", va="center", transform=ax.transAxes)
        return fig

    all_dists = np.array([c["distance"] for c in contacts])
    fig, ax = plt.subplots(figsize=figsize)

    if split_by is None:
        ax.hist(all_dists, bins=n_bins, density=True,
                color="#4472C4", alpha=0.85, edgecolor="none",
                label=f"all contacts (n={len(all_dists)})")
    else:
        unique_vals = sorted({c[split_by] for c in contacts})
        for val in unique_vals:
            sub = np.array([c["distance"] for c in contacts if c[split_by] == val])
            color = ELEMENT_COLORS.get(val, "#AAAAAA") if split_by == "element" else None
            ax.hist(sub, bins=n_bins, density=True, alpha=0.6, edgecolor="none",
                    label=f"{val} (n={len(sub)})", color=color)

    med = float(np.nanmedian(all_dists))
    ax.axvline(med, color="k", linewidth=1, linestyle="--",
               label=f"median = {med:.2f} Å")
    ax.set_xlabel("H-bond partner distance (Å)")
    ax.set_ylabel("density")
    ax.set_title(f"H-bond partner distances for real ≥3-hbond waters\n(n = {len(all_dists)} contacts total)")
    ax.legend()
    fig.tight_layout()
    return fig


def print_atomtype_min_table(dist_dict: dict[str, np.ndarray]) -> None:
    """Print a sorted table of global minimum distances and medians per atom type."""
    rows = []
    for lbl, dists in dist_dict.items():
        finite = dists[np.isfinite(dists)]
        if len(finite) == 0:
            continue
        rows.append((lbl, float(np.min(finite)), float(np.median(finite)), int(len(finite))))
    rows.sort(key=lambda r: r[1])

    header = f"{'atom type':<25}  {'global min (Å)':>14}  {'median (Å)':>10}  {'n waters':>9}"
    print(header)
    print("-" * len(header))
    for lbl, gmin, med, n in rows:
        print(f"{lbl:<25}  {gmin:>14.3f}  {med:>10.3f}  {n:>9}")
