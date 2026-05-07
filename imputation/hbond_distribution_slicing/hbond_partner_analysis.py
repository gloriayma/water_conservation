from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))

from boltzgen.data import const
from boltzgen.data.data import Structure

from gloria_hbond_helpers import gloria_remove_weak_solvents
from impute_solvents_from_triples import filter_solvent_clashes, impute_solvents_from_atom_triples

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NPZ_ROOT = Path("/data/rbg/shared/datasets/processed_rcsb/rcsb_solvents/structures")

MOL_CATEGORY_MAP: dict[int, str] = {
    const.chain_type_ids["PROTEIN"]: "PROTEIN",
    const.chain_type_ids["DNA"]: "DNA",
    const.chain_type_ids["RNA"]: "RNA",
    const.chain_type_ids["NONPOLYMER"]: "NONPOLYMER",
    const.chain_type_ids["SOLVENT"]: "SOLVENT",
    const.chain_type_ids["iSOLVENT"]: "SOLVENT",  # treat imputed waters same as real for partner labeling
}

PROTEIN_BACKBONE_HBOND_ATOMS = {"N", "O"}  # CA, C are not H-bond partners; N=donor, O=acceptor

NUCLEIC_BACKBONE_ATOMS = set(const.nucleic_backbone_atom_names)

# Explicit donor / acceptor / both for protein sidechain atoms
PROTEIN_SIDECHAIN_ROLES: dict[tuple[str, str], str] = {
    # donors
    ("LYS", "NZ"):  "donor",
    ("ARG", "NH1"): "donor", ("ARG", "NH2"): "donor", ("ARG", "NE"): "donor",
    ("TRP", "NE1"): "donor",
    ("ASN", "ND2"): "donor",
    ("GLN", "NE2"): "donor",
    # acceptors
    ("ASP", "OD1"): "acceptor", ("ASP", "OD2"): "acceptor",
    ("GLU", "OE1"): "acceptor", ("GLU", "OE2"): "acceptor",
    ("ASN", "OD1"): "acceptor",
    ("GLN", "OE1"): "acceptor",
    ("MET", "SD"):  "acceptor",
    # hydroxyl / ambiguous (both donor and acceptor)
    ("SER", "OG"):  "both",
    ("THR", "OG1"): "both",
    ("TYR", "OH"):  "both",
    ("CYS", "SG"):  "both",
    # histidine tautomer-dependent — report as both
    ("HIS", "ND1"): "both", ("HIS", "NE2"): "both",
}

# Nucleic acid base H-bond roles (standard residue names in the NPZ format)
NUCLEIC_BASE_ROLES: dict[tuple[str, str], str] = {
    # Adenine (DNA=DA, RNA=RA)
    ("DA", "N6"): "donor",    ("RA", "N6"): "donor",
    ("DA", "N1"): "acceptor", ("RA", "N1"): "acceptor",
    ("DA", "N3"): "acceptor", ("RA", "N3"): "acceptor",
    ("DA", "N7"): "acceptor", ("RA", "N7"): "acceptor",
    # Guanine
    ("DG", "N1"): "donor",    ("RG", "N1"): "donor",
    ("DG", "N2"): "donor",    ("RG", "N2"): "donor",
    ("DG", "O6"): "acceptor", ("RG", "O6"): "acceptor",
    ("DG", "N7"): "acceptor", ("RG", "N7"): "acceptor",
    ("DG", "N3"): "acceptor", ("RG", "N3"): "acceptor",
    # Cytosine
    ("DC", "N4"): "donor",    ("RC", "N4"): "donor",
    ("DC", "N3"): "acceptor", ("RC", "N3"): "acceptor",
    ("DC", "O2"): "acceptor", ("RC", "O2"): "acceptor",
    # Thymine (DNA only)
    ("DT", "N3"): "donor",
    ("DT", "O2"): "acceptor", ("DT", "O4"): "acceptor",
    # Uracil (RNA only)
    ("RU", "N3"): "donor",
    ("RU", "O2"): "acceptor", ("RU", "O4"): "acceptor",
    # RNA 2'-OH (ribose) — both donor and acceptor
    ("RA", "O2'"): "both", ("RG", "O2'"): "both",
    ("RC", "O2'"): "both", ("RU", "O2'"): "both",
}

HBOND_MIN_DIST = const.hbond_min_dist  # 2.5 Å
HBOND_MAX_DIST = const.hbond_max_dist  # 3.5 Å

# ---------------------------------------------------------------------------
# Tier 1: atom-level classification helpers
# ---------------------------------------------------------------------------

def get_mol_category(mol_type_int: int) -> str:
    """Map a chain mol_type integer to a human-readable category string."""
    return MOL_CATEGORY_MAP.get(int(mol_type_int), "UNKNOWN")


def classify_backbone_vs_sidechain(atom_name: str, mol_category: str) -> str:
    """
    Return 'backbone', 'sidechain', 'base', or 'N/A'.

    For protein: backbone = {N, CA, C, O}; only N and O are H-bond capable.
    For DNA/RNA: backbone = nucleic_backbone_atom_names set.
    """
    if mol_category == "PROTEIN":
        pbb = set(const.protein_backbone_atom_names)
        return "backbone" if atom_name in pbb else "sidechain"
    elif mol_category in ("DNA", "RNA"):
        return "backbone" if atom_name in NUCLEIC_BACKBONE_ATOMS else "base"
    else:
        return "N/A"


def get_hbond_role(res_name: str, atom_name: str, mol_category: str, bb_sc: str) -> str:
    """
    Classify the H-bond role of a partner atom as 'donor', 'acceptor', 'both', or 'unknown'.

    Uses residue + atom name lookups for protein and nucleic acid atoms.
    Ligand atoms are labeled 'unknown' (would need CCD for reliable assignment).
    Water oxygens are 'both'.
    """
    if mol_category == "PROTEIN":
        if bb_sc == "backbone":
            if atom_name == "N":
                return "donor"    # amide N–H
            elif atom_name == "O":
                return "acceptor" # carbonyl O
            else:
                return "unknown"  # CA, C not H-bond capable
        else:
            return PROTEIN_SIDECHAIN_ROLES.get((res_name, atom_name), "unknown")

    elif mol_category in ("DNA", "RNA"):
        if bb_sc == "backbone":
            # Phosphate oxygens are pure acceptors; ribose oxygens are acceptors
            return "acceptor" if atom_name.startswith("O") else "unknown"
        else:
            return NUCLEIC_BASE_ROLES.get((res_name, atom_name), "unknown")

    elif mol_category == "SOLVENT":
        return "both"  # water O is both donor and acceptor

    else:
        # NONPOLYMER / UNKNOWN: ligand chemistry too diverse to infer without CCD
        return "unknown"


# ---------------------------------------------------------------------------
# Tier 2: build per-atom lookup arrays from a structure
# ---------------------------------------------------------------------------

def build_atom_to_residue_map(structure: Structure) -> np.ndarray:
    """Return array of length n_atoms mapping each atom index to its residue index."""
    atom_to_residue = np.full(len(structure.atoms), -1, dtype=np.int64)
    for res_idx, res in enumerate(structure.residues):
        atom_start = res["atom_idx"]
        atom_end = atom_start + res["atom_num"]
        atom_to_residue[atom_start:atom_end] = res_idx
    return atom_to_residue


def build_atom_to_chain_map(structure: Structure) -> np.ndarray:
    """Return array of length n_atoms mapping each atom index to its chain index."""
    atom_to_chain = np.full(len(structure.atoms), -1, dtype=np.int64)
    for chain_idx, chain in enumerate(structure.chains):
        atom_start = chain["atom_idx"]
        atom_end = atom_start + chain["atom_num"]
        atom_to_chain[atom_start:atom_end] = chain_idx
    return atom_to_chain


def get_hbond_candidate_info(structure: Structure) -> dict:
    """
    Build a dict of parallel arrays for all N/O/F atoms present in the structure.

    These are the atoms that can participate in H-bonds with water.
    Includes protein, nucleic, ligand, AND other water atoms — mol_category
    labels distinguish them, so callers can filter downstream.

    Keys: atom_indices, coords, atom_names, res_names, mol_categories, bb_sc, roles.
    """
    atom_to_residue = build_atom_to_residue_map(structure)
    atom_to_chain = build_atom_to_chain_map(structure)

    hbond_mask = (
        (
            np.char.startswith(structure.atoms["name"], "N")
            | np.char.startswith(structure.atoms["name"], "O")
            | np.char.startswith(structure.atoms["name"], "F")
        )
        & structure.atoms["is_present"]
    )

    atom_indices = np.flatnonzero(hbond_mask)
    atom_names = structure.atoms["name"][atom_indices]
    coords = structure.coords["coords"][atom_indices]

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
    roles = np.array([
        get_hbond_role(rn, an, mc, bs)
        for rn, an, mc, bs in zip(res_names, atom_names, mol_categories, bb_sc)
    ])

    return {
        "atom_indices": atom_indices,
        "coords": coords,
        "atom_names": atom_names,
        "res_names": res_names,
        "mol_categories": mol_categories,
        "bb_sc": bb_sc,
        "roles": roles,
    }


# ---------------------------------------------------------------------------
# Tier 3: find partners and build descriptors
# ---------------------------------------------------------------------------

def find_hbond_partners(
    water_coords: np.ndarray,
    candidate_info: dict,
    min_dist: float = HBOND_MIN_DIST,
    max_dist: float = HBOND_MAX_DIST,
) -> list[list[int]]:
    """
    For each water coordinate, return a list of indices into candidate_info arrays
    for atoms within [min_dist, max_dist].

    The index is a position in the candidate arrays, NOT the original atom index.
    """
    if len(water_coords) == 0 or len(candidate_info["coords"]) == 0:
        return [[] for _ in range(len(water_coords))]

    tree = cKDTree(candidate_info["coords"])
    within_max = tree.query_ball_point(water_coords, r=max_dist)

    partners: list[list[int]] = []
    for water_pos, neighbors in zip(water_coords, within_max):
        valid = [
            n for n in neighbors
            if np.linalg.norm(water_pos - candidate_info["coords"][n]) >= min_dist
        ]
        partners.append(valid)
    return partners


def describe_partner(candidate_info: dict, candidate_idx: int) -> dict:
    """Return a flat descriptor dict for one partner atom (by its candidate array index)."""
    return {
        "mol_category": str(candidate_info["mol_categories"][candidate_idx]),
        "res_name":     str(candidate_info["res_names"][candidate_idx]),
        "atom_name":    str(candidate_info["atom_names"][candidate_idx]),
        "bb_sc":        str(candidate_info["bb_sc"][candidate_idx]),
        "role":         str(candidate_info["roles"][candidate_idx]),
    }


def get_water_partner_descriptors(
    water_coords: np.ndarray,
    candidate_info: dict,
    min_dist: float = HBOND_MIN_DIST,
    max_dist: float = HBOND_MAX_DIST,
) -> list[list[dict]]:
    """
    For each water, return a list of partner descriptor dicts.

    Output: one list per water, each list containing one dict per H-bond partner.
    """
    partners_by_water = find_hbond_partners(water_coords, candidate_info, min_dist, max_dist)
    return [
        [describe_partner(candidate_info, idx) for idx in partner_indices]
        for partner_indices in partners_by_water
    ]


def flatten_descriptors(descriptors_by_water: list[list[dict]]) -> list[dict]:
    """Flatten the per-water list-of-lists into a single list of partner dicts."""
    return [d for water_partners in descriptors_by_water for d in water_partners]


# ---------------------------------------------------------------------------
# Tier 4: structure loading and preparation
# ---------------------------------------------------------------------------

def load_gt_structure(pdb_id: str, npz_root: Path = NPZ_ROOT) -> Structure:
    """Load a GT structure and normalize to one solvent atom per chain."""
    npz_path = npz_root / f"{pdb_id.lower()}.npz"
    structure = Structure.load(npz_path)
    return structure.to_one_solvent_per_chain(structure)


def filter_to_min_hbonds(structure: Structure, min_hbonds: int) -> Structure:
    """Keep only solvent chains with at least min_hbonds H-bond partners."""
    return gloria_remove_weak_solvents(structure, min_hbonds=min_hbonds)


def build_imputed_structure(
    gt_structure: Structure,
    max_hbond_length: float,
    protein_clash_rad: float,
    solvent_clash_rad: float,
) -> Structure:
    """
    Strip waters from gt_structure, impute new ones geometrically, then filter clashes.

    The returned Structure has iSOLVENT chains for the surviving imputed waters.
    """
    stripped = gt_structure.remove_solvents()
    imputed = impute_solvents_from_atom_triples(stripped, max_hbond_length=max_hbond_length)
    return filter_solvent_clashes(
        imputed,
        protein_clash_rad=protein_clash_rad,
        solvent_clash_rad=solvent_clash_rad,
    )


def extract_solvent_coords(
    structure: Structure,
    mol_types: tuple[int, ...] = (
        const.chain_type_ids["SOLVENT"],
        const.chain_type_ids["iSOLVENT"],
    ),
) -> np.ndarray:
    """
    Extract one coordinate per solvent chain, filtered to the given mol_types.

    Default includes both real (SOLVENT) and imputed (iSOLVENT) waters.
    Pass mol_types=(const.chain_type_ids["SOLVENT"],) to get only real waters,
    or (const.chain_type_ids["iSOLVENT"],) for only imputed.
    """
    atom_coords = structure.coords["coords"]
    atom_present = structure.atoms["is_present"]
    indices = []

    for chain in structure.chains:
        if chain["mol_type"] not in mol_types:
            continue
        atom_start = chain["atom_idx"]
        atom_end = atom_start + chain["atom_num"]
        present_offsets = np.flatnonzero(atom_present[atom_start:atom_end])
        if len(present_offsets) > 0:
            indices.append(atom_start + present_offsets[0])

    if not indices:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(atom_coords[np.asarray(indices, dtype=np.int64)], dtype=float)


# ---------------------------------------------------------------------------
# Tier 5: per-PDB analysis runner
# ---------------------------------------------------------------------------

def analyze_one_pdb(
    pdb_id: str,
    min_hbonds: int = 3,
    max_hbond_length: float = 3.5,
    protein_clash_rad: float = 2.0,
    solvent_clash_rad: float = 2.0,
    npz_root: Path = NPZ_ROOT,
    verbose: bool = False,
) -> dict:
    """
    Run the full H-bond partner analysis for one PDB entry.

    Returns a dict with:
      - pdb_id
      - real_descriptors:    list[list[dict]]  (one per real ≥min_hbonds water)
      - imputed_descriptors: list[list[dict]]  (one per surviving imputed water)
      - n_real, n_imputed
    """
    gt = load_gt_structure(pdb_id, npz_root)

    # --- real waters: filter to >= min_hbonds ---
    real_gt = filter_to_min_hbonds(gt, min_hbonds)
    real_coords = extract_solvent_coords(real_gt, mol_types=(const.chain_type_ids["SOLVENT"],))

    # Build candidate info from the stripped structure (non-water atoms only),
    # so partner labels reflect protein/ligand contacts and not water-water H-bonds.
    real_candidate_info = get_hbond_candidate_info(gt.remove_solvents())
    real_descriptors = get_water_partner_descriptors(real_coords, real_candidate_info)

    # --- imputed waters ---
    imputed = build_imputed_structure(gt, max_hbond_length, protein_clash_rad, solvent_clash_rad)
    imputed_coords = extract_solvent_coords(imputed, mol_types=(const.chain_type_ids["iSOLVENT"],))

    # Build candidate info from the stripped structure (non-water atoms only),
    # matching the basis on which imputed waters were placed.
    stripped_candidate_info = get_hbond_candidate_info(gt.remove_solvents())
    imputed_descriptors = get_water_partner_descriptors(imputed_coords, stripped_candidate_info)

    if verbose:
        print(f"{pdb_id}: {len(real_coords)} real (≥{min_hbonds} hbond), {len(imputed_coords)} imputed")

    return {
        "pdb_id": pdb_id,
        "real_descriptors": real_descriptors,
        "imputed_descriptors": imputed_descriptors,
        "n_real": len(real_coords),
        "n_imputed": len(imputed_coords),
    }


def collect_analysis_results(
    pdb_ids: list[str],
    min_hbonds: int = 3,
    max_hbond_length: float = 3.5,
    protein_clash_rad: float = 2.0,
    solvent_clash_rad: float = 2.0,
    npz_root: Path = NPZ_ROOT,
    verbose: bool = True,
) -> list[dict]:
    """
    Run analyze_one_pdb for each PDB ID and return the raw per-PDB result dicts.

    Use get_flat_descriptors and get_all_partner_counts to extract what you need
    without running imputation a second time.
    """
    results = []
    for pdb_id in pdb_ids:
        try:
            result = analyze_one_pdb(
                pdb_id,
                min_hbonds=min_hbonds,
                max_hbond_length=max_hbond_length,
                protein_clash_rad=protein_clash_rad,
                solvent_clash_rad=solvent_clash_rad,
                npz_root=npz_root,
                verbose=verbose,
            )
            results.append(result)
        except Exception as e:
            print(f"  WARNING: {pdb_id} failed — {e}")
    return results


def get_flat_descriptors(
    results: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Extract flat partner descriptor lists from collected per-PDB results."""
    real_flat = [d for r in results for d in flatten_descriptors(r["real_descriptors"])]
    imputed_flat = [d for r in results for d in flatten_descriptors(r["imputed_descriptors"])]
    return real_flat, imputed_flat


def get_partner_counts(descriptors_by_water: list[list[dict]]) -> list[int]:
    """Return the number of H-bond partners for each water in the nested list."""
    return [len(partners) for partners in descriptors_by_water]


def get_all_partner_counts(results: list[dict]) -> tuple[list[int], list[int]]:
    """Extract per-water partner counts from collected per-PDB results."""
    real_counts = [c for r in results for c in get_partner_counts(r["real_descriptors"])]
    imputed_counts = [c for r in results for c in get_partner_counts(r["imputed_descriptors"])]
    return real_counts, imputed_counts


def collect_all_descriptors(
    pdb_ids: list[str],
    min_hbonds: int = 3,
    max_hbond_length: float = 3.5,
    protein_clash_rad: float = 2.0,
    solvent_clash_rad: float = 2.0,
    npz_root: Path = NPZ_ROOT,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Convenience wrapper: run collect_analysis_results and return flat descriptor lists."""
    results = collect_analysis_results(
        pdb_ids, min_hbonds, max_hbond_length, protein_clash_rad, solvent_clash_rad,
        npz_root, verbose,
    )
    return get_flat_descriptors(results)


# ---------------------------------------------------------------------------
# Tier 6: plotting helpers
# ---------------------------------------------------------------------------

def _extract_field(descriptors: list[dict], field: str) -> list[str]:
    return [d[field] for d in descriptors]


def plot_hbond_count_histogram(
    real_counts: list[int],
    imputed_counts: list[int],
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """
    Grouped bar chart of per-water H-bond partner counts, real vs imputed side by side.

    real_counts / imputed_counts: one integer per water (output of get_partner_counts
    or get_all_partner_counts). Bars are normalized to fraction so the two populations
    are comparable regardless of size.
    """
    all_counts = real_counts + imputed_counts
    if not all_counts:
        fig, ax = plt.subplots(figsize=figsize)
        return fig

    lo, hi = min(all_counts), max(all_counts)
    x = np.arange(lo, hi + 1)
    width = 0.35

    def _fracs(counts):
        total = len(counts)
        arr = np.array(counts)
        return np.array([(arr == v).sum() / total for v in x]) if total > 0 else np.zeros(len(x))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, _fracs(real_counts),    width, label=f"real ≥3 hbonds (n={len(real_counts)} waters)",    color="steelblue", alpha=0.85)
    ax.bar(x + width / 2, _fracs(imputed_counts),  width, label=f"imputed (n={len(imputed_counts)} waters)", color="tomato",    alpha=0.85)
    ax.set_xticks(x)
    ax.set_xlabel("number of H-bond partners per water")
    ax.set_ylabel("fraction of waters")
    ax.set_title("H-bond partner count distribution: real vs imputed")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_distribution_comparison(
    real_vals: list[str],
    imputed_vals: list[str],
    title: str,
    xlabel: str,
    top_n: int | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """
    Side-by-side normalized bar chart comparing real vs imputed partner distributions.

    If top_n is set, only the top_n most common categories (by combined count) are shown.
    """
    all_vals = real_vals + imputed_vals
    categories, counts = np.unique(all_vals, return_counts=True)
    order = np.argsort(-counts)
    categories = categories[order]
    if top_n is not None:
        categories = categories[:top_n]

    def _norm_counts(vals, cats):
        total = len(vals)
        if total == 0:
            return np.zeros(len(cats))
        val_arr = np.array(vals)
        return np.array([(val_arr == c).sum() / total for c in cats])

    real_fracs = _norm_counts(real_vals, categories)
    imputed_fracs = _norm_counts(imputed_vals, categories)

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, real_fracs, width, label=f"real ≥3 hbonds (n={len(real_vals)})", color="steelblue", alpha=0.85)
    ax.bar(x + width / 2, imputed_fracs, width, label=f"imputed (n={len(imputed_vals)})", color="tomato", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("fraction of all partner contacts")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_all_distributions(
    real_descriptors: list[dict],
    imputed_descriptors: list[dict],
    top_n_res: int = 20,
) -> list[plt.Figure]:
    """
    Plot distributions for mol_category, bb_sc, role, and res_name.
    Returns list of figures so the caller can save or display them.
    """
    figs = []

    figs.append(plot_distribution_comparison(
        _extract_field(real_descriptors, "mol_category"),
        _extract_field(imputed_descriptors, "mol_category"),
        title="H-bond partner molecule type",
        xlabel="mol_category",
    ))

    figs.append(plot_distribution_comparison(
        _extract_field(real_descriptors, "bb_sc"),
        _extract_field(imputed_descriptors, "bb_sc"),
        title="H-bond partner backbone vs sidechain",
        xlabel="bb_sc",
    ))

    figs.append(plot_distribution_comparison(
        _extract_field(real_descriptors, "role"),
        _extract_field(imputed_descriptors, "role"),
        title="H-bond partner chemical role",
        xlabel="role (donor / acceptor / both / unknown)",
    ))

    figs.append(plot_distribution_comparison(
        _extract_field(real_descriptors, "res_name"),
        _extract_field(imputed_descriptors, "res_name"),
        title=f"H-bond partner residue type (top {top_n_res})",
        xlabel="res_name",
        top_n=top_n_res,
    ))

    return figs
