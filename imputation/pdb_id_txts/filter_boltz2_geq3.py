"""
Filter boltz2.txt to keep only PDB IDs with >= 3 solvents after the geq3
H-bond filter (gloria_remove_weak_solvents with min_hbonds=3).

Writes passing IDs to pdb_id_txts/boltz2_geq3.txt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from boltzgen.data.data import Structure
from basic_helpers import NPZ_ROOT, resolve_npz_path
from gloria_hbond_helpers import gloria_remove_weak_solvents

INPUT_TXT = Path(__file__).parent / "pdb_id_txts/boltz2.txt"
OUTPUT_TXT = Path(__file__).parent / "pdb_id_txts/boltz2_geq3.txt"
MIN_HBONDS = 3
MIN_SOLVENTS = 3


def apply_geq3_filter(pdb_id: str) -> int | None:
    npz_path = resolve_npz_path(pdb_id, NPZ_ROOT)
    if not npz_path.exists():
        return None
    structure = Structure.load(npz_path)
    structure = structure.to_one_solvent_per_chain(structure)
    filtered = gloria_remove_weak_solvents(structure, min_hbonds=MIN_HBONDS)
    return filtered.count_solvents()


def main():
    pdb_ids = INPUT_TXT.read_text().splitlines()
    pdb_ids = [x.strip() for x in pdb_ids if x.strip()]
    print(f"Input: {len(pdb_ids)} IDs")

    passed, failed_filter, missing = [], [], []
    for i, pdb_id in enumerate(pdb_ids):
        if i % 50 == 0:
            print(f"  {i}/{len(pdb_ids)} processed...")
        n = apply_geq3_filter(pdb_id)
        if n is None:
            missing.append(pdb_id)
        elif n >= MIN_SOLVENTS:
            passed.append(pdb_id)
        else:
            failed_filter.append((pdb_id, n))

    OUTPUT_TXT.write_text("\n".join(passed) + "\n")
    print(f"\nResults:")
    print(f"  Passed (>={MIN_SOLVENTS} solvents after geq3 filter): {len(passed)}")
    print(f"  Filtered out (<{MIN_SOLVENTS} solvents):              {len(failed_filter)}")
    print(f"  Missing .npz:                                          {len(missing)}")
    print(f"\nOutput written to: {OUTPUT_TXT}")
    if missing:
        print(f"Missing IDs: {missing[:10]}{'...' if len(missing) > 10 else ''}")


if __name__ == "__main__":
    main()
