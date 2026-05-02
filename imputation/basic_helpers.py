from pathlib import Path



def resolve_npz_path(pdb_id: str, npz_root: Path, npz_path: str | None = None) -> Path:
    if npz_path:
        return Path(npz_path)
    return Path(npz_root) / f"{pdb_id.lower()}.npz"