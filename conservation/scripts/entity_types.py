"""
Batch-query RCSB GraphQL to determine the molecule types present in each PDB entry.

For each entry we record:
  - has_protein      : at least one polypeptide entity
  - has_dna          : at least one polydeoxyribonucleotide (or hybrid) entity
  - has_rna          : at least one polyribonucleotide (or hybrid) entity
  - has_ligand       : at least one non-polymer entity (small molecules, ions)
  - has_sugar        : at least one branched entity (glycans)
  - poly_types       : sorted list of distinct polymer entity types

Results are cached to entity_types_cache.json so re-runs are instant.
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any

CONSERVATION_DIR = Path("/data/rbg/users/gloriama/dev/water_conservation/conservation")
CACHE_FILE = CONSERVATION_DIR / "entity_types_cache.json"

GRAPHQL_URL = "https://data.rcsb.org/graphql"
BATCH_SIZE = 500
RETRY_WAIT = 5  # seconds between retries

QUERY_TEMPLATE = """
{{
  entries(entry_ids: [{ids}]) {{
    rcsb_id
    polymer_entities {{
      entity_poly {{
        type
      }}
    }}
    nonpolymer_entities {{
      rcsb_nonpolymer_entity {{
        pdbx_description
      }}
    }}
    branched_entities {{
      rcsb_branched_entity {{
        pdbx_description
      }}
    }}
  }}
}}
"""

_DNA_TYPES = {
    "polydeoxyribonucleotide",
    "polydeoxyribonucleotide/polyribonucleotide hybrid",
}
_RNA_TYPES = {
    "polyribonucleotide",
    "polydeoxyribonucleotide/polyribonucleotide hybrid",
}


def _query_batch(pdb_ids: list[str]) -> dict[str, Any]:
    """Query RCSB GraphQL for a batch of PDB IDs (uppercase). Returns raw data dict."""
    ids_str = ", ".join(f'"{i.upper()}"' for i in pdb_ids)
    query = QUERY_TEMPLATE.format(ids=ids_str)
    payload = json.dumps({"query": query}).encode()
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(RETRY_WAIT * (attempt + 1))


def _parse_entry(entry: dict) -> dict:
    poly_types = sorted({
        pe["entity_poly"]["type"]
        for pe in (entry.get("polymer_entities") or [])
        if pe.get("entity_poly") and pe["entity_poly"].get("type")
    })
    has_protein = any("polypeptide" in t for t in poly_types)
    has_dna = any(t in _DNA_TYPES for t in poly_types)
    has_rna = any(t in _RNA_TYPES for t in poly_types)
    has_ligand = bool(entry.get("nonpolymer_entities"))
    has_sugar = bool(entry.get("branched_entities"))
    return {
        "has_protein": has_protein,
        "has_dna": has_dna,
        "has_rna": has_rna,
        "has_ligand": has_ligand,
        "has_sugar": has_sugar,
        "poly_types": poly_types,
    }


def fetch_entity_types(pdb_ids: list[str], verbose: bool = True) -> dict[str, dict]:
    """
    Fetch entity type info for a list of lowercase PDB IDs.
    Reads from cache if available; queries RCSB and updates cache otherwise.
    Returns dict[pdb_id_lower -> info_dict].
    """
    # Load existing cache
    cache: dict[str, dict] = {}
    if CACHE_FILE.exists():
        cache = json.load(open(CACHE_FILE))

    missing = [p for p in pdb_ids if p.lower() not in cache]
    if verbose:
        print(f"Cache hits: {len(pdb_ids) - len(missing)}  |  To fetch: {len(missing)}")

    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i: i + BATCH_SIZE]
        if verbose:
            print(f"  Fetching batch {i // BATCH_SIZE + 1} / {-(-len(missing) // BATCH_SIZE)} ...")
        data = _query_batch(batch)
        for entry in (data.get("data") or {}).get("entries") or []:
            key = entry["rcsb_id"].lower()
            cache[key] = _parse_entry(entry)
        time.sleep(0.1)  # be polite to RCSB

    # Save updated cache
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

    return {p.lower(): cache.get(p.lower(), {}) for p in pdb_ids}


def classify_entry(info: dict) -> str:
    """Return a human-readable label for the dominant composition of an entry."""
    if not info:
        return "unknown"
    parts = []
    if info.get("has_protein"):
        parts.append("protein")
    if info.get("has_dna"):
        parts.append("DNA")
    if info.get("has_rna"):
        parts.append("RNA")
    if info.get("has_sugar"):
        parts.append("sugar/glycan")
    if info.get("has_ligand"):
        parts.append("ligand")
    return " + ".join(parts) if parts else "unknown"
