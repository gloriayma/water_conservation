# Results — Water Conservation Pipeline

Clean numerical results only. For methodology and exploration notes, see EXPLORATION_LOG.md.

---

## Dataset scale (as of 2026-05-04)

| | Count |
|--|--|
| Total structures in npz dataset | 238,857 |
| X-ray structures | 195,210 |
| X-ray, ≤2.0 Å, with waters | 97,838 |
| X-ray, ≤2.5 Å, with waters | 152,340 |

---

## Cluster filtering (RCSB 100% sequence identity)

### At 2.0 Å resolution cutoff, min_cluster_size=5

| Stage | Count |
|-------|-------|
| Total RCSB clusters | 1,049,376 |
| Clusters with ≥1 qualifying PDB | 46,646 |
| **Clusters with ≥5 qualifying PDbs** | **2,894** |

Cluster size distribution (among clusters with ≥1 qualifying PDB):

| Size range | # clusters |
|-----------|------------|
| 1 | 33,443 |
| 2–4 | 10,309 |
| 5–9 | 1,768 |
| 10–19 | 645 |
| 20–49 | 328 |
| 50–99 | 99 |
| 100–199 | 32 |
| 200–499 | 16 |
| 500–999 | 4 |
| ≥1000 | 2 |

### At 2.5 Å resolution cutoff, min_cluster_size=5

| Stage | Count |
|-------|-------|
| Clusters with ≥1 qualifying PDB | 71,378 |
| **Clusters with ≥5 qualifying PDbs** | **5,230** |

Cluster size distribution (among clusters with ≥1 qualifying PDB):

| Size range | # clusters |
|-----------|------------|
| 1 | 48,741 |
| 2–4 | 17,407 |
| 5–9 | 3,309 |
| 10–19 | 1,130 |
| 20–49 | 534 |
| 50–99 | 176 |
| 100–199 | 47 |
| 200–499 | 26 |
| 500–999 | 6 |
| ≥1000 | 2 |

---

## Schema fields availability

| Field | Available in npz |
|-------|-----------------|
| Water oxygen coordinates | ✅ |
| B-factors | ✅ |
| Occupancy | ❌ |
| Alt-loc identifiers | ❌ |
| Unit cell / space group | ❌ |
