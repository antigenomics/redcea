# tcrempnet_vdbscan.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import faiss

# reuse your existing "knee-based" eps estimator
from tcremp.tcremp_cluster import estimate_dbscan_eps


# ============================================================
# Bucketing by cumulative length fractions (simple boundaries)
# ============================================================

@dataclass(frozen=True)
class LengthBucket:
    lengths: Tuple[int, ...]
    indices: np.ndarray
    frac: float


def make_length_buckets(lengths: np.ndarray, min_group_frac: float = 0.05) -> List[LengthBucket]:
    """
    Build buckets by cumulative fraction over sorted lengths:
      - collect counts per length
      - walk lengths in increasing order accumulating until >= min_group_frac
      - close bucket, reset accumulator
      - last bucket absorbs remaining tail
    """
    n = int(len(lengths))
    if n == 0:
        return []

    uniq, counts = np.unique(lengths, return_counts=True)
    order = np.argsort(uniq)
    uniq = uniq[order].astype(int)
    counts = counts[order].astype(int)

    fracs = counts / n

    # boundaries are indices in `uniq` where bucket ends (inclusive)
    boundaries: List[int] = []
    cum = 0.0
    for i, f in enumerate(fracs):
        cum += float(f)
        if cum >= min_group_frac:
            boundaries.append(i)
            cum = 0.0

    # ensure tail included
    if not boundaries or boundaries[-1] != (len(uniq) - 1):
        boundaries.append(len(uniq) - 1)

    # precompute length -> indices once
    length_to_idx: Dict[int, np.ndarray] = {L: np.where(lengths == L)[0] for L in uniq}

    buckets: List[LengthBucket] = []
    start = 0
    for end in boundaries:
        bucket_lengths = tuple(uniq[start : end + 1].tolist())
        idx = np.concatenate([length_to_idx[L] for L in bucket_lengths], axis=0)
        buckets.append(LengthBucket(bucket_lengths, idx, len(idx) / n))
        start = end + 1

    return buckets


# ============================================================
# Symmetric variable-radius neighborhood via FAISS range_search
# ============================================================

def _neighbors_symmetric_range_search(
    X: np.ndarray,
    buckets: List[LengthBucket],
    bucket_eps: List[float],
) -> List[np.ndarray]:
    """
    Symmetric neighborhood:
      i--j exists iff dist(i,j) <= eps_i AND dist(i,j) <= eps_j.
    Implementation:
      - query points in each bucket with radius eps_bucket (candidates)
      - filter candidate j by dist <= eps_j
    """
    n, d = X.shape
    X32 = np.asarray(X, dtype=np.float32, order="C")

    eps_per_point = np.zeros(n, dtype=np.float32)
    for b, eps in zip(buckets, bucket_eps):
        eps_per_point[b.indices] = float(eps)

    index = faiss.IndexFlatL2(d)
    index.add(X32)

    neigh_lists: List[List[int]] = [[] for _ in range(n)]

    for bi, (bucket, eps) in enumerate(zip(buckets, bucket_eps)):
        idx = bucket.indices
        if len(idx) == 0 or eps <= 0:
            continue

        radius2 = float(eps) ** 2
        Q = X32[idx]
        lims, D, I = index.range_search(Q, radius2)

        for q in range(len(idx)):
            i = int(idx[q])
            s = int(lims[q])
            e = int(lims[q + 1])
            if e <= s:
                continue

            cand_I = I[s:e]
            cand_D = D[s:e]  # squared L2 distances

            eps_j2 = (eps_per_point[cand_I] ** 2).astype(np.float32, copy=False)
            mask = (cand_I != i) & (cand_D <= eps_j2)

            if np.any(mask):
                neigh_lists[i].extend(cand_I[mask].astype(int).tolist())

    neighbors: List[np.ndarray] = []
    for i in range(n):
        if neigh_lists[i]:
            neighbors.append(np.fromiter(set(neigh_lists[i]), dtype=np.int64))
        else:
            neighbors.append(np.empty(0, dtype=np.int64))
    return neighbors


# ============================================================
# vDBSCAN main
# ============================================================

def variable_eps_dbscan(
    X: np.ndarray,
    cdr3_lengths: np.ndarray,
    knn_distances: Optional[np.ndarray] = None,
    min_group_frac: float = 0.05,
    min_samples: int = 5,
    # reuse precomputed kNN distances if available
    eps_k: int = 4,
    eps_quantile: float = 0.05,
    eps_poly_degree: int = 10,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Variable-eps DBSCAN:
      - bucket points by length using cumulative boundaries
      - estimate eps per bucket using existing estimate_dbscan_eps
        * preferred: from precomputed knn_distances[:, eps_k-1]
        * fallback: compute from X[bucket] (avoid if you already have knn_distances)
      - build symmetric neighborhoods via FAISS range_search
      - expand clusters like classic DBSCAN (min_samples)
    """
    X = np.asarray(X)
    n = int(X.shape[0])
    if n == 0:
        return np.empty(0, dtype=np.int64), {"buckets": [], "bucket_eps": []}

    buckets = make_length_buckets(cdr3_lengths, min_group_frac=min_group_frac)
    print(f'made buckets {buckets}')

    # --- estimate eps per bucket (reuse knn_distances if given) ---
    bucket_eps: List[float] = []
    for b in buckets:
        if len(b.indices) < max(min_samples, eps_k):
            bucket_eps.append(0.0)
            continue

        if knn_distances is not None:
            if knn_distances.shape[0] != n:
                raise ValueError("knn_distances must have same N rows as X.")
            if knn_distances.shape[1] < eps_k:
                raise ValueError(f"knn_distances K={knn_distances.shape[1]} < eps_k={eps_k}.")

            kth = knn_distances[b.indices, eps_k - 1]
            eps = estimate_dbscan_eps(
                data=None,
                distances=kth,
                n_neighbors=eps_k,
                quantile=eps_quantile,
                poly_degree=eps_poly_degree,
            )
        else:
            # fallback (slower): estimate from points directly
            eps = estimate_dbscan_eps(
                data=X[b.indices],
                distances=None,
                n_neighbors=eps_k,
                quantile=eps_quantile,
                poly_degree=eps_poly_degree,
            )

        bucket_eps.append(float(eps))

    # --- neighbors ---
    neighbors = _neighbors_symmetric_range_search(X, buckets, bucket_eps)

    # --- DBSCAN expansion ---
    neigh_sizes = np.fromiter((len(nbrs) for nbrs in neighbors), dtype=np.int32, count=n)
    is_core = neigh_sizes >= (min_samples - 1)

    labels = -np.ones(n, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if not is_core[i]:
            continue

        labels[i] = cluster_id
        stack = [i]

        while stack:
            p = stack.pop()
            for q in neighbors[p]:
                q = int(q)
                if not visited[q]:
                    visited[q] = True
                    if is_core[q]:
                        stack.append(q)
                if labels[q] == -1:
                    labels[q] = cluster_id

        cluster_id += 1

    info = {
        "buckets": [
            {"lengths": b.lengths, "size": int(len(b.indices)), "frac": float(b.frac)}
            for b in buckets
        ],
        "bucket_eps": bucket_eps,
        "min_group_frac": float(min_group_frac),
        "min_samples": int(min_samples),
        "eps_k": int(eps_k),
    }

    logging.info(
        f"vDBSCAN done: clusters={len(np.unique(labels[labels>=0]))}, "
        f"noise={(labels==-1).sum()}, N={n}"
    )
    return labels, info
