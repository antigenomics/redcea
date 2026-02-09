from __future__ import annotations
import logging
import numpy as np


class UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        p = self.parent[x]
        while p != self.parent[p]:
            p = self.parent[p]
        while x != p:
            nxt = self.parent[x]
            self.parent[x] = p
            x = nxt
        return p

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def vdbscan_from_knn(
    knn_indices: np.ndarray,
    knn_distances_l2: np.ndarray,
    eps_i_l2: np.ndarray,
    num_points_for_core: int,
    sym_rule: str = "asymmetric",
) -> np.ndarray:
    """
    vDBSCAN from kNN lists.

    Assumptions:
    - knn_distances are L2 (NOT squared)
    - eps_i is L2 (NOT squared)
    - No silent conversions.

    Core:
      core(i) if count{j: dist(i,j) <= eps_i[i]} >= num_points_for_core

    Core connectivity:
      sym_rule:
        - "asymmetric": threshold = eps_i[i]
        - "max":        threshold = max(eps_i[i], eps_i[j])
        - "min":        threshold = min(eps_i[i], eps_i[j])

    Border:
      assign p to nearest core c such that dist(p,c) <= eps_i[c]
      else noise (-1)
    """
    if sym_rule not in {"asymmetric", "max", "min"}:
        raise ValueError("sym_rule must be one of: asymmetric, max, min")

    if knn_indices.ndim != 2 or knn_distances_l2.ndim != 2:
        raise ValueError("knn_indices and knn_distances_l2 must be 2D arrays")
    if knn_indices.shape != knn_distances_l2.shape:
        raise ValueError("knn_indices and knn_distances_l2 must have the same shape")

    N, K = knn_indices.shape
    eps_i_l2 = np.asarray(eps_i_l2, dtype=np.float32)
    if eps_i_l2.shape != (N,):
        raise ValueError(f"eps_i_l2 must have shape (N,), got {eps_i_l2.shape} for N={N}")

    if not np.isfinite(knn_distances_l2).all():
        raise ValueError("knn_distances_l2 contains NaN/inf")
    if not np.isfinite(eps_i_l2).all():
        raise ValueError("eps_i_l2 contains NaN/inf")
    if (eps_i_l2 < 0).any():
        raise ValueError("eps_i_l2 contains negative values")

    # 1) core mask
    core_counts = np.zeros(N, dtype=np.int32)
    for i in range(N):
        core_counts[i] = int(np.sum(knn_distances_l2[i] <= eps_i_l2[i]))

    is_core = core_counts >= int(num_points_for_core)
    logging.info("vDBSCAN: core points = %d / %d", int(is_core.sum()), N)

    # 2) union-find among core points
    uf = UnionFind(N)
    for i in range(N):
        if not is_core[i]:
            continue
        ei = float(eps_i_l2[i])
        neighs = knn_indices[i]
        dists = knn_distances_l2[i]
        for j, dij in zip(neighs, dists):
            j = int(j)
            if j < 0 or j == i or j >= N:
                continue
            if not is_core[j]:
                continue

            if sym_rule == "asymmetric":
                thr = ei
            elif sym_rule == "max":
                thr = max(ei, float(eps_i_l2[j]))
            else:
                thr = min(ei, float(eps_i_l2[j]))

            if float(dij) <= thr:
                uf.union(i, j)

    # 3) label core components
    labels = -np.ones(N, dtype=np.int64)
    root_to_cid: dict[int, int] = {}
    next_cid = 0
    for i in range(N):
        if not is_core[i]:
            continue
        r = uf.find(i)
        cid = root_to_cid.get(r)
        if cid is None:
            cid = next_cid
            root_to_cid[r] = cid
            next_cid += 1
        labels[i] = cid

    logging.info("vDBSCAN: core components = %d", next_cid)

    # 4) border assignment
    assigned = 0
    for p in range(N):
        if is_core[p]:
            continue

        best_c = -1
        best_d = float("inf")

        neighs = knn_indices[p]
        dists = knn_distances_l2[p]
        for j, dpj in zip(neighs, dists):
            j = int(j)
            if j < 0 or j == p or j >= N:
                continue
            if not is_core[j]:
                continue
            if float(dpj) <= float(eps_i_l2[j]) and float(dpj) < best_d:
                best_d = float(dpj)
                best_c = j

        if best_c >= 0:
            labels[p] = labels[best_c]
            assigned += 1

    logging.info("vDBSCAN: border assigned=%d, noise=%d", assigned, int(np.sum(labels < 0)))
    return labels
