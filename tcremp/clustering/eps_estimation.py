import logging
import time
from typing import Dict

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


# === copied 1:1 from your snippet ===

def estimate_dbscan_eps(
    data,
    distances=None,
    n_neighbors: int = 4,
    quantile: float = 0.05,
    poly_degree: int = 10,
):
    """
    Estimate DBSCAN eps using k-NN distances and KneeLocator.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Original data (used only if distances is None).
    distances : array-like or None
        Optional precomputed distances to k-th neighbor for each point
        (1D array). If provided, we skip NearestNeighbors fitting.
    """
    start = time.time()

    if distances is None:
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs = neigh.fit(data)
        dists, _ = nbrs.kneighbors(data)
        kth_distances = dists[:, n_neighbors - 1]
    else:
        kth_distances = np.asarray(distances)

    kth_distances = np.sort(kth_distances)
    
    tol = 1e-8
    start = np.searchsorted(kth_distances, tol, side="right")  # пропускаем все ~0
    kth_distances = kth_distances[start:]
    
    if len(kth_distances) > 20000:
        kth_distances = np.random.choice(
            kth_distances, size=20000, replace=False
        )
        kth_distances = np.sort(kth_distances)

    knee = KneeLocator(
        range(1, len(kth_distances) + 1),  # x values
        kth_distances,                     # y values
        S=1.0,
        online=False,  # disable KneeLocator online mode for better performance
        curve="convex",  # convex curve for kNN distances (increasing and convex)
        interp_method="polynomial",
        polynomial_degree=poly_degree,
        direction="increasing",
    )

    if knee.knee is not None:
        if knee.knee < 1 or knee.knee > len(kth_distances):
            logging.warning(
                f"KneeLocator returned invalid knee index {knee.knee}, returning cluster max."
            )
            eps = kth_distances[-1]
        eps = kth_distances[knee.knee]
    else:
        eps = kth_distances[int(len(kth_distances) * quantile)]

    elapsed = time.time() - start
    logging.info(f"Estimated eps for DBSCAN: {eps:.4f}, time: {elapsed:.2f} sec.")
    return eps


def cluster_dbscan(data, eps=None, min_samples=5):
    start = time.time()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data)
    elapsed = time.time() - start
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    logging.info(
        f"DBSCAN completed: clusters = {n_clusters}, noise points = {n_noise}, time: {elapsed:.2f} sec."
    )
    return labels


# === new: eps by group learned on SAMPLE only ===

def estimate_eps_by_group_from_sample(
    sample_group_id: np.ndarray,
    sample_ss_distances_l2: np.ndarray,  # shape (n_sample, k), L2 (NOT squared)
    kth_neighbor: int = 4,
    quantile: float = 0.05,
    poly_degree: int = 6,
) -> Dict[int, float]:
    """
    Estimate one eps per group using only sample-sample kNN distances (L2).
    """
    if sample_ss_distances_l2.ndim != 2:
        raise ValueError("sample_ss_distances_l2 must be 2D (n_sample, k)")
    n_sample, k = sample_ss_distances_l2.shape
    if sample_group_id.shape != (n_sample,):
        raise ValueError(f"sample_group_id must be shape (n_sample,), got {sample_group_id.shape}")

    if kth_neighbor < 1 or kth_neighbor > k:
        raise ValueError(f"kth_neighbor={kth_neighbor} must be in [1, {k}]")

    if not np.isfinite(sample_ss_distances_l2).all():
        raise ValueError("sample_ss_distances_l2 contains NaN/inf (refusing to continue)")

    eps_by_gid: Dict[int, float] = {}
    for gid in np.unique(sample_group_id):
        gid = int(gid)
        idx = np.where(sample_group_id == gid)[0]
        if idx.size < kth_neighbor:
            raise ValueError(
                f"Group {gid} too small for kth_neighbor={kth_neighbor}: size={idx.size}. "
                "Merge more (increase min_frac) or reduce kth_neighbor."
            )

        kth = sample_ss_distances_l2[idx, kth_neighbor - 1]
        eps = estimate_dbscan_eps(
                data=None,
                distances=kth,
                n_neighbors=kth_neighbor,
                quantile=quantile,
                poly_degree=poly_degree,
                   
        )
        if not np.isfinite(eps):
            raise RuntimeError(f"Non-finite eps for group {gid}: {eps}")
        else:
            logging.info(f"Estimated eps for group {gid}: {eps:.4f}")
        eps_by_gid[gid] = eps

    return eps_by_gid


def eps_per_point_from_group_id(group_id_all: np.ndarray, eps_by_gid: Dict[int, float]) -> np.ndarray:
    group_id_all = np.asarray(group_id_all, dtype=np.int32)
    eps = np.empty(group_id_all.shape[0], dtype=np.float32)
    for i, gid in enumerate(group_id_all):
        gid = int(gid)
        if gid not in eps_by_gid:
            raise KeyError(f"Missing eps for group_id={gid}")
        eps[i] = np.float32(eps_by_gid[gid])
    if not np.isfinite(eps).all():
        raise RuntimeError("eps_i contains NaN/inf after mapping")
    return eps


# === new: flexible eps estimation based on multiple data sources ===

def build_cdr3_groups_from_data(
    representations: np.ndarray,  # CDR3 lengths
    min_frac: float = 0.05,
) -> tuple:
    """
    Build CDR3 length-based groups from given data.
    
    Returns:
        (lengths, len_to_gid, gid) where:
        - lengths: original CDR3 lengths
        - len_to_gid: mapping of length -> group id
        - gid: group id per point
    """
    from tcremp.clustering.cdr3_grouping import (
        build_len_to_group_id, map_len_to_group_id
    )
    
    len_to_gid = build_len_to_group_id(representations, min_frac=min_frac)
    gid = map_len_to_group_id(representations, len_to_gid, unknown_len_policy="nearest")
    return representations, len_to_gid, gid


def estimate_eps_by_group_flexible(
    group_id: np.ndarray,
    dist_matrix: np.ndarray,  # shape (n, k), L2 distances
    kth_neighbor: int = 4,
    quantile: float = 0.05,
    poly_degree: int = 6,
) -> Dict[int, float]:
    """
    Estimate one eps per group using provided distances matrix.
    This is a generalization of estimate_eps_by_group_from_sample that works
    with any distance matrix (sample-sample, background-background, or joint).
    
    Parameters
    ----------
    group_id : np.ndarray
        Group ID for each point in the distance matrix
    dist_matrix : np.ndarray
        k-NN distances matrix, shape (n_points, k)
    kth_neighbor : int
        Which neighbor to use (1-based)
    quantile : float
        Quantile for knee estimation fallback
    poly_degree : int
        Polynomial degree for knee fitting
        
    Returns
    -------
    eps_by_gid : Dict[int, float]
        Mapping of group_id -> eps value
    """
    if dist_matrix.ndim != 2:
        raise ValueError("dist_matrix must be 2D (n_points, k)")
    
    n_points, k = dist_matrix.shape
    group_id = np.asarray(group_id, dtype=np.int32)
    
    if group_id.shape[0] != n_points:
        raise ValueError(f"group_id shape {group_id.shape[0]} doesn't match dist_matrix {n_points}")
    
    if kth_neighbor < 1 or kth_neighbor > k:
        raise ValueError(f"kth_neighbor={kth_neighbor} must be in [1, {k}]")
    
    if not np.isfinite(dist_matrix).all():
        raise ValueError("dist_matrix contains NaN/inf")
    
    eps_by_gid: Dict[int, float] = {}
    
    for gid in np.unique(group_id):
        gid = int(gid)
        idx = np.where(group_id == gid)[0]
        
        if idx.size < kth_neighbor:
            raise ValueError(
                f"Group {gid} too small for kth_neighbor={kth_neighbor}: size={idx.size}. "
                "Merge more (increase min_frac) or reduce kth_neighbor."
            )
        
        kth = dist_matrix[idx, kth_neighbor - 1]
        eps = estimate_dbscan_eps(
            data=None,
            distances=kth,
            n_neighbors=kth_neighbor,
            quantile=quantile,
            poly_degree=poly_degree,
        )
        
        if not np.isfinite(eps):
            raise RuntimeError(f"Non-finite eps for group {gid}: {eps}")
        else:
            logging.info(f"Estimated eps for group {gid}: {eps:.4f}")
        
        eps_by_gid[gid] = eps
    
    return eps_by_gid
