import logging
import time
from typing import Dict

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


def knn_neighbor_distances(knn_distances: np.ndarray, neighbor_rank: int) -> np.ndarray:
    """
    Return distances to the `neighbor_rank`-th nearest *other* point.

    Column 0 is assumed to be self-distance, so neighbor ranks start at column 1.
    """
    knn_distances = np.asarray(knn_distances)
    if knn_distances.ndim != 2 or knn_distances.shape[1] < 1:
        raise ValueError("knn_distances must be a 2D array with at least one neighbor column")
    if neighbor_rank < 1:
        raise ValueError("neighbor_rank must be >= 1")

    col_idx = neighbor_rank
    if col_idx >= knn_distances.shape[1]:
        raise ValueError(
            f"neighbor_rank={neighbor_rank} requires column {col_idx}, "
            f"but knn_distances has only {knn_distances.shape[1]} columns"
        )
    return knn_distances[:, col_idx]


def legacy_kth_returned_neighbor_distances(knn_distances: np.ndarray, kth_neighbor: int) -> np.ndarray:
    """
    Return distances from the legacy TCRempNet kNN column convention.

    `kth_neighbor` is 1-based over the returned FAISS columns, where column 0
    is usually the self-match. Thus legacy kth=5 means column 4.
    """
    knn_distances = np.asarray(knn_distances)
    if knn_distances.ndim != 2 or knn_distances.shape[1] < 1:
        raise ValueError("knn_distances must be a 2D array with at least one neighbor column")
    if kth_neighbor < 1:
        raise ValueError("kth_neighbor must be >= 1")

    col_idx = int(kth_neighbor) - 1
    if col_idx >= knn_distances.shape[1]:
        raise ValueError(
            f"kth_neighbor={kth_neighbor} requires column {col_idx}, "
            f"but knn_distances has only {knn_distances.shape[1]} columns"
        )
    return knn_distances[:, col_idx]


# === copied 1:1 from your snippet ===

def estimate_dbscan_eps(
    data,
    distances=None,
    n_neighbors: int = 4,
    quantile: float = 0.05,   # оставляем для совместимости, но legacy его не использует
    poly_degree: int = 10,
):
    """
    Legacy eps estimation from old TCRemp/RedCEA implementation.
    """
    start = time.time()

    if distances is None:
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs = neigh.fit(data)
        dists, _ = nbrs.kneighbors(data)
        kth_distances = legacy_kth_returned_neighbor_distances(dists, n_neighbors)
        total_num = len(data)
    else:
        kth_distances = np.asarray(distances)
        total_num = len(kth_distances)

    number_of_points_for_knee = min(
        total_num,
        max(20000, int(total_num * 0.2)),
    )

    chosen_elements = np.random.choice(
        kth_distances,
        size=number_of_points_for_knee,
    )

    distances_sorted = np.sort(chosen_elements)

    knee = KneeLocator(
        range(1, len(distances_sorted) + 1),
        distances_sorted,
        S=1.0,
        curve="concave",
        interp_method="polynomial",
        polynomial_degree=poly_degree,
        online=True,
        direction="increasing",
    )

    eps = distances_sorted[knee.knee]

    logging.info(
        f"Estimated eps for DBSCAN: {eps:.4f}, total time: {(time.time() - start):.2f} sec."
    )

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


def cluster_dbscan_with_filter(
    data,
    eps=None,
    min_samples=5,
    nearest_neighbor_distances=None,
    return_details: bool = False,
):
    """
    Legacy TCRempNet DBSCAN:
    pre-filter points with d1 > eps, then run DBSCAN on the remainder.
    Filtered-out points are marked as noise (-1).
    """
    if nearest_neighbor_distances is None:
        raise ValueError("nearest_neighbor_distances is required for legacy DBSCAN pre-filtering")

    data = np.asarray(data)
    d1 = np.asarray(nearest_neighbor_distances)
    if data.shape[0] != d1.shape[0]:
        raise ValueError(
            "nearest_neighbor_distances must have one value per row in data: "
            f"got data={data.shape[0]} rows and d1={d1.shape[0]}"
        )

    n_total = data.shape[0]
    start = time.time()
    mask = d1 <= eps
    n_filtered_out = int(np.sum(~mask))
    logging.info(
        "Filtered out %d points out of %d (%.2f%%) due to large d1 > eps",
        n_filtered_out,
        n_total,
        100.0 * n_filtered_out / max(1, n_total),
    )

    labels = np.full(n_total, -1, dtype=int)
    if not np.any(mask):
        logging.info("Filtered DBSCAN completed: all points were filtered out before clustering.")
        if return_details:
            return labels, {
                "keep_mask": mask,
                "filtered_data_size": 0,
                "n_filtered_out": n_filtered_out,
            }
        return labels

    filtered_data = data[mask]
    db = DBSCAN(eps=eps, min_samples=min_samples)
    filtered_labels = db.fit_predict(filtered_data)

    labels[mask] = filtered_labels

    elapsed = time.time() - start
    n_clusters = len(set(filtered_labels)) - (1 if -1 in filtered_labels else 0)
    n_noise = int(np.count_nonzero(labels == -1))
    logging.info(
        "Filtered DBSCAN completed: clusters = %d, noise points = %d, time: %.2f sec.",
        n_clusters,
        n_noise,
        elapsed,
    )
    if return_details:
        return labels, {
            "keep_mask": mask,
            "filtered_data_size": int(filtered_data.shape[0]),
            "n_filtered_out": n_filtered_out,
        }
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

        kth = legacy_kth_returned_neighbor_distances(sample_ss_distances_l2[idx], kth_neighbor)
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
    from redcea.clustering.cdr3_grouping import (
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
        
        kth = legacy_kth_returned_neighbor_distances(dist_matrix[idx], kth_neighbor)
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
