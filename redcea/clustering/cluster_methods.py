# tcrempnet/clustering/cluster_methods.py
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .preprocess import standardize_data, apply_pca
from .eps_estimation import estimate_dbscan_eps, cluster_dbscan


# --- optional dependency: networkit ---
try:
    import networkit as nk
    import networkit.community as nkc
except Exception:
    nk = None
    nkc = None


# ==========================
# === DBSCAN (classic) =====
# ==========================

def run_dbscan_clustering(
    df: pd.DataFrame,
    n_components: int = 50,
    min_samples: int = 5,
    n_neighbors: int = 4,
) -> np.ndarray:
    """
    Standardize -> PCA -> estimate eps -> DBSCAN.

    Baseline / debugging helper.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("run_dbscan_clustering expects a pandas.DataFrame")

    standardized = standardize_data(df.values)
    reduced = apply_pca(standardized, n_components=n_components)

    eps = estimate_dbscan_eps(reduced, n_neighbors=n_neighbors)
    labels = cluster_dbscan(reduced, eps=eps, min_samples=min_samples)
    return labels


# ==========================
# === Leiden utils =========
# ==========================

def _require_knn_shapes(knn_indices: np.ndarray, knn_distances: np.ndarray) -> tuple[int, int]:
    if knn_indices.ndim != 2 or knn_distances.ndim != 2:
        raise ValueError("knn_indices and knn_distances must be 2D arrays (N, k)")
    if knn_indices.shape != knn_distances.shape:
        raise ValueError(f"Shape mismatch: knn_indices={knn_indices.shape} knn_distances={knn_distances.shape}")
    if not np.isfinite(knn_distances).all():
        raise ValueError("knn_distances contains NaN/inf")
    return int(knn_indices.shape[0]), int(knn_indices.shape[1])


def _build_nk_graph_from_knn(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    metric: str = "dissimilarity",
    max_distance: float | None = None,
):
    """
    Build a weighted undirected NetworKit graph from k-NN results.

    Notes:
      - metric == "dissimilarity": smaller distance => larger weight via 1/(1+d)
      - metric == "similarity":    distances are treated as weights directly
      - knn_distances are assumed to be L2 (NOT squared).
    """
    if nk is None:
        raise ImportError(
            "NetworKit is required for Leiden clustering. "
            "Install it via 'conda install -c conda-forge networkit'."
        )

    n, k = _require_knn_shapes(knn_indices, knn_distances)

    logging.info(f"Building NetworKit graph from k-NN (N={n}, k={k}) ...")
    G = nk.Graph(n, weighted=True, directed=False)

    def dist_to_weight(d: float) -> float:
        if metric == "dissimilarity":
            return 1.0 / (1.0 + float(d))
        if metric == "similarity":
            return float(d)
        raise ValueError(f"Unknown metric='{metric}'")

    for i in range(n):
        neighs = knn_indices[i]
        dists = knn_distances[i]
        for j, dist in zip(neighs, dists):
            j = int(j)
            if j < 0 or j == i:
                continue
            if metric == "dissimilarity" and max_distance is not None and float(dist) > float(max_distance):
                continue
            w = dist_to_weight(float(dist))
            if w <= 0:
                continue
            if not G.hasEdge(i, j):
                G.addEdge(i, j, w)

    logging.info(
        f"Graph built: nodes={G.numberOfNodes()}, edges={G.numberOfEdges()}, "
        f"weighted={G.isWeighted()}, directed={G.isDirected()}"
    )
    return G


def run_leiden_clustering(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    resolution: float = 10,
    n_iterations: int = 3,
    n_threads: int = 8,
    metric: str = "dissimilarity",
    max_distance: float | None = None,
    min_cluster_size: int | None = None,
    min_cluster_size_mask: np.ndarray | None = None,
    # compatibility alias used in some call-sites
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """
    Parallel Leiden clustering (NetworKit.ParallelLeiden) on a k-NN graph.
    """
    if nk is None or nkc is None:
        raise ImportError("NetworKit (networkit) with community module is required for run_leiden_clustering.")

    if n_jobs is not None:
        n_threads = int(n_jobs)

    n, k = _require_knn_shapes(knn_indices, knn_distances)

    logging.info(
        f"Running ParallelLeiden on k-NN graph with N={n}, k={k}, "
        f"resolution={resolution}, iterations={n_iterations}, n_threads={n_threads}"
    )

    nk.setNumberOfThreads(int(n_threads))
    G = _build_nk_graph_from_knn(
        knn_indices=knn_indices,
        knn_distances=knn_distances,
        metric=metric,
        max_distance=max_distance,
    )

    algo = nkc.ParallelLeiden(
        G,
        randomize=True,
        gamma=float(resolution),
        iterations=int(n_iterations),
    )
    algo.run()
    part = algo.getPartition()
    labels = np.asarray(part.getVector(), dtype=np.int64)

    logging.info(f"Leiden finished: clusters={part.numberOfSubsets()}, labels shape={labels.shape}")

    if min_cluster_size is not None and int(min_cluster_size) > 1:
        counted_labels = labels
        size_label = "size"
        if min_cluster_size_mask is not None:
            mask = np.asarray(min_cluster_size_mask, dtype=bool)
            if mask.shape != labels.shape:
                raise ValueError("min_cluster_size_mask must have the same shape as labels.")
            counted_labels = labels[mask]
            size_label = "sample size"

        uniq, counts = np.unique(counted_labels, return_counts=True)
        small = set(uniq[counts < int(min_cluster_size)])
        if small:
            logging.info(
                f"Marking {len(small)} small clusters "
                f"({size_label} < {min_cluster_size}) as noise (-1)."
            )
            mask = np.isin(labels, list(small))
            labels[mask] = -1

    return labels


def hierarchical_leiden_clustering(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    base_resolution: float = 1.0,
    sub_resolution: float = 5.0,
    min_cluster_size: int = 20,
    sub_min_cluster_size: int = 10,
    n_iterations: int = 3,
    n_threads: int = 8,
    metric: str = "dissimilarity",
) -> np.ndarray:
    """
    Two-level Leiden clustering (Seurat-like):
      1) coarse global clustering (base_resolution)
      2) subclustering inside each large cluster (sub_resolution)
    """
    _require_knn_shapes(knn_indices, knn_distances)
    N = knn_indices.shape[0]

    logging.info(f"[HL] Running global Leiden (resolution={base_resolution})")
    global_labels = run_leiden_clustering(
        knn_indices=knn_indices,
        knn_distances=knn_distances,
        resolution=base_resolution,
        n_iterations=n_iterations,
        n_threads=n_threads,
        metric=metric,
        min_cluster_size=min_cluster_size,
    )

    final_labels = np.array(global_labels, copy=True)
    next_cluster_id = int(final_labels.max()) + 1

    uniq_clusters = [int(c) for c in np.unique(global_labels) if int(c) != -1]
    logging.info(f"[HL] Global clusters found: {uniq_clusters}")

    for cl in uniq_clusters:
        idx = np.where(global_labels == cl)[0]
        size = int(len(idx))

        if size < 2 * int(sub_min_cluster_size):
            logging.info(f"[HL] Cluster {cl}: size {size}, too small for subdivision -> keep as is.")
            continue

        logging.info(f"[HL] Subclustering cluster {cl} (size={size}) with resolution={sub_resolution}")

        sub_indices = knn_indices[idx]
        sub_distances = knn_distances[idx]

        mapping = {int(old): int(new) for new, old in enumerate(idx)}
        remapped_indices = np.vectorize(lambda x: mapping.get(int(x), -1), otypes=[np.int64])(
            sub_indices
        ).astype(np.int64)

        sub_labels = run_leiden_clustering(
            knn_indices=remapped_indices,
            knn_distances=sub_distances,
            resolution=sub_resolution,
            n_iterations=n_iterations,
            n_threads=n_threads,
            metric=metric,
            min_cluster_size=sub_min_cluster_size,
        )

        if len(np.unique(sub_labels[sub_labels >= 0])) <= 1:
            logging.info(f"[HL] Cluster {cl} not subdivided meaningfully -> keeping original label.")
            continue

        logging.info(f"[HL] Cluster {cl} subdivided into {len(np.unique(sub_labels[sub_labels >= 0]))} subclusters.")

        for sub in np.unique(sub_labels):
            if int(sub) == -1:
                continue
            mask = (sub_labels == sub)
            final_labels[idx[mask]] = next_cluster_id
            next_cluster_id += 1

    if final_labels.shape != (N,):
        raise RuntimeError("hierarchical_leiden_clustering produced wrong output shape")
    return final_labels


# ==================================
# === Leiden -> DBSCAN hybrid =======
# ==================================

def hierarchical_leiden_dbscan_clustering(
    data_reduced: np.ndarray,
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    resolution: float,
    k_neighbors: int,
    n_jobs: int,  # used only for NetworKit threads (Leiden step)
    metric: str = "dissimilarity",
    num_points_for_core=5,
) -> np.ndarray:
    """
    1) Global Leiden at given resolution (NetworKit threads = n_jobs)
    2) DBSCAN (auto-eps) inside each Leiden cluster — SEQUENTIAL (no multiprocessing)
    """
    if data_reduced is None:
        raise ValueError("data_reduced is required (used for intra-cluster DBSCAN eps estimation)")
    if not isinstance(data_reduced, np.ndarray):
        data_reduced = np.asarray(data_reduced)
    if not np.isfinite(data_reduced).all():
        raise ValueError("data_reduced contains NaN/inf")

    N, _ = _require_knn_shapes(knn_indices, knn_distances)
    if data_reduced.shape[0] != N:
        raise ValueError(f"data_reduced N mismatch: data_reduced={data_reduced.shape[0]} knn={N}")

    # --- Step 1: Leiden on full graph (threaded internally by NetworKit) ---
    leiden_labels = run_leiden_clustering(
        knn_indices=knn_indices,
        knn_distances=knn_distances,
        resolution=resolution,
        n_threads=int(n_jobs),
        metric=metric,
    )

    final_labels = -np.ones(len(leiden_labels), dtype=int)
    offset = 0

    # --- LOG cluster sizes ---
    uniq, counts = np.unique(leiden_labels[leiden_labels >= 0], return_counts=True)
    for cid, size in zip(uniq, counts):
        logging.info(f"[Leiden] cluster {int(cid)}: size={int(size)}")

    # --- Step 2: SEQUENTIAL DBSCAN inside each Leiden cluster ---
    MIN_DBSCAN_CLUSTER_SIZE = 10
    leiden_clusters = np.unique(leiden_labels[leiden_labels >= 0])

    for cid in leiden_clusters:
        idx = np.where(leiden_labels == cid)[0]
        size = int(len(idx))

        if size < MIN_DBSCAN_CLUSTER_SIZE:
            logging.info(
                f"[Leiden->DBSCAN] cluster {int(cid)} size={size} < {MIN_DBSCAN_CLUSTER_SIZE}, "
                "skip DBSCAN -> single cluster"
            )
            labels_local = np.zeros(size, dtype=int)
        else:
            eps = estimate_dbscan_eps(
                data_reduced[idx],
                n_neighbors=int(k_neighbors),
            )
            if float(eps) == 0.0:
                labels_local = np.ones(size, dtype=int)
            else:
                labels_local = cluster_dbscan(data_reduced[idx], eps=eps, min_samples=num_points_for_core)

        mask = labels_local >= 0
        final_labels[idx[mask]] = labels_local[mask] + offset
        if mask.any():
            offset += int(labels_local.max()) + 1

    return final_labels
