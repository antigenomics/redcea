# tcrempnet/clustering/cluster_methods.py
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

import numpy as np
import pandas as pd

from redcea.config import PipelineConfig
from redcea.debug import summarize_distribution

from .preprocess import standardize_data, apply_pca
from .eps_estimation import (
    estimate_dbscan_eps,
    cluster_dbscan,
    cluster_dbscan_with_filter,
    knn_neighbor_distances,
    legacy_kth_returned_neighbor_distances,
)
from .cdr3_grouping import compute_cdr3_len, build_len_to_group_id, map_len_to_group_id
from .eps_estimation import estimate_eps_by_group_from_sample, eps_per_point_from_group_id, estimate_eps_by_group_flexible
from .vdbscan import vdbscan_from_knn


@dataclass
class JointDbscanDebugArtifacts:
    labels: np.ndarray
    eps: float
    eps_column: int
    eps_distances: np.ndarray
    d1_column: int
    d1_distances: np.ndarray
    keep_mask: np.ndarray
    self_index_fraction: float


@dataclass
class JointVdbscanDebugArtifacts:
    labels: np.ndarray
    eps_estimation_based_on: str
    kth_neighbor: int
    len_to_gid: dict[int, int]
    sample_group_id: np.ndarray
    background_group_id: np.ndarray
    joint_group_id: np.ndarray
    eps_by_gid: dict[int, float]


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


def run_dbscan_clustering_with_prefilter(
    data_reduced: np.ndarray,
    *,
    eps: float,
    nearest_neighbor_distances: np.ndarray,
    min_samples: int = 5,
) -> np.ndarray:
    """
    Legacy TCRempNet DBSCAN:
    use precomputed reduced embeddings, estimate noise by d1 > eps,
    then run DBSCAN on the remaining points.
    """
    if not isinstance(data_reduced, np.ndarray):
        data_reduced = np.asarray(data_reduced)
    if not np.isfinite(data_reduced).all():
        raise ValueError("data_reduced contains NaN/inf")

    return cluster_dbscan_with_filter(
        data_reduced,
        eps=eps,
        min_samples=min_samples,
        nearest_neighbor_distances=nearest_neighbor_distances,
    )


def _nearest_neighbor_distances_from_knn(knn_distances: np.ndarray) -> np.ndarray:
    return knn_neighbor_distances(knn_distances, neighbor_rank=1)


def _legacy_eps_distances_from_knn(knn_distances: np.ndarray, kth_neighbor: int) -> tuple[int, np.ndarray]:
    eps_column = int(kth_neighbor) - 1
    return eps_column, legacy_kth_returned_neighbor_distances(knn_distances, kth_neighbor)


def run_joint_dbscan_clustering_with_diagnostics(
    *,
    config: PipelineConfig,
    knn,
) -> JointDbscanDebugArtifacts:
    d0 = np.asarray(knn.distances[:, 0], dtype=np.float64)
    d1 = np.asarray(_nearest_neighbor_distances_from_knn(knn.distances), dtype=np.float64)
    eps_column, eps_distances_raw = _legacy_eps_distances_from_knn(knn.distances, config.eps_k_neighbors)
    eps_distances = np.asarray(eps_distances_raw, dtype=np.float64)
    self_index_fraction = float(np.mean(knn.indices[:, 0] == np.arange(knn.indices.shape[0])))

    d0_stats = summarize_distribution(d0)
    d1_stats = summarize_distribution(d1)
    eps_input_stats = summarize_distribution(eps_distances)

    logging.info(
        "DBSCAN input: shape=%s cluster_algo=%s min_samples=%d k_neighbors=%d eps_k_neighbors=%d pc_components=%d",
        tuple(knn.data_reduced.shape),
        config.cluster_algo,
        config.core_min_samples,
        config.k_neighbors,
        config.eps_k_neighbors,
        config.cluster_pc_components,
    )
    logging.info(
        "KNN self check: self_index_fraction=%.6f, d0_min=%.6f, d0_max=%.6f, d0_q99=%.6f",
        self_index_fraction,
        d0_stats["min"],
        d0_stats["max"],
        d0_stats["q99"],
    )
    logging.info(
        "Nearest non-self neighbor distance stats: min=%.6f q01=%.6f q05=%.6f q25=%.6f q50=%.6f q75=%.6f q95=%.6f q99=%.6f max=%.6f",
        d1_stats["min"],
        d1_stats["q01"],
        d1_stats["q05"],
        d1_stats["q25"],
        d1_stats["q50"],
        d1_stats["q75"],
        d1_stats["q95"],
        d1_stats["q99"],
        d1_stats["max"],
    )

    eps = float(
        estimate_dbscan_eps(
            data=None,
            distances=eps_distances,
            n_neighbors=config.eps_k_neighbors,
        )
    )
    logging.info(
        "DBSCAN eps input stats: column=%d min=%.6f q01=%.6f q05=%.6f q25=%.6f q50=%.6f q75=%.6f q95=%.6f q99=%.6f max=%.6f eps=%.6f",
        eps_column,
        eps_input_stats["min"],
        eps_input_stats["q01"],
        eps_input_stats["q05"],
        eps_input_stats["q25"],
        eps_input_stats["q50"],
        eps_input_stats["q75"],
        eps_input_stats["q95"],
        eps_input_stats["q99"],
        eps_input_stats["max"],
        eps,
    )

    labels, filter_details = cluster_dbscan_with_filter(
        knn.data_reduced,
        eps=eps,
        min_samples=config.core_min_samples,
        nearest_neighbor_distances=d1,
        return_details=True,
    )
    keep_mask = np.asarray(filter_details["keep_mask"], dtype=bool)
    logging.info(
        "DBSCAN prefilter summary: d1_column=%d removed=%d kept=%d",
        1,
        int(np.count_nonzero(~keep_mask)),
        int(np.count_nonzero(keep_mask)),
    )

    return JointDbscanDebugArtifacts(
        labels=np.asarray(labels),
        eps=eps,
        eps_column=eps_column,
        eps_distances=eps_distances,
        d1_column=1,
        d1_distances=d1,
        keep_mask=keep_mask,
        self_index_fraction=self_index_fraction,
    )


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

    if min_cluster_size_mask is not None:
        min_cluster_size_mask = np.asarray(min_cluster_size_mask, dtype=bool)
        if min_cluster_size_mask.shape != labels.shape:
            raise ValueError(
                "min_cluster_size_mask must have the same shape as labels: "
                f"expected {labels.shape}, got {min_cluster_size_mask.shape}"
            )

    if min_cluster_size is not None and int(min_cluster_size) > 1:
        labels_for_counting = labels if min_cluster_size_mask is None else labels[min_cluster_size_mask]
        uniq, counts = np.unique(labels_for_counting, return_counts=True)
        small = set(uniq[counts < int(min_cluster_size)])
        if small:
            scope = "using masked membership counts" if min_cluster_size_mask is not None else "globally"
            logging.info(
                f"Marking {len(small)} small clusters (size < {min_cluster_size}) as noise (-1), {scope}."
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
    min_cluster_size_mask: np.ndarray | None = None,
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
        min_cluster_size_mask=min_cluster_size_mask,
    )

    final_labels = np.array(global_labels, copy=True)
    next_cluster_id = int(final_labels.max()) + 1

    uniq_clusters = [int(c) for c in np.unique(global_labels) if int(c) != -1]
    logging.info(f"[HL] Global clusters found: {uniq_clusters}")

    for cl in uniq_clusters:
        idx = np.where(global_labels == cl)[0]
        size = int(len(idx))
        sample_size = (
            int(np.count_nonzero(min_cluster_size_mask[idx]))
            if min_cluster_size_mask is not None
            else size
        )

        if sample_size < 2 * int(sub_min_cluster_size):
            logging.info(
                f"[HL] Cluster {cl}: size {size}, sample_size {sample_size}, too small for subdivision -> keep as is."
            )
            continue

        logging.info(
            f"[HL] Subclustering cluster {cl} (size={size}, sample_size={sample_size}) "
            f"with resolution={sub_resolution}"
        )

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
            min_cluster_size_mask=min_cluster_size_mask[idx] if min_cluster_size_mask is not None else None,
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


def hierarchical_vdbscan_leiden_clustering(
    *,
    config: PipelineConfig,
    joint_representations: pd.DataFrame,
    sample_representations: pd.DataFrame,
    background_representations: pd.DataFrame,
    knn,
    vdbscan_labels: np.ndarray | None = None,
) -> np.ndarray:
    """
    1) Joint vDBSCAN on the full sample+background graph
    2) Leiden subclustering inside each non-noise vDBSCAN cluster

    vDBSCAN defines coarse density-connected components; Leiden then refines
    each component without discarding additional points as noise.
    """
    if len(joint_representations) != len(knn.indices):
        raise ValueError(
            "joint_representations and KNN artifacts must have the same number of rows: "
            f"{len(joint_representations)} != {len(knn.indices)}"
        )
    if vdbscan_labels is None:
        gid_all, eps_by_gid, _, _, _ = _estimate_vdbscan_group_assignments(
            config=config,
            joint_representations=joint_representations,
            sample_representations=sample_representations,
            background_representations=background_representations,
            knn=knn,
        )
        eps_i_all = eps_per_point_from_group_id(gid_all, eps_by_gid)
        vdbscan_labels = np.asarray(
            vdbscan_from_knn(
                knn_indices=knn.indices,
                knn_distances_l2=knn.distances,
                eps_i_l2=eps_i_all,
                num_points_for_core=config.core_min_samples,
                sym_rule=config.vdbscan_sym_rule,
            )
        )
    else:
        vdbscan_labels = np.asarray(vdbscan_labels)
    if vdbscan_labels.shape != (len(joint_representations),):
        raise ValueError(
            "vdbscan_labels must be a 1D array aligned with joint_representations: "
            f"expected {(len(joint_representations),)}, got {vdbscan_labels.shape}"
        )
    final_labels = -np.ones(len(vdbscan_labels), dtype=np.int64)
    next_cluster_id = 0

    coarse_clusters = [int(cluster_id) for cluster_id in np.unique(vdbscan_labels) if int(cluster_id) != -1]
    logging.info(
        "[vDBSCAN->Leiden] coarse clusters=%d, noise_points=%d",
        len(coarse_clusters),
        int(np.count_nonzero(vdbscan_labels == -1)),
    )
    for coarse_cluster_id in coarse_clusters:
        idx = np.where(vdbscan_labels == coarse_cluster_id)[0]
        size = int(len(idx))
        if size <= 1:
            final_labels[idx] = next_cluster_id
            next_cluster_id += 1
            continue

        logging.info(
            "[vDBSCAN->Leiden] refining coarse cluster %d (size=%d) with resolution=%.4f",
            coarse_cluster_id,
            size,
            float(config.leiden_resolution),
        )
        sub_indices = knn.indices[idx]
        sub_distances = knn.distances[idx]
        mapping = {int(old): int(new) for new, old in enumerate(idx)}
        remapped_indices = np.vectorize(lambda x: mapping.get(int(x), -1), otypes=[np.int64])(sub_indices).astype(np.int64)
        sub_labels = run_leiden_clustering(
            knn_indices=remapped_indices,
            knn_distances=sub_distances,
            resolution=config.leiden_resolution,
            n_jobs=config.normalized_nproc,
            min_cluster_size=None,
        )
        positive_subclusters = [int(subcluster) for subcluster in np.unique(sub_labels) if int(subcluster) != -1]
        if not positive_subclusters:
            final_labels[idx] = next_cluster_id
            next_cluster_id += 1
            continue
        for subcluster in positive_subclusters:
            mask = sub_labels == subcluster
            final_labels[idx[mask]] = next_cluster_id
            next_cluster_id += 1

    return final_labels


def _pick_cdr3_column(joint_representations: pd.DataFrame) -> str:
    if "cdr3aa_beta" in joint_representations.columns:
        return "cdr3aa_beta"
    if "cdr3aa_alpha" in joint_representations.columns:
        return "cdr3aa_alpha"
    if "cdr3aa" in joint_representations.columns:
        return "cdr3aa"
    raise ValueError(
        "Cannot find CDR3 AA column in representations. "
        "Expected one of: cdr3aa_beta, cdr3aa_alpha, cdr3aa"
    )


def _estimate_vdbscan_group_assignments(
    *,
    config: PipelineConfig,
    joint_representations: pd.DataFrame,
    sample_representations: pd.DataFrame,
    background_representations: pd.DataFrame,
    knn,
) -> tuple[np.ndarray, dict[int, float], dict[int, int], np.ndarray, np.ndarray]:
    cdr3_col = _pick_cdr3_column(joint_representations)
    eps_estimation_based_on = config.eps_estimation_based_on
    kth_neighbor_for_eps = config.eps_k_neighbors

    if eps_estimation_based_on == "sample":
        logging.info("Running vDBSCAN (eps-by-group from SAMPLE only; L2 distances)")
        sample_len = compute_cdr3_len(sample_representations[cdr3_col])
        len_to_gid = build_len_to_group_id(sample_len, min_frac=0.05)

        sample_gid = map_len_to_group_id(sample_len, len_to_gid, unknown_len_policy="nearest")
        bg_len = compute_cdr3_len(background_representations[cdr3_col])
        bg_gid = map_len_to_group_id(bg_len, len_to_gid, unknown_len_policy="nearest")

        eps_by_gid = estimate_eps_by_group_from_sample(
            sample_group_id=sample_gid,
            sample_ss_distances_l2=knn.dist_ss,
            kth_neighbor=int(kth_neighbor_for_eps),
        )
        gid_all = np.concatenate([sample_gid, bg_gid]).astype(np.int32, copy=False)
        return gid_all, eps_by_gid, len_to_gid, sample_gid, bg_gid
    if eps_estimation_based_on == "background":
        logging.info("Running vDBSCAN (eps-by-group from BACKGROUND only; L2 distances)")
        bg_len = compute_cdr3_len(background_representations[cdr3_col])
        len_to_gid = build_len_to_group_id(bg_len, min_frac=0.05)

        bg_gid = map_len_to_group_id(bg_len, len_to_gid, unknown_len_policy="nearest")
        sample_len = compute_cdr3_len(sample_representations[cdr3_col])
        sample_gid = map_len_to_group_id(sample_len, len_to_gid, unknown_len_policy="nearest")

        eps_by_gid = estimate_eps_by_group_from_sample(
            sample_group_id=bg_gid,
            sample_ss_distances_l2=knn.dist_bb,
            kth_neighbor=int(kth_neighbor_for_eps),
        )
        gid_all = np.concatenate([sample_gid, bg_gid]).astype(np.int32, copy=False)
        return gid_all, eps_by_gid, len_to_gid, sample_gid, bg_gid
    if eps_estimation_based_on == "all":
        logging.info("Running vDBSCAN (eps-by-group from SAMPLE+BACKGROUND combined; L2 distances)")
        joint_len = compute_cdr3_len(joint_representations[cdr3_col])
        len_to_gid = build_len_to_group_id(joint_len, min_frac=0.05)

        gid_all = map_len_to_group_id(joint_len, len_to_gid, unknown_len_policy="nearest")
        eps_by_gid = estimate_eps_by_group_flexible(
            group_id=gid_all,
            dist_matrix=knn.distances,
            kth_neighbor=int(kth_neighbor_for_eps),
        )
        sample_gid = gid_all[: len(sample_representations)].astype(np.int32, copy=False)
        bg_gid = gid_all[len(sample_representations) :].astype(np.int32, copy=False)
        return gid_all, eps_by_gid, len_to_gid, sample_gid, bg_gid

    raise ValueError(
        f"Unknown eps_estimation_based_on='{eps_estimation_based_on}'. "
        "Expected one of: sample, background, all"
    )


def run_joint_vdbscan_clustering_with_diagnostics(
    *,
    config: PipelineConfig,
    joint_representations: pd.DataFrame,
    sample_representations: pd.DataFrame,
    background_representations: pd.DataFrame,
    knn,
) -> JointVdbscanDebugArtifacts:
    gid_all, eps_by_gid, len_to_gid, sample_gid, bg_gid = _estimate_vdbscan_group_assignments(
        config=config,
        joint_representations=joint_representations,
        sample_representations=sample_representations,
        background_representations=background_representations,
        knn=knn,
    )
    eps_i_all = eps_per_point_from_group_id(gid_all, eps_by_gid)
    labels = vdbscan_from_knn(
        knn_indices=knn.indices,
        knn_distances_l2=knn.distances,
        eps_i_l2=eps_i_all,
        num_points_for_core=config.core_min_samples,
        sym_rule=config.vdbscan_sym_rule,
    )
    return JointVdbscanDebugArtifacts(
        labels=np.asarray(labels),
        eps_estimation_based_on=config.eps_estimation_based_on,
        kth_neighbor=int(config.eps_k_neighbors),
        len_to_gid={int(length): int(gid) for length, gid in len_to_gid.items()},
        sample_group_id=np.asarray(sample_gid, dtype=np.int32),
        background_group_id=np.asarray(bg_gid, dtype=np.int32),
        joint_group_id=np.asarray(gid_all, dtype=np.int32),
        eps_by_gid={int(gid): float(eps) for gid, eps in eps_by_gid.items()},
    )


def run_joint_clustering(
    *,
    config: PipelineConfig,
    joint_representations: pd.DataFrame,
    sample_representations: pd.DataFrame,
    background_representations: pd.DataFrame,
    knn,
):
    sample_mask = np.zeros(len(joint_representations), dtype=bool)
    sample_mask[: len(sample_representations)] = True

    if config.cluster_algo == "dbscan":
        _, eps_distances = _legacy_eps_distances_from_knn(knn.distances, config.eps_k_neighbors)
        eps = estimate_dbscan_eps(
            data=None,
            distances=eps_distances,
            n_neighbors=config.eps_k_neighbors,
        )
        return run_dbscan_clustering_with_prefilter(
            knn.data_reduced,
            eps=eps,
            nearest_neighbor_distances=_nearest_neighbor_distances_from_knn(knn.distances),
            min_samples=config.core_min_samples,
        )
    if config.cluster_algo == "leiden_dbscan":
        return hierarchical_leiden_dbscan_clustering(
            data_reduced=knn.data_reduced,
            knn_indices=knn.indices,
            knn_distances=knn.distances,
            resolution=config.leiden_resolution,
            k_neighbors=config.eps_k_neighbors,
            num_points_for_core=config.core_min_samples,
            n_jobs=config.normalized_nproc,
        )
    if config.cluster_algo == "hierarchical_leiden":
        return hierarchical_leiden_clustering(
            knn_indices=knn.indices,
            knn_distances=knn.distances,
            base_resolution=config.leiden_resolution,
            sub_resolution=config.leiden_sub_resolution,
            n_iterations=3,
            n_threads=config.normalized_nproc,
            metric="dissimilarity",
            min_cluster_size=5,
            sub_min_cluster_size=3,
            min_cluster_size_mask=sample_mask,
        )
    if config.cluster_algo == "leiden":
        return run_leiden_clustering(
            knn_indices=knn.indices,
            knn_distances=knn.distances,
            resolution=config.leiden_resolution,
            n_jobs=config.normalized_nproc,
            min_cluster_size=config.core_min_samples,
            min_cluster_size_mask=sample_mask,
        )
    if config.cluster_algo == "vdbscan":
        gid_all, eps_by_gid, _, _, _ = _estimate_vdbscan_group_assignments(
            config=config,
            joint_representations=joint_representations,
            sample_representations=sample_representations,
            background_representations=background_representations,
            knn=knn,
        )
        eps_i_all = eps_per_point_from_group_id(gid_all, eps_by_gid)
        return vdbscan_from_knn(
            knn_indices=knn.indices,
            knn_distances_l2=knn.distances,
            eps_i_l2=eps_i_all,
            num_points_for_core=config.core_min_samples,
            sym_rule=config.vdbscan_sym_rule,
        )
    if config.cluster_algo == "vdbscan_leiden":
        return hierarchical_vdbscan_leiden_clustering(
            config=config,
            joint_representations=joint_representations,
            sample_representations=sample_representations,
            background_representations=background_representations,
            knn=knn,
        )

    raise ValueError(
        f"Unknown cluster_algo='{config.cluster_algo}'. "
        "Expected one of: dbscan, leiden_dbscan, hierarchical_leiden, leiden, vdbscan, vdbscan_leiden"
    )
