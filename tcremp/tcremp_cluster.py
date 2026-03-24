import numpy as np
import pandas as pd
import time
import argparse
import logging
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
import multiprocessing as mp

import faiss
try:
    import networkit as nk
    import networkit.community as nkc
except Exception:
    nk = None
    nkc = None


# =====================================
# === Standardization and PCA utils ===
# =====================================

def standardize_data(data: np.ndarray):
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32, copy=False)

    start = time.time()
    means = np.mean(data, axis=0, dtype=np.float32)
    stds = np.std(data, axis=0, dtype=np.float32)
    stds[stds == 0] = 1.0
    data -= means
    data /= stds
    elapsed = time.time() - start
    logging.info(f"Standardization (in-place, overflow-safe) completed in {elapsed:.2f} sec.")
    return data


def apply_pca(data, n_components=50):
    start = time.time()
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data)
    elapsed = time.time() - start
    logging.info(f"PCA completed: {n_components} components, time: {elapsed:.2f} sec.")
    return reduced


def apply_pca_incremental(data, n_components=50, batch_size=100000):
    start = time.time()
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    ipca.fit(data)
    reduced = ipca.transform(data)
    elapsed = time.time() - start
    logging.info(f"Incremental PCA completed, time: {elapsed:.2f} sec.")
    return reduced

def prepare_data_for_clustering(df: pd.DataFrame, n_components):
    df = standardize_data(df.values)
    df = apply_pca(df, n_components=n_components)
    return df



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

    knee = KneeLocator(
        range(1, len(kth_distances) + 1),  # x values
        kth_distances,                     # y values
        S=1.0,
        curve="concave",
        interp_method="polynomial",
        polynomial_degree=poly_degree,
        online=True,
        direction="increasing",
    )

    if knee.knee is not None:
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


def run_dbscan_clustering(df: pd.DataFrame, n_components: int = 50, min_samples: int = 5, n_neighbors: int = 4):
    # Standardize data
    standardized = standardize_data(df.values)

    # Apply PCA and reduce dimensionality
    reduced = apply_pca(standardized, n_components=n_components)

    # Estimate optimal eps using k-nearest neighbors and KneeLocator
    eps = estimate_dbscan_eps(reduced, n_neighbors=n_neighbors)

    # Run DBSCAN clustering
    labels = cluster_dbscan(reduced, eps=eps, min_samples=min_samples)
    return labels


def compute_blockwise_knn_merged(
    bg,
    sample,
    k_neighbors: int,
    bg_index_path=None,
    sample_index_path=None,
    rebuild_bg: bool = False,
    rebuild_sample: bool = False,
    save_blocks: bool = False,
    output_dir=None,
    nproc: int = 1,
):
    """
    Compute k-NN distances for the joint set (sample + background) using FAISS.

    This is a simplified, non-blockwise implementation used by TCRempNet.
    It concatenates sample and bg data (sample first), builds a single FAISS
    index, and searches for k nearest neighbors for each point.

    Parameters
    ----------
    bg : array-like, shape (n_bg, n_features)
        Background data in the same feature space as sample.
    sample : array-like, shape (n_sample, n_features)
        Sample data.
    k_neighbors : int
        Number of neighbors to retrieve.
    Other parameters are kept for API compatibility and are currently ignored.

    Returns
    -------
    distances : np.ndarray, shape (n_total, k_neighbors)
    indices   : np.ndarray, shape (n_total, k_neighbors)
        Distances and indices of the k nearest neighbors in the concatenated
        (sample + bg) array.
    """
    # Convert inputs to float32 numpy arrays
    if isinstance(sample, pd.DataFrame):
        sample_arr = sample.to_numpy(dtype="float32")
    else:
        sample_arr = np.asarray(sample, dtype="float32")

    if isinstance(bg, pd.DataFrame):
        bg_arr = bg.to_numpy(dtype="float32")
    else:
        bg_arr = np.asarray(bg, dtype="float32")

    data = np.vstack([sample_arr, bg_arr])
    n, d = data.shape
    logging.info(
        f"Building FAISS index for joint data in compute_blockwise_knn_merged: "
        f"N={n}, d={d}, k={k_neighbors}, n_threads={nproc}"
    )

    faiss.omp_set_num_threads(int(nproc))
    index = faiss.IndexFlatL2(d)
    index.add(data)
    distances, indices = index.search(data, k_neighbors)
    logging.info("Finished FAISS k-NN search in compute_blockwise_knn_merged.")
    return distances, indices


def _build_nk_graph_from_knn(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    metric: str = "dissimilarity",
    max_distance: float | None = None,
):
    """
    Build a weighted undirected NetworKit graph from k-NN results.
    """
    if nk is None:
        raise ImportError(
            "NetworKit is required for Leiden clustering. "
            "Install it via 'conda install -c conda-forge networkit'."
        )

    n, k = knn_indices.shape
    if knn_distances.shape != (n, k):
        raise ValueError("knn_indices and knn_distances must have the same shape")

    logging.info(f"Building NetworKit graph from k-NN (N={n}, k={k}) ...")
    G = nk.Graph(n, weighted=True, directed=False)

    def dist_to_weight(d: float) -> float:
        if metric == "dissimilarity":
            # smaller distance -> larger weight
            return 1.0 / (1.0 + float(d))
        elif metric == "similarity":
            return float(d)
        else:
            raise ValueError(f"Unknown metric='{metric}'")

    for i in range(n):
        neighs = knn_indices[i]
        dists = knn_distances[i]
        for j, dist in zip(neighs, dists):
            j = int(j)
            if j < 0 or j == i:
                continue
            if metric == "dissimilarity" and max_distance is not None and dist > max_distance:
                continue
            w = dist_to_weight(dist)
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
) -> np.ndarray:
    """
    Parallel Leiden clustering (NetworKit.ParallelLeiden) on a k-NN graph.

    Parameters
    ----------
    knn_indices, knn_distances : np.ndarray, shape (N, k)
        k-NN graph (indices in the concatenated array and distances).
    resolution : float
        Gamma parameter for Leiden. Larger -> more, smaller clusters.
    n_iterations : int
        Number of Leiden iterations (2–3 usually enough).
    n_threads : int
        Number of threads for NetworKit.
    metric : {"dissimilarity", "similarity"}
        How to interpret knn_distances.
    max_distance : float or None
        When metric == "dissimilarity", edges with distance > max_distance
        can be skipped.
    min_cluster_size : int or None
        If set, clusters smaller than this size are marked as noise (-1).

    Returns
    -------
    labels : np.ndarray, shape (N,)
        Cluster labels for each point.
    """
    if nk is None or nkc is None:
        raise ImportError(
            "NetworKit with community module is required for run_leiden_clustering."
        )

    n = knn_indices.shape[0]
    logging.info(
        f"Running ParallelLeiden on k-NN graph with N={n}, "
        f"k={knn_indices.shape[1]}, resolution={resolution}, "
        f"iterations={n_iterations}, n_threads={n_threads}"
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

    logging.info(
        f"Leiden finished: clusters={part.numberOfSubsets()}, "
        f"labels shape={labels.shape}"
    )

    if min_cluster_size is not None and min_cluster_size > 1:
        uniq, counts = np.unique(labels, return_counts=True)
        small = set(uniq[counts < min_cluster_size])
        if small:
            logging.info(
                f"Marking {len(small)} small clusters (size < {min_cluster_size}) "
                "as noise (-1)."
            )
            mask = np.isin(labels, list(small))
            labels[mask] = -1

    return labels

def hierarchical_leiden_clustering(
    knn_indices,
    knn_distances,
    base_resolution=1.0,
    sub_resolution=5.0,
    min_cluster_size=20,
    sub_min_cluster_size=10,
    n_iterations=3,
    n_threads=8,
    metric="dissimilarity",
):
    """
    Two-level Leiden clustering similar to Seurat:
    1) coarse global clustering (base_resolution)
    2) subclustering inside each large cluster (sub_resolution)

    Returns
    -------
    final_labels : np.ndarray of shape (N,)
    """
    N = knn_indices.shape[0]

    # --- STEP 1: Global Leiden
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

    # --- STEP 2: Subclustering inside each large cluster ---
    next_cluster_id = final_labels.max() + 1

    uniq_clusters = [c for c in np.unique(global_labels) if c != -1]

    logging.info(f"[HL] Global clusters found: {uniq_clusters}")

    for cl in uniq_clusters:
        idx = np.where(global_labels == cl)[0]
        size = len(idx)

        if size < 2 * sub_min_cluster_size:
            logging.info(f"[HL] Cluster {cl}: size {size}, too small for subdivision → keep as is.")
            continue

        logging.info(f"[HL] Subclustering cluster {cl} (size={size}) with resolution={sub_resolution}")

        # extract kNN subgraph for this cluster
        sub_indices = knn_indices[idx]
        sub_distances = knn_distances[idx]

        # convert neighbor indices to subcluster-local indexing
        mapping = {old: new for new, old in enumerate(idx)}
        remapped_indices = np.vectorize(lambda x: mapping.get(x, -1))(sub_indices)

        # run Leiden inside cluster
        sub_labels = run_leiden_clustering(
            knn_indices=remapped_indices,
            knn_distances=sub_distances,
            resolution=sub_resolution,
            n_iterations=n_iterations,
            n_threads=n_threads,
            metric=metric,
            min_cluster_size=sub_min_cluster_size,
        )

        # if only one cluster found → skip
        if len(np.unique(sub_labels[sub_labels >= 0])) <= 1:
            logging.info(f"[HL] Cluster {cl} not subdivided meaningfully → keeping original label.")
            continue

        # reassign new global cluster ids
        logging.info(f"[HL] Cluster {cl} subdivided into {len(np.unique(sub_labels[sub_labels >= 0]))} subclusters.")

        for sub in np.unique(sub_labels):
            if sub == -1:  # noise inside cluster
                continue
            mask = (sub_labels == sub)
            final_labels[idx[mask]] = next_cluster_id
            next_cluster_id += 1

    return final_labels


def _run_dbscan_on_chunk(args):
    """
    Worker: run DBSCAN inside one Leiden cluster
    """
    (
        indices,
        data_reduced,
        k_neighbors
    ) = args
    MIN_DBSCAN_CLUSTER_SIZE = 10
    if len(indices) < MIN_DBSCAN_CLUSTER_SIZE:
        logging.info(
            f"[Leiden→DBSCAN] cluster size={len(indices)} < {MIN_DBSCAN_CLUSTER_SIZE}, "
            "skip DBSCAN → single cluster"
        )
        return indices, np.zeros(len(indices), dtype=int)


    eps = estimate_dbscan_eps(
        data_reduced[indices],
        n_neighbors=k_neighbors
    )
    if eps == 0:
        return indices, np.ones(len(indices), dtype=int)
        
    labels = cluster_dbscan(data_reduced[indices], eps=eps)

    return indices, labels


def hierarchical_leiden_dbscan_clustering(
    data_reduced,
    knn_indices,
    knn_distances,
    resolution,
    k_neighbors,
    n_jobs
):
    """
    1) Global Leiden at low resolution
    2) DBSCAN (with auto-eps) inside each Leiden cluster
    """

    # --- Step 1: Leiden on full graph ---
    leiden_labels = run_leiden_clustering(
        knn_indices=knn_indices,
        knn_distances=knn_distances,
        resolution=resolution,
        n_threads=n_jobs
    )

    final_labels = -np.ones(len(leiden_labels), dtype=int)
    offset = 0

    leiden_clusters = np.unique(leiden_labels[leiden_labels >= 0])

    tasks = []
    for cid in leiden_clusters:
        idx = np.where(leiden_labels == cid)[0]
        tasks.append((idx, data_reduced, k_neighbors))

    # --- LOG cluster sizes ---
    uniq, counts = np.unique(leiden_labels[leiden_labels >= 0], return_counts=True)
    for cid, size in zip(uniq, counts):
        logging.info(f"[Leiden] cluster {cid}: size={size}")

    # --- Step 2: parallel DBSCAN ---
    with mp.Pool(n_jobs) as pool:
        results = pool.map(_run_dbscan_on_chunk, tasks)

    for indices, labels in results:
        mask = labels >= 0
        final_labels[indices[mask]] = labels[mask] + offset
        if mask.any():
            offset += labels.max() + 1

    return final_labels



def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )

    parser = argparse.ArgumentParser(description="Run clustering using PCA + DBSCAN")
    parser.add_argument("--input", type=str, help="Path to the input data file (CSV/TSV).")
    parser.add_argument("--output", type=str, help="Path to save the clustering results.")
    parser.add_argument("--components", type=int, default=50,
                        help="Number of PCA components (default: 50)")
    parser.add_argument("--min_samples", type=int, default=5,
                        help="min_samples parameter for DBSCAN (default: 5)")
    parser.add_argument("--kth_neighbor", type=int, default=4,
                        help="k-th neighbor parameter for Knee estimation (default: 4)")
    args = parser.parse_args()

    logging.info("Loading data...")
    df = pd.read_csv(args.input, sep='\t')

    logging.info("Starting clustering...")
    labels = run_dbscan_clustering(df,
                                   n_components=args.components,
                                   min_samples=args.min_samples,
                                   n_neighbors=args.kth_neighbor)

    df["cluster"] = labels
    df.to_csv(args.output, sep='\t')
    logging.info(f"Clustering results saved to {args.output}")


if __name__ == "__main__":
    main()
