import sys

sys.path.append("../")
sys.path.append("../../mirpy")

import numpy as np
import pandas as pd
import time
import argparse
import logging
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import DBSCAN, HDBSCAN
from kneed import KneeLocator
from tcremp.utils import log_memory_usage
import faiss
from pathlib import Path

# =====================================
# === FAISS search multiprocessing  ===
# =====================================

import multiprocessing as mp

def _faiss_worker(args):
    """Один процесс FAISS-поиска."""
    index_path, data_chunk, k = args
    index = faiss.read_index(str(index_path))
    D, I = index.search(data_chunk, k)
    return D, I

def parallel_faiss_search(data, index_path, k_neighbors=10, nproc=4):
    """Распараллеленный FAISS поиск по data с индексом index_path."""
    data = np.ascontiguousarray(data.astype("float32"))
    n = len(data)
    if n == 0:
        return np.zeros((0, k_neighbors), dtype=np.float32), np.zeros((0, k_neighbors), dtype=np.int64)

    chunk_size = int(np.ceil(n / nproc))
    chunks = [data[i:i + chunk_size] for i in range(0, n, chunk_size)]
    tasks = [(index_path, chunk, k_neighbors) for chunk in chunks]

    with mp.get_context("spawn").Pool(nproc) as pool:
        results = pool.map(_faiss_worker, tasks)

    D = np.vstack([r[0] for r in results])
    I = np.vstack([r[1] for r in results])
    return D, I


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


# =====================================
# === k-NN and clustering utilities ===
# =====================================

def get_k_neighbors_distance_matrix(data, n_neighbors=4):
    data = np.ascontiguousarray(data.astype('float32'))
    index = faiss.IndexFlatL2(data.shape[1])
    index.add(data)
    squared_distances, _ = index.search(data, n_neighbors)
    distances = np.sqrt(squared_distances)
    return distances


def estimate_dbscan_eps(data, distances, poly_degree=10):
    total_start = time.time()
    total_num = len(data)
    number_of_points_for_knee = min(total_num, max(20000, int(total_num * 0.2)))
    chosen_elements = np.random.choice(distances, size=number_of_points_for_knee)
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
    logging.info(f"Estimated eps for DBSCAN: {eps:.4f}, total time: {(time.time() - total_start):.2f} sec.")
    return eps


def cluster_dbscan_with_filter(data, eps, min_samples, d1, algo="dbscan"):
    try:
        data = data.to_numpy()
    except Exception:
        pass

    n_total = data.shape[0]
    start = time.time()

    mask = d1 <= eps
    n_filtered_out = np.sum(~mask)
    logging.info(f"Filtered out {n_filtered_out} points out of {n_total} ({n_filtered_out / n_total:.2%}) due to large d1 > eps")
    filtered_data = data[mask]

    if algo == "dbscan":
        model_name = "DBSCAN"
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        model_name = "HDBSCAN"
        model = HDBSCAN()

    filtered_labels = model.fit_predict(filtered_data)
    labels = np.full(data.shape[0], -1, dtype=int)
    labels[mask] = filtered_labels

    elapsed = time.time() - start
    n_clusters = len(set(filtered_labels)) - (1 if -1 in filtered_labels else 0)
    n_noise = list(labels).count(-1)
    logging.info(f"{model_name} completed: clusters = {n_clusters}, noise points = {n_noise}, time: {elapsed:.2f} sec.")
    return labels


def prepare_data_for_clustering(df: pd.DataFrame, n_components):
    df = standardize_data(df.values)
    log_memory_usage('after standardization')
    df = apply_pca(df, n_components=n_components)
    log_memory_usage('after reduction')
    return df


def run_dbscan_clustering(df: pd.DataFrame, eps, closest_neigh_dist_array, min_samples: int = 5, algo: str = "dbscan"):
    labels = cluster_dbscan_with_filter(df, eps=eps, min_samples=min_samples, d1=closest_neigh_dist_array, algo=algo)
    log_memory_usage(f'after {algo}')
    return labels


# =====================================
# === Blockwise FAISS distances =======
# =====================================

def build_or_load_index(data: np.ndarray, index_path: str | Path, rebuild: bool = False):
    index_path = Path(index_path)
    d = data.shape[1]
    start = time.time()

    if index_path.exists() and not rebuild:
        index = faiss.read_index(str(index_path))
        elapsed = time.time() - start
        logging.info(f"Loaded existing FAISS index from {index_path} in {elapsed:.2f} sec.")
        return index

    logging.info(f"Building new FAISS index for data of shape {data.shape}...")
    index = faiss.IndexFlatL2(d)
    index.add(np.ascontiguousarray(data.astype('float32')))
    faiss.write_index(index, str(index_path))
    elapsed = time.time() - start
    logging.info(f"Built and saved FAISS index to {index_path} in {elapsed:.2f} sec.")
    return index

def compute_blockwise_knn_merged(
    bg: np.ndarray,
    sample: np.ndarray,
    k_neighbors: int,
    bg_index_path: str | Path,
    sample_index_path: str | Path,
    rebuild_bg: bool = False,
    rebuild_sample: bool = True,
    save_blocks: bool = True,
    output_dir: str | Path = None,
    max_cached_neighbors: int = 20,  # <= фиксируем максимум
    nproc: int = 4
):
    """
    Return KNN distance matrix ((M+N)×k_neighbors).
    Precomputes and caches within-set distances (sample→sample, bg→bg)
    up to `max_cached_neighbors` nearest neighbors.
    """
    bg = np.ascontiguousarray(bg.astype('float32'))
    sample = np.ascontiguousarray(sample.astype('float32'))
    n_bg, n_sample = len(bg), len(sample)

    bg_index_path = Path(bg_index_path)
    sample_index_path = Path(sample_index_path)
    bg_dir = bg_index_path.parent
    sample_dir = sample_index_path.parent

    bg_prefix = bg_index_path.stem.replace("_faiss", "")
    sample_prefix = sample_index_path.stem.replace("_faiss", "")

    bg_matrix_path = bg_dir / f"{bg_prefix}_matrix.npy"
    sample_matrix_path = sample_dir / f"{sample_prefix}_matrix.npy"

    # === Build/load indices ===
    index_bg = build_or_load_index(bg, bg_index_path, rebuild=rebuild_bg)
    index_sample = build_or_load_index(sample, sample_index_path, rebuild=rebuild_sample)

    # === Helper functions ===
    def _merge_topk(A, B, k):
        if A.size == 0: allD = B
        elif B.size == 0: allD = A
        else: allD = np.concatenate([A, B], axis=1)
        k_eff = min(k, allD.shape[1])
        idx = np.argpartition(allD, kth=k_eff - 1, axis=1)[:, :k_eff]
        rows = np.arange(allD.shape[0])[:, None]
        topk = allD[rows, idx]
        topk.sort(axis=1)
        if k_eff < k:
            pad = np.full((allD.shape[0], k - k_eff), np.inf, dtype=allD.dtype)
            topk = np.concatenate([topk, pad], axis=1)
        return topk

    def _load_or_compute(name, path, func):
        if path.exists():
            logging.info(f"Loading cached {name} from {path}")
            arr = np.load(path)
            if arr.shape[1] >= k_neighbors:
                return arr[:, :k_neighbors]
            logging.warning(f"Cached {name} has only {arr.shape[1]} neighbors, recomputing for max_cached_neighbors={max_cached_neighbors}")
        logging.info(f"Computing {name} up to {max_cached_neighbors} neighbors...")
        arr = func()
        np.save(path, arr)
        logging.info(f"Saved {name} to {path}")
        return arr[:, :k_neighbors]

    # === Compute or load within-set blocks (with fixed max_cached_neighbors) ===
    D_s_s = _load_or_compute(
        "sample→sample",
        sample_matrix_path,
        lambda: np.sqrt(index_sample.search(sample, min(max_cached_neighbors + 1, max(1, n_sample)))[0])
    )
    if D_s_s.shape[1] >= 2:
        D_s_s = D_s_s[:, 1:1 + k_neighbors]
    else:
        D_s_s = np.empty((n_sample, 0), dtype=np.float32)

    D_b_b = _load_or_compute(
        "bg→bg",
        bg_matrix_path,
        lambda: np.sqrt(index_bg.search(bg, max_cached_neighbors)[0])
    )
    D_b_b = D_b_b[:, :k_neighbors]

    # === Cross-blocks always recomputed ===
    logging.info("Computing sample→bg and bg→sample (no cache)...")
    D_s_bg = np.sqrt(index_bg.search(sample, k_neighbors)[0])
    D_b_s = np.sqrt(index_sample.search(bg, min(k_neighbors, max(1, n_sample)))[0])

    # === Merge and return ===
    D_sample = _merge_topk(D_s_bg, D_s_s, k_neighbors)
    D_bg = _merge_topk(D_b_b, D_b_s, k_neighbors)
    distances = np.vstack([D_sample, D_bg]).astype(np.float32)

    if save_blocks and output_dir:
        output_dir = Path(output_dir)
        np.save(output_dir / "dist_merged.npy", distances)
        logging.info(f"Saved merged matrix to {output_dir / 'dist_merged.npy'}")

    return distances
