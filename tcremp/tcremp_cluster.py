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


def get_k_neighbors_distance_matrix(data, n_neighbors=4):
    data = np.ascontiguousarray(data.astype('float32'))
    index = faiss.IndexFlatL2(data.shape[1])
    index.add(data)

    squared_distances, _ = index.search(data, n_neighbors)
    distances = np.sqrt(squared_distances)
    return distances


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


def compute_blockwise_distances(
    bg: np.ndarray,
    sample: np.ndarray,
    k_neighbors: int,
    bg_index_path: str | Path,
    sample_index_path: str | Path,
    rebuild_bg: bool = False,
    rebuild_sample: bool = True,
    save_blocks: bool = True,
    output_dir: str | Path = None,
):
    bg = np.ascontiguousarray(bg.astype('float32'))
    sample = np.ascontiguousarray(sample.astype('float32'))
    n_bg, n_sample = len(bg), len(sample)

    logging.info(f"=== Starting blockwise FAISS distance computation ===")
    logging.info(f"Background vectors: {n_bg}, sample vectors: {n_sample}, dim={bg.shape[1]}")

    t_total = time.time()

    # Build or load indices
    t0 = time.time()
    index_bg = build_or_load_index(bg, bg_index_path, rebuild=rebuild_bg)
    t_bg = time.time() - t0

    t0 = time.time()
    index_sample = build_or_load_index(sample, sample_index_path, rebuild=rebuild_sample)
    t_sample = time.time() - t0

    logging.info(f"Index setup: bg {t_bg:.2f}s, sample {t_sample:.2f}s")

    # Compute distances
    t0 = time.time()
    logging.info("Computing bg→bg distances...")
    D_bg_bg, _ = index_bg.search(bg, k_neighbors)
    t_bg_bg = time.time() - t0
    logging.info(f"bg→bg done in {t_bg_bg:.2f} sec.")

    t0 = time.time()
    logging.info("Computing sample→bg distances...")
    D_sample_bg, _ = index_bg.search(sample, k_neighbors)
    t_sbg = time.time() - t0
    logging.info(f"sample→bg done in {t_sbg:.2f} sec.")

    t0 = time.time()
    logging.info("Computing sample→sample distances...")
    D_sample_sample, _ = index_sample.search(sample, k_neighbors)
    t_ss = time.time() - t0
    logging.info(f"sample→sample done in {t_ss:.2f} sec.")

    # Combine into one block matrix
    t0 = time.time()
    logging.info("Combining distance blocks...")
    D_full = np.zeros((n_bg + n_sample, n_bg + n_sample), dtype=np.float32)
    D_full[:n_bg, :n_bg] = np.sqrt(D_bg_bg[:, [1]]) if D_bg_bg.ndim == 2 else 0
    D_full[:n_bg, n_bg:] = np.sqrt(D_sample_bg.T)
    D_full[n_bg:, :n_bg] = np.sqrt(D_sample_bg)
    D_full[n_bg:, n_bg:] = np.sqrt(D_sample_sample)
    t_combine = time.time() - t0
    logging.info(f"Blocks combined in {t_combine:.2f} sec.")

    if save_blocks and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "D_bg_bg.npy", D_bg_bg)
        np.save(output_dir / "D_sample_bg.npy", D_sample_bg)
        np.save(output_dir / "D_sample_sample.npy", D_sample_sample)
        np.save(output_dir / "D_full.npy", D_full)
        logging.info(f"Saved distance blocks to {output_dir}")

    t_total = time.time() - t_total
    logging.info(f"=== Total blockwise FAISS time: {t_total:.2f} sec ===")

    return D_full
