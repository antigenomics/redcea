import logging
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# === copied 1:1 from your snippet ===

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


def prepare_data_for_clustering(df: pd.DataFrame, n_components):
    df = standardize_data(df.values)
    df = apply_pca(df, n_components=n_components)
    return df
