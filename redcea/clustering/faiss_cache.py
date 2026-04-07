from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import faiss


# ---------------- Strict helpers (no silent fixes) ----------------

def _require_finite(name: str, X: np.ndarray) -> None:
    if not np.isfinite(X).all():
        bad = np.where(~np.isfinite(X))
        coords = list(zip(bad[0][:10].tolist(), bad[1][:10].tolist()))
        raise ValueError(f"{name} contains NaN/inf (first positions): {coords}")


def _as_float32_contig(X: np.ndarray, name: str) -> np.ndarray:
    if not np.issubdtype(X.dtype, np.floating):
        raise TypeError(f"{name} must be floating dtype; got {X.dtype}")
    if X.dtype != np.float32:
        logging.info("%s: casting %s -> float32 for FAISS", name, X.dtype)
        X = X.astype(np.float32, copy=False)
    return np.ascontiguousarray(X)


def _squared_l2_to_l2_inplace(distances: np.ndarray, name: str) -> np.ndarray:
    if (distances < 0).any():
        bad = np.where(distances < 0)
        coords = list(zip(bad[0][:10].tolist(), bad[1][:10].tolist()))
        raise ValueError(f"{name} contains negative squared distances (first positions): {coords}")
    np.sqrt(distances, out=distances)
    return distances


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.save(tmp, arr)
    if not tmp.exists() and tmp.with_suffix(tmp.suffix + ".npy").exists():
        tmp = tmp.with_suffix(tmp.suffix + ".npy")
    os.replace(tmp, path)


# ---------------- Paths ----------------

@dataclass(frozen=True)
class SplitKnnPaths:
    sample_index_path: Path
    bg_index_path: Path

    # self-knn cache prefixes (without extensions)
    sample_self_prefix: Path
    bg_self_prefix: Path


def _default_paths(output_dir: Path, k_neighbors: int) -> SplitKnnPaths:
    """
    We suffix self-kNN caches with '__l2' to avoid mixing with old squared caches.
    """
    return SplitKnnPaths(
        sample_index_path=output_dir / "sample.index",
        bg_index_path=output_dir / "bg.index",
        sample_self_prefix=output_dir / f"knn_sample_sample__k{k_neighbors}__l2",
        bg_self_prefix=output_dir / f"knn_bg_bg__k{k_neighbors}__l2",
    )


def _knn_files(prefix: Path) -> Tuple[Path, Path, Path]:
    return (
        prefix.with_suffix(".indices.npy"),
        prefix.with_suffix(".distances.npy"),
        prefix.with_suffix(".meta.json"),
    )


# ---------------- FAISS index build/load ----------------

def _build_or_load_index(data: np.ndarray, index_path: Path, rebuild: bool, nproc: int) -> faiss.Index:
    faiss.omp_set_num_threads(int(nproc))

    if (not rebuild) and index_path.exists():
        logging.info("Loading FAISS index: %s", index_path)
        return faiss.read_index(str(index_path))

    logging.info("Building FAISS IndexFlatL2: %s (rebuild=%s)", index_path, rebuild)
    n, d = data.shape
    idx = faiss.IndexFlatL2(d)
    idx.add(data)
    tmp = index_path.with_suffix(index_path.suffix + ".tmp")
    faiss.write_index(idx, str(tmp))
    os.replace(tmp, index_path)
    return idx


# ---------------- self-kNN cache ----------------

def _load_self_knn(prefix: Path, mmap: bool = False) -> Tuple[dict, np.ndarray, np.ndarray]:
    idx_path, dist_path, meta_path = _knn_files(prefix)
    if not (idx_path.exists() and dist_path.exists() and meta_path.exists()):
        raise FileNotFoundError(f"Missing self-kNN cache files for prefix={prefix}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mmap_mode = "r" if mmap else None
    indices = np.load(idx_path, mmap_mode=mmap_mode)
    distances = np.load(dist_path, mmap_mode=mmap_mode)
    return meta, np.asarray(distances), np.asarray(indices)


def _save_self_knn(prefix: Path, distances_l2: np.ndarray, indices: np.ndarray, meta: dict) -> None:
    idx_path, dist_path, meta_path = _knn_files(prefix)
    _atomic_save_npy(idx_path, indices)
    _atomic_save_npy(dist_path, distances_l2)
    _atomic_write_json(meta_path, meta)


def _get_or_compute_self_knn(
    data: np.ndarray,
    index_path: Path,
    knn_prefix: Path,
    k: int,
    rebuild_index: bool,
    rebuild_knn: bool,
    nproc: int,
    mmap: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if (not rebuild_knn) and knn_prefix.with_suffix(".meta.json").exists():
        meta, dist, ind = _load_self_knn(knn_prefix, mmap=mmap)

        if meta.get("distance") != "l2":
            raise RuntimeError(
                f"Self-kNN cache distance='{meta.get('distance')}', expected 'l2'. "
                f"Likely old squared cache. Delete it or set rebuild_knn=True."
            )
        if meta.get("shape") != list(data.shape) or int(meta.get("k", -1)) != int(k):
            raise RuntimeError(
                f"Self-kNN cache mismatch for {knn_prefix}: meta.shape={meta.get('shape')} meta.k={meta.get('k')} "
                f"vs expected shape={list(data.shape)} k={k}. Set rebuild_knn=True."
            )
        if dist.shape != (data.shape[0], k) or ind.shape != (data.shape[0], k):
            raise RuntimeError(
                f"Self-kNN cache wrong array shapes: dist={dist.shape}, ind={ind.shape}, expected={(data.shape[0], k)}"
            )
        logging.info("Loaded cached self-kNN: %s", knn_prefix)
        return dist.astype(np.float32, copy=False), ind.astype(np.int32, copy=False)

    idx = _build_or_load_index(data, index_path=index_path, rebuild=rebuild_index, nproc=nproc)
    faiss.omp_set_num_threads(int(nproc))

    t0 = time.time()
    dist_sq, ind = idx.search(data, int(k))
    logging.info("Computed self-kNN: N=%d k=%d in %.2fs", data.shape[0], k, time.time() - t0)

    dist_sq = np.asarray(dist_sq, dtype=np.float32)
    ind = np.asarray(ind, dtype=np.int32)

    dist_l2 = _squared_l2_to_l2_inplace(dist_sq, name="self_knn_distances_sq")

    meta = {
        "shape": list(data.shape),
        "k": int(k),
        "distance": "l2",
        "created_at": time.time(),
        "index_path": str(index_path),
        "faiss": getattr(faiss, "__version__", "unknown"),
    }
    _save_self_knn(knn_prefix, dist_l2, ind, meta)
    logging.info("Saved cached self-kNN: %s", knn_prefix)
    return dist_l2, ind


# ---------------- cross-kNN (on the fly) ----------------

def _compute_cross_knn_l2(
    query: np.ndarray,
    db: np.ndarray,
    db_index_path: Path,
    rebuild_db_index: bool,
    k: int,
    nproc: int,
) -> Tuple[np.ndarray, np.ndarray]:
    db_index = _build_or_load_index(db, index_path=db_index_path, rebuild=rebuild_db_index, nproc=nproc)
    faiss.omp_set_num_threads(int(nproc))
    t0 = time.time()
    dist_sq, ind = db_index.search(query, int(k))
    logging.info("Computed cross kNN: queryN=%d dbN=%d k=%d in %.2fs", query.shape[0], db.shape[0], k, time.time() - t0)
    dist_sq = np.asarray(dist_sq, dtype=np.float32)
    ind = np.asarray(ind, dtype=np.int32)
    dist_l2 = _squared_l2_to_l2_inplace(dist_sq, name="cross_knn_distances_sq")
    return dist_l2, ind


# ---------------- public API ----------------


def compute_split_knn(
    bg,
    sample,
    k_neighbors: int,
    bg_index_path=None,
    sample_index_path=None,
    rebuild_bg: bool = False,
    rebuild_sample: bool = False,
    save_blocks: bool = True,
    output_dir=None,
    nproc: int = 1,
    mmap_cached: bool = False,
    rebuild_bg_knn: bool = False,
    rebuild_sample_knn: bool = False,
    bg_size_truncation: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute split kNN with required semantics:
      - cache sample index + sample-sample kNN (if save_blocks=True)
      - cache bg index     + bg-bg kNN        (if save_blocks=True)
      - compute cross on the fly: sample->bg and bg->sample (not cached)

    Returns:
      dist_ss, ind_ss, dist_bb, ind_bb, dist_sb, ind_sb, dist_bs, ind_bs

    Distances are L2 (NOT squared). Strict: NaN/inf -> raise.
    """

    if isinstance(sample, pd.DataFrame):
        sample_arr = sample.to_numpy()
    else:
        sample_arr = np.asarray(sample)

    if isinstance(bg, pd.DataFrame):
        bg_arr = bg.to_numpy()
    else:
        bg_arr = np.asarray(bg)

    sample_arr = _as_float32_contig(sample_arr, "sample")
    bg_arr = _as_float32_contig(bg_arr, "bg")

    _require_finite("sample", sample_arr)
    _require_finite("bg", bg_arr)

    # --- truncate bg BEFORE any indexing / knn ---
    truncN = None
    if bg_size_truncation is not None:
        truncN = int(bg_size_truncation)
        if truncN <= 0:
            raise ValueError("bg_size_truncation must be a positive integer.")
        if truncN > bg_arr.shape[0]:
            raise ValueError(f"bg_size_truncation={truncN} > len(bg)={bg_arr.shape[0]}.")
        bg_arr = np.ascontiguousarray(bg_arr[:truncN], dtype=np.float32)

    if output_dir is None and (bg_index_path is None or sample_index_path is None):
        raise ValueError(
            "Provide output_dir or explicit bg_index_path + sample_index_path, "
            "otherwise we don't know where to store indices/self-kNN caches."
        )

    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    paths = _default_paths(out, k_neighbors) if out is not None else None

    # resolve index paths robustly (avoid touching `paths` when output_dir is None)
    if sample_index_path is None:
        if paths is None:
            raise ValueError("sample_index_path must be provided when output_dir is None.")
        sample_index_path = paths.sample_index_path
    else:
        sample_index_path = Path(sample_index_path)

    if bg_index_path is None:
        if paths is None:
            raise ValueError("bg_index_path must be provided when output_dir is None.")
        bg_index_path = paths.bg_index_path
    else:
        bg_index_path = Path(bg_index_path)

    # helpers to namespace caches when bg is truncated (so we don't mix full-bg caches with trunc-bg caches)
    def _with_trunc_suffix(p: Path, n: int) -> Path:
        return p.with_name(p.stem + f".trunc{n}" + p.suffix)

    def _with_trunc_prefix(prefix, n: int):
        # prefix can be Path or str depending on your _default_paths
        if isinstance(prefix, Path):
            return prefix.with_name(prefix.name + f".trunc{n}")
        return str(prefix) + f".trunc{n}"

    # default prefixes for cached self-kNN
    sample_self_prefix = paths.sample_self_prefix if paths is not None else None
    bg_self_prefix = paths.bg_self_prefix if paths is not None else None

    # apply trunc namespace to bg-related caches/index
    if truncN is not None:
        bg_index_path = _with_trunc_suffix(bg_index_path, truncN)
        if save_blocks:
            if bg_self_prefix is None:
                raise ValueError("Internal error: bg_self_prefix is None while save_blocks=True.")
            bg_self_prefix = _with_trunc_prefix(bg_self_prefix, truncN)

    if save_blocks:
        if out is None or paths is None:
            raise ValueError("save_blocks=True requires output_dir (to store self-kNN arrays).")

        dist_ss, ind_ss = _get_or_compute_self_knn(
            data=sample_arr,
            index_path=sample_index_path,
            knn_prefix=sample_self_prefix,
            k=k_neighbors,
            rebuild_index=rebuild_sample,
            rebuild_knn=rebuild_sample_knn,
            nproc=nproc,
            mmap=mmap_cached,
        )

        dist_bb, ind_bb = _get_or_compute_self_knn(
            data=bg_arr,
            index_path=bg_index_path,
            knn_prefix=bg_self_prefix,
            k=k_neighbors,
            rebuild_index=rebuild_bg,
            rebuild_knn=rebuild_bg_knn,
            nproc=nproc,
            mmap=mmap_cached,
        )
    else:
        sample_index = _build_or_load_index(sample_arr, sample_index_path, rebuild_sample, nproc=nproc)
        bg_index = _build_or_load_index(bg_arr, bg_index_path, rebuild_bg, nproc=nproc)

        faiss.omp_set_num_threads(int(nproc))
        dist_ss_sq, ind_ss = sample_index.search(sample_arr, int(k_neighbors))
        dist_bb_sq, ind_bb = bg_index.search(bg_arr, int(k_neighbors))

        dist_ss = _squared_l2_to_l2_inplace(np.asarray(dist_ss_sq, dtype=np.float32), name="dist_ss_sq")
        dist_bb = _squared_l2_to_l2_inplace(np.asarray(dist_bb_sq, dtype=np.float32), name="dist_bb_sq")
        ind_ss = np.asarray(ind_ss, dtype=np.int32)
        ind_bb = np.asarray(ind_bb, dtype=np.int32)

    # cross kNNs computed on-the-fly, using (possibly truncated) bg_arr and its (possibly namespaced) index path
    dist_sb, ind_sb = _compute_cross_knn_l2(
        query=sample_arr,
        db=bg_arr,
        db_index_path=bg_index_path,
        rebuild_db_index=rebuild_bg,
        k=k_neighbors,
        nproc=nproc,
    )

    dist_bs, ind_bs = _compute_cross_knn_l2(
        query=bg_arr,
        db=sample_arr,
        db_index_path=sample_index_path,
        rebuild_db_index=rebuild_sample,
        k=k_neighbors,
        nproc=nproc,
    )

    return dist_ss, ind_ss, dist_bb, ind_bb, dist_sb, ind_sb, dist_bs, ind_bs



def compute_blockwise_knn_merged(*args, **kwargs):
    """
    Compatibility wrapper.
    New behavior: returns split kNN (ss, bb, sb, bs) with L2 distances.
    """
    return compute_split_knn(*args, **kwargs)
