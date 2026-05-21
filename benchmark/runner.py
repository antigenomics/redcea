from __future__ import annotations

import json
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from redcea.clustering.cdr3_grouping import build_len_to_group_id, compute_cdr3_len, map_len_to_group_id
from redcea.clustering.cluster_methods import run_leiden_clustering
from redcea.clustering.eps_estimation import (
    cluster_dbscan,
    eps_per_point_from_group_id,
    estimate_dbscan_eps,
    estimate_eps_by_group_flexible,
)
from redcea.clustering.preprocess import prepare_data_for_clustering
from redcea.clustering.vdbscan import vdbscan_from_knn


ASSIGNMENT_OUTPUT_COLUMNS = [
    "run_id",
    "dataset",
    "dataset_mode",
    "epitope",
    "donor_id",
    "method",
    "parameter_json",
    "clonotype_id",
    "cdr3",
    "cdr3_length",
    "v_gene",
    "j_gene",
    "chain",
    "sample_label",
    "sample_type",
    "timepoint",
    "subject_id",
    "replicate_id",
    "truth_label",
    "label",
    "cluster_id",
    "is_noise",
    "epsilon_used",
    "leiden_cluster_id",
    "dbscan_cluster_id",
    "vdbscan_cluster_id",
    "local_dbscan_cluster_id",
    "local_vdbscan_cluster_id",
    "local_leiden_cluster_id",
    "final_cluster_id",
]


def extract_embeddings(df: pd.DataFrame) -> np.ndarray:
    if "embedding" in df.columns:
        return np.asarray(df["embedding"].tolist(), dtype=np.float32)
    emb_cols = [col for col in df.columns if str(col).startswith("emb_")]
    if emb_cols:
        return df[emb_cols].to_numpy(dtype=np.float32, copy=False)

    metadata_tokens = [
        "id",
        "index",
        "clone",
        "count",
        "freq",
        "fraction",
        "size",
        "length",
        "len",
        "timepoint",
        "padj",
        "pvalue",
        "qvalue",
    ]
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    embedding_cols = [
        col for col in numeric_cols if not any(token in str(col).lower() for token in metadata_tokens)
    ]
    if not embedding_cols:
        raise KeyError("Expected an 'embedding' column, 'emb_*' columns, or numeric coordinate columns.")
    return df[embedding_cols].to_numpy(dtype=np.float32, copy=False)


def compact_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _knn(embeddings: np.ndarray, k: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    n_neighbors = min(max(2, int(k)), len(embeddings))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    distances, indices = nn.fit(embeddings).kneighbors(embeddings)
    return distances.astype(np.float32), indices.astype(np.int64)


def _relabel(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    is_noise = labels < 0
    out = labels.copy()
    if (~is_noise).any():
        _, relabeled = np.unique(out[~is_noise], return_inverse=True)
        out[~is_noise] = relabeled
    return out, is_noise


def _cdr3_lengths(frame: pd.DataFrame) -> np.ndarray:
    if "cdr3_length" in frame.columns:
        return frame["cdr3_length"].to_numpy(dtype=np.int32, copy=False)
    if "cdr3" not in frame.columns:
        raise KeyError("Expected 'cdr3' or 'cdr3_length' in benchmark metadata.")
    return compute_cdr3_len(frame["cdr3"])


def _estimate_eps(distances: np.ndarray, min_samples: int, strategy: str, percentile: float | None) -> float:
    kth = distances[:, min(min_samples - 1, distances.shape[1] - 1)]
    if strategy == "knee":
        eps = float(estimate_dbscan_eps(data=None, distances=kth, n_neighbors=min_samples))
    elif strategy == "percentile":
        if percentile is None:
            raise ValueError("percentile must be provided for percentile epsilon strategy")
        eps = float(np.percentile(kth, percentile))
    else:
        raise ValueError(f"Unknown epsilon strategy: {strategy}")
    if np.isfinite(eps) and eps > 0:
        return eps
    positive = kth[np.isfinite(kth) & (kth > 0)]
    if len(positive):
        return float(np.min(positive))
    return float(np.finfo(np.float32).eps)


def _run_dbscan_existing(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    *,
    min_samples: int,
    epsilon_strategy: str,
    percentile: float | None = None,
    distance_metric: str = "euclidean",
) -> pd.DataFrame:
    distances, _ = _knn(embeddings, max(min_samples + 1, 2), distance_metric)
    eps = _estimate_eps(distances, min_samples, epsilon_strategy, percentile)
    cluster_id, is_noise = _relabel(cluster_dbscan(embeddings, eps=eps, min_samples=min_samples))
    out = metadata.copy().reset_index(drop=True)
    out["cluster_id"] = cluster_id
    out["is_noise"] = is_noise
    out["epsilon_used"] = float(eps)
    return out


def _run_vdbscan_existing(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    *,
    min_samples: int,
    epsilon_strategy: str,
    percentile: float | None = None,
    min_group_size: int = 100,
    distance_metric: str = "euclidean",
) -> pd.DataFrame:
    out = metadata.copy().reset_index(drop=True)
    lengths = _cdr3_lengths(out)
    out["cdr3_length"] = lengths
    distances, indices = _knn(embeddings, max(min_samples + 1, 2), distance_metric)

    if epsilon_strategy == "knee":
        len_to_gid = build_len_to_group_id(lengths, min_frac=max(1.0 / len(lengths), min_group_size / max(1, len(lengths))))
        gid = map_len_to_group_id(lengths, len_to_gid, unknown_len_policy="nearest")
        eps_by_gid = estimate_eps_by_group_flexible(gid, distances, kth_neighbor=min_samples)
        eps_i = eps_per_point_from_group_id(gid, eps_by_gid)
        raw = vdbscan_from_knn(
            knn_indices=indices,
            knn_distances_l2=distances,
            eps_i_l2=eps_i,
            num_points_for_core=min_samples,
            sym_rule="asymmetric",
        )
        cluster_id, is_noise = _relabel(raw)
        out["cluster_id"] = cluster_id
        out["is_noise"] = is_noise
        out["epsilon_used"] = eps_i
        return out

    final_cluster_id = np.full(len(out), -1, dtype=int)
    final_is_noise = np.ones(len(out), dtype=bool)
    epsilon_used = np.full(len(out), np.nan, dtype=float)
    offset = 0
    global_eps = _estimate_eps(distances, min_samples, "percentile", percentile)
    for length_value, idx in out.groupby("cdr3_length").groups.items():
        idx = np.asarray(list(idx), dtype=int)
        group_distances = distances[idx]
        group_embeddings = embeddings[idx]
        eps = global_eps if len(idx) < max(min_group_size, min_samples) else _estimate_eps(
            group_distances,
            min_samples,
            "percentile",
            percentile,
        )
        labels = cluster_dbscan(group_embeddings, eps=eps, min_samples=min_samples)
        cluster_id, is_noise = _relabel(labels)
        keep = ~is_noise
        if keep.any():
            final_cluster_id[idx[keep]] = cluster_id[keep] + offset
            offset += int(cluster_id[keep].max()) + 1
        final_is_noise[idx] = is_noise
        epsilon_used[idx] = eps
    out["cluster_id"] = final_cluster_id
    out["is_noise"] = final_is_noise
    out["epsilon_used"] = epsilon_used
    return out


def _run_leiden_existing(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    *,
    k: int,
    resolution: float,
    distance_metric: str = "euclidean",
    min_cluster_size: int | None = None,
) -> pd.DataFrame:
    out = metadata.copy().reset_index(drop=True)
    if len(out) == 0:
        out["cluster_id"] = np.asarray([], dtype=int)
        out["is_noise"] = np.asarray([], dtype=bool)
        return out
    if len(out) == 1:
        out["cluster_id"] = np.asarray([0], dtype=int)
        out["is_noise"] = np.asarray([False if min_cluster_size is None or min_cluster_size <= 1 else True], dtype=bool)
        return out
    distances, indices = _knn(embeddings, max(k + 1, 2), distance_metric)
    labels = run_leiden_clustering(
        knn_indices=indices,
        knn_distances=distances,
        resolution=resolution,
        metric="dissimilarity",
        n_threads=1,
        min_cluster_size=min_cluster_size,
    )
    cluster_id, is_noise = _relabel(labels)
    out["cluster_id"] = cluster_id
    out["is_noise"] = is_noise if min_cluster_size is not None else False
    return out


def _hybrid_partition(
    base_assignments: pd.DataFrame,
    subset_runner: Callable[[np.ndarray, pd.DataFrame], pd.DataFrame],
    embeddings: np.ndarray,
    *,
    base_cluster_col: str,
    local_cluster_col: str,
) -> pd.DataFrame:
    out = base_assignments.copy().reset_index(drop=True)
    out[local_cluster_col] = -1
    out["final_cluster_id"] = -1
    out["is_noise"] = out["is_noise"].astype(bool)
    offset = 0
    for cluster_value, subset in out.loc[~out["is_noise"]].groupby(base_cluster_col):
        idx = subset.index.to_numpy(dtype=int)
        local = subset_runner(embeddings[idx], subset.reset_index(drop=True))
        local_cluster = local["cluster_id"].to_numpy(dtype=int)
        local_noise = local["is_noise"].to_numpy(dtype=bool)
        out.loc[idx, local_cluster_col] = local_cluster
        out.loc[idx, "is_noise"] = local_noise
        keep = ~local_noise
        if keep.any():
            out.loc[idx[keep], "final_cluster_id"] = local_cluster[keep] + offset
            offset += int(local_cluster[keep].max()) + 1
    return out


def standardized_assignment_table(
    assignments: pd.DataFrame,
    dataset: str,
    dataset_mode: str,
    method: str,
    parameter_json: str,
    epitope: str | None,
    donor_id: str | None,
) -> pd.DataFrame:
    out = assignments.copy().reset_index(drop=True)
    out["dataset"] = dataset
    out["dataset_mode"] = dataset_mode
    out["epitope"] = epitope
    out["donor_id"] = donor_id
    out["method"] = method
    out["parameter_json"] = parameter_json
    out["cdr3_length"] = _cdr3_lengths(out)
    for col, default in [("sample_label", None), ("truth_label", None), ("cluster_id", -1), ("is_noise", True)]:
        if col not in out.columns:
            out[col] = default
    return out


def slim_assignment_table(assignments: pd.DataFrame, run_id: str) -> pd.DataFrame:
    out = assignments.copy()
    if "run_id" not in out.columns:
        out.insert(0, "run_id", run_id)
    embedding_columns = [
        column
        for column in out.columns
        if str(column).startswith("emb_") or str(column) == "embedding" or str(column) == "embedding_id"
    ]
    if embedding_columns:
        out = out.drop(columns=embedding_columns)
    columns = [column for column in ASSIGNMENT_OUTPUT_COLUMNS if column in out.columns]
    extra_cluster_columns = [
        column
        for column in out.columns
        if column not in columns
        and (
            str(column).endswith("_cluster_id")
            or str(column).startswith("local_")
            or str(column).startswith("epsilon")
        )
    ]
    columns.extend(extra_cluster_columns)
    return out.loc[:, columns].copy()


@dataclass
class ClusteringRunResult:
    run_id: str
    assignments: pd.DataFrame
    metadata: dict[str, Any]


class ClusteringBenchmarkRunner:
    def __init__(
        self,
        *,
        assignments_dir: str | Path = "results/clustering_assignments",
        metadata_path: str | Path = "results/run_metadata/clustering_runs.tsv",
        metadata_parts_dir: str | Path = "results/run_metadata/clustering_run_parts",
        pca_components: int = 50,
    ) -> None:
        self.assignments_dir = Path(assignments_dir)
        self.metadata_path = Path(metadata_path)
        self.metadata_parts_dir = Path(metadata_parts_dir)
        self.pca_components = int(pca_components)
        self.assignments_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_parts_dir.mkdir(parents=True, exist_ok=True)

    def _run_method(self, method: str, embeddings: np.ndarray, frame: pd.DataFrame, parameters: dict[str, Any]) -> pd.DataFrame:
        if method == "dbscan":
            return _run_dbscan_existing(embeddings, frame, **parameters)
        if method == "vdbscan_length":
            return _run_vdbscan_existing(embeddings, frame, **parameters)
        if method == "leiden":
            return _run_leiden_existing(embeddings, frame, **parameters)
        if method == "leiden_min_size":
            return _run_leiden_existing(embeddings, frame, **parameters)
        if method == "leiden_dbscan":
            base = _run_leiden_existing(
                embeddings,
                frame,
                k=parameters["leiden_k"],
                resolution=parameters["leiden_resolution"],
                distance_metric=parameters.get("distance_metric", "euclidean"),
            ).rename(columns={"cluster_id": "leiden_cluster_id"})
            out = _hybrid_partition(
                base,
                lambda local_emb, local_df: _run_dbscan_existing(
                    local_emb,
                    local_df,
                    min_samples=parameters["dbscan_min_samples"],
                    epsilon_strategy=parameters["epsilon_strategy"],
                    percentile=parameters.get("percentile"),
                    distance_metric=parameters.get("distance_metric", "euclidean"),
                ),
                embeddings,
                base_cluster_col="leiden_cluster_id",
                local_cluster_col="local_dbscan_cluster_id",
            )
            out["cluster_id"] = out["final_cluster_id"]
            return out
        if method == "leiden_vdbscan":
            base = _run_leiden_existing(
                embeddings,
                frame,
                k=parameters["leiden_k"],
                resolution=parameters["leiden_resolution"],
                distance_metric=parameters.get("distance_metric", "euclidean"),
            ).rename(columns={"cluster_id": "leiden_cluster_id"})
            out = _hybrid_partition(
                base,
                lambda local_emb, local_df: _run_vdbscan_existing(
                    local_emb,
                    local_df,
                    min_samples=parameters["vdbscan_min_samples"],
                    epsilon_strategy=parameters["epsilon_strategy"],
                    percentile=parameters.get("percentile"),
                    min_group_size=parameters.get("min_group_size", 100),
                    distance_metric=parameters.get("distance_metric", "euclidean"),
                ),
                embeddings,
                base_cluster_col="leiden_cluster_id",
                local_cluster_col="local_vdbscan_cluster_id",
            )
            out["cluster_id"] = out["final_cluster_id"]
            return out
        if method == "dbscan_leiden":
            base = _run_dbscan_existing(
                embeddings,
                frame,
                min_samples=parameters["dbscan_min_samples"],
                epsilon_strategy=parameters["epsilon_strategy"],
                percentile=parameters.get("percentile"),
                distance_metric=parameters.get("distance_metric", "euclidean"),
            ).rename(columns={"cluster_id": "dbscan_cluster_id"})
            out = _hybrid_partition(
                base,
                lambda local_emb, local_df: _run_leiden_existing(
                    local_emb,
                    local_df,
                    k=min(parameters["leiden_k"], max(2, len(local_df) - 1)),
                    resolution=parameters["leiden_resolution"],
                    distance_metric=parameters.get("distance_metric", "euclidean"),
                ),
                embeddings,
                base_cluster_col="dbscan_cluster_id",
                local_cluster_col="local_leiden_cluster_id",
            )
            out["cluster_id"] = out["final_cluster_id"]
            return out
        if method == "vdbscan_leiden":
            base = _run_vdbscan_existing(
                embeddings,
                frame,
                min_samples=parameters["vdbscan_min_samples"],
                epsilon_strategy=parameters["epsilon_strategy"],
                percentile=parameters.get("percentile"),
                min_group_size=parameters.get("min_group_size", 100),
                distance_metric=parameters.get("distance_metric", "euclidean"),
            ).rename(columns={"cluster_id": "vdbscan_cluster_id"})
            out = _hybrid_partition(
                base,
                lambda local_emb, local_df: _run_leiden_existing(
                    local_emb,
                    local_df,
                    k=min(parameters["leiden_k"], max(2, len(local_df) - 1)),
                    resolution=parameters["leiden_resolution"],
                    distance_metric=parameters.get("distance_metric", "euclidean"),
                ),
                embeddings,
                base_cluster_col="vdbscan_cluster_id",
                local_cluster_col="local_leiden_cluster_id",
            )
            out["cluster_id"] = out["final_cluster_id"]
            return out
        raise KeyError("Unknown benchmark method: {0}".format(method))

    def run(
        self,
        dataset_df: pd.DataFrame,
        *,
        dataset: str,
        dataset_mode: str,
        method: str,
        parameters: dict[str, Any],
        epitope: str | None = None,
        donor_id: str | None = None,
        run_id: str | None = None,
    ) -> ClusteringRunResult:
        parameter_json = compact_json(parameters)
        run_id = run_id or "{0}_{1}_{2}".format(dataset_mode, method, uuid.uuid4().hex[:10])
        started = time.perf_counter()
        error_traceback = ""
        try:
            embeddings = extract_embeddings(dataset_df)
            embeddings = prepare_data_for_clustering(pd.DataFrame(embeddings), n_components=min(self.pca_components, embeddings.shape[1]))
            raw = self._run_method(method, embeddings, dataset_df.copy(), parameters)
            assignments = standardized_assignment_table(raw, dataset, dataset_mode, method, parameter_json, epitope, donor_id)
            status = "success"
            error_message = ""
        except Exception as exc:
            error_traceback = traceback.format_exc()
            assignments = standardized_assignment_table(
                dataset_df.copy().assign(cluster_id=-1, is_noise=True),
                dataset,
                dataset_mode,
                method,
                parameter_json,
                epitope,
                donor_id,
            )
            status = "error"
            error_message = "{0}: {1}".format(type(exc).__name__, str(exc))
        runtime_seconds = time.perf_counter() - started
        n_noise = int(assignments["is_noise"].sum())
        n_clusters = int(assignments.loc[~assignments["is_noise"], "cluster_id"].nunique())
        metadata = {
            "run_id": run_id,
            "dataset": dataset,
            "method": method,
            "parameter_json": parameter_json,
            "n_points": int(len(assignments)),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_fraction": float(n_noise / max(1, len(assignments))),
            "runtime_seconds": float(runtime_seconds),
            "status": status,
            "error_message": error_message,
            "error_traceback": error_traceback,
        }
        self._save(run_id, assignments, metadata)
        return ClusteringRunResult(run_id=run_id, assignments=assignments, metadata=metadata)

    def _save(self, run_id: str, assignments: pd.DataFrame, metadata: dict[str, Any]) -> None:
        out = slim_assignment_table(assignments, run_id)
        out.to_parquet(self.assignments_dir / "{0}.parquet".format(run_id), index=False)
        pd.DataFrame([metadata]).to_csv(
            self.metadata_parts_dir / "{0}.tsv".format(run_id),
            sep="\t",
            index=False,
        )
