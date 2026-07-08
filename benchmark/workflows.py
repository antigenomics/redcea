from __future__ import annotations

import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from benchmark.airr_utils import (
    best_clone_id_key,
    clone_id_candidates,
    read_airr_like_table,
    standardize_metadata_frame,
)
from benchmark.evaluation import (
    _compute_vdjdb_metrics_for_frame,
    compute_cross_donor_overlap,
    compute_known_yfv_recovery,
    compute_vdjdb_metrics,
    compute_yfv_cluster_enrichment,
    summarize_yfv_enrichment,
)
from benchmark.plotting import (
    plot_cross_donor_overlap,
    plot_density_by_length,
    plot_final_method_summary,
    plot_vdjdb_benchmark,
    plot_vdjdb_signal_concentration,
    plot_yfv_enrichment_summary,
    plot_yfv_known_recovery_heatmap,
)
from benchmark.data_sources import DEFAULT_TCRVDB_PADJ_THRESHOLD, DEFAULT_TCRVDB_PATH
from benchmark.import_archived_redcea_results import parse_run_identity
from benchmark.prepare_datasets import build_vdjdb_truth_table
from benchmark.run_benchmark import consolidate_run_metadata
from benchmark.run_benchmark import standardize_redcea_assignments
from benchmark.data_sources import REPO_ROOT


ASSIGNMENT_EVAL_COLUMNS = [
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
    "truth_label",
    "cluster_id",
    "is_noise",
]


def log_step(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[{0}] {1}".format(timestamp, message), flush=True)


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def ensure_output_dirs() -> None:
    for path in [
        Path("results/clustering_assignments"),
        Path("results/run_metadata"),
        Path("results/enrichment"),
        Path("results/metrics"),
        Path("results/density"),
        Path("figures/clustering_strategy"),
        Path("reports"),
    ]:
        repo_path(path).mkdir(parents=True, exist_ok=True)
    return None


def read_assignment_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        return pd.read_parquet(path)
    try:
        import pyarrow.parquet as pq

        available_columns = set(pq.ParquetFile(path).schema.names)
        selected_columns = [column for column in columns if column in available_columns]
        missing_columns = [column for column in columns if column not in available_columns]
        if missing_columns:
            log_step("Assignment parquet missing optional columns {0}: {1}".format(missing_columns, path))
        return pd.read_parquet(path, columns=selected_columns)
    except Exception as exc:
        log_step(
            "Could not inspect parquet schema cheaply for {0}; falling back to full read then column subset. Error: {1}".format(
                path,
                exc,
            )
        )
        frame = pd.read_parquet(path)
        selected_columns = [column for column in columns if column in frame.columns]
        return frame[selected_columns].copy()


def load_assignment_tables(
    assignments_dir: str | Path = "results/clustering_assignments",
    *,
    metadata_parts_dir: str | Path = "results/run_metadata/clustering_run_parts",
    metadata_path: str | Path = "results/run_metadata/clustering_runs.tsv",
    only_success: bool = True,
    dataset_mode: str | None = None,
    columns: list[str] | None = ASSIGNMENT_EVAL_COLUMNS,
    consolidate_metadata: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()
    assignments_root = repo_path(assignments_dir)
    metadata_parts_root = repo_path(metadata_parts_dir)
    metadata_out = repo_path(metadata_path)
    if consolidate_metadata:
        log_step("Consolidating run metadata from {0}".format(metadata_parts_root))
        run_metadata = consolidate_run_metadata(
            metadata_parts_dir=metadata_parts_root,
            metadata_path=metadata_out,
        )
    else:
        log_step("Loading run metadata without consolidation: {0}".format(metadata_out))
        run_metadata = pd.read_csv(metadata_out, sep="\t")
    if len(run_metadata):
        status_counts = run_metadata["status"].value_counts(dropna=False).to_dict()
    else:
        status_counts = {}
    log_step("Loaded run metadata rows={0}, status_counts={1}".format(len(run_metadata), status_counts))
    selected_metadata = run_metadata.copy()
    if dataset_mode is not None and len(selected_metadata) and "dataset_mode" in selected_metadata.columns:
        selected_metadata = selected_metadata.loc[selected_metadata["dataset_mode"] == dataset_mode].copy()
    elif dataset_mode is not None and len(selected_metadata) and "dataset" in selected_metadata.columns:
        if dataset_mode == "vdjdb":
            selected_metadata = selected_metadata.loc[selected_metadata["dataset"].astype(str).str.startswith("vdjdb")].copy()
        elif dataset_mode == "yfv":
            selected_metadata = selected_metadata.loc[selected_metadata["dataset"].astype(str).eq("yfv_repertoires")].copy()
    if only_success and len(selected_metadata) and "status" in selected_metadata.columns:
        selected_metadata = selected_metadata.loc[selected_metadata["status"] == "success"].copy()
    selected_run_ids = selected_metadata["run_id"].dropna().astype(str).drop_duplicates().tolist() if len(selected_metadata) else []
    frames = []
    assignment_paths = [assignments_root / f"{run_id}.parquet" for run_id in selected_run_ids]
    existing_assignment_paths = [path for path in assignment_paths if path.exists()]
    missing_assignment_paths = len(assignment_paths) - len(existing_assignment_paths)
    log_step(
        "Selecting assignment parquet files in {0}: dataset_mode={1}, only_success={2}, selected_runs={3}, existing_files={4}, missing_files={5}, columns={6}".format(
            assignments_root,
            dataset_mode,
            only_success,
            len(selected_run_ids),
            len(existing_assignment_paths),
            missing_assignment_paths,
            "all" if columns is None else len(columns),
        )
    )
    loaded_paths = 0
    for path in existing_assignment_paths:
        frames.append(read_assignment_parquet(path, columns=columns))
        loaded_paths += 1
    assignments = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log_step(
        "Loaded assignment tables: files={0}, rows={1}, columns={2}".format(
            loaded_paths,
            len(assignments),
            len(assignments.columns),
        )
    )
    return assignments, selected_metadata


def _load_metadata_lookup(metadata_paths: list[Path | None]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen_paths: set[Path] = set()
    for path in metadata_paths:
        if path is None:
            continue
        resolved = Path(path)
        if resolved in seen_paths or not resolved.exists():
            continue
        seen_paths.add(resolved)
        try:
            frames.append(pd.read_csv(resolved, sep="\t"))
        except Exception as exc:
            log_step("Skipping unreadable metadata source {0}: {1}".format(resolved, exc))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "run_id" not in combined.columns:
        return pd.DataFrame()
    return combined.drop_duplicates(subset=["run_id"], keep="last").reset_index(drop=True)


def _compute_vdjdb_metrics_from_redcea_runs(
    *,
    assignments_dir: str | Path,
    redcea_runs_dir: str | Path,
    tcrvdb_path: str | Path = DEFAULT_TCRVDB_PATH,
    padj_threshold: float = DEFAULT_TCRVDB_PADJ_THRESHOLD,
    columns: list[str] | None = ASSIGNMENT_EVAL_COLUMNS,
) -> pd.DataFrame:
    ensure_output_dirs()
    assignments_root = repo_path(assignments_dir)
    assignments_root.mkdir(parents=True, exist_ok=True)
    redcea_runs_root = repo_path(redcea_runs_dir)
    truth_table = build_vdjdb_truth_table(tcrvdb_path, padj_threshold=padj_threshold)

    rows: list[dict[str, object]] = []
    rebuilt_assignments = 0
    reused_assignments = 0
    skipped_runs = 0
    unreadable_runs = 0

    if not redcea_runs_root.exists():
        log_step("redcea_runs directory does not exist, skipping direct VDJdb scan: {0}".format(redcea_runs_root))
        return pd.DataFrame()

    run_dirs = [
        path
        for path in sorted(redcea_runs_root.iterdir())
        if path.is_dir() and path.name.startswith("vdjdb_")
    ]
    for run_dir in tqdm(run_dirs, desc="VDJdb runs", mininterval=0.5):
        run_id = run_dir.name
        cluster_path = run_dir / "{0}_tcremp_clusters.tsv".format(run_id)
        if not cluster_path.exists():
            skipped_runs += 1
            continue
        try:
            run_info = parse_run_identity(run_id)
        except Exception as exc:
            unreadable_runs += 1
            log_step("Skipping VDJdb run with unparseable run_id={0}: {1}".format(run_id, exc))
            continue

        dataset = str(run_info["dataset"])
        dataset_mode = str(run_info["dataset_mode"])
        method = str(run_info["method"])
        parameter_json = "{}"
        runtime_seconds = float(np.nan)
        assignment_path = assignments_root / "{0}.parquet".format(run_id)

        if assignment_path.exists():
            assignments = read_assignment_parquet(assignment_path, columns=columns)
            reused_assignments += 1
        else:
            cluster_df = pd.read_csv(cluster_path, sep="\t")
            assignments = standardize_redcea_assignments(
                cluster_df,
                run_id=run_id,
                dataset=dataset,
                dataset_mode=dataset_mode,
                method=method,
                parameter_json=parameter_json,
                epitope=str(run_info["epitope"]),
                donor_id=None,
                truth_table=truth_table,
            )
            assignments.to_parquet(assignment_path, index=False)
            if columns is not None:
                selected_columns = [column for column in columns if column in assignments.columns]
                assignments = assignments[selected_columns].copy()
            rebuilt_assignments += 1

        compact_metadata = {
            "run_id": run_id,
            "n_clusters": int(assignments.loc[~assignments["is_noise"], "cluster_id"].nunique()) if len(assignments) else 0,
            "noise_fraction": float(assignments["is_noise"].mean()) if len(assignments) else 0.0,
            "runtime_seconds": runtime_seconds,
        }
        metric_row = _compute_vdjdb_metrics_for_frame(assignments, compact_metadata)
        if metric_row is not None:
            rows.append(metric_row)

    metrics_df = pd.DataFrame(rows)
    summary_bits = [
        "runs={0}".format(len(metrics_df)),
        "reused_assignments={0}".format(reused_assignments),
    ]
    if rebuilt_assignments:
        summary_bits.append("rebuilt_assignments={0}".format(rebuilt_assignments))
    if skipped_runs:
        summary_bits.append("skipped_without_clusters={0}".format(skipped_runs))
    if unreadable_runs:
        summary_bits.append("skipped_unparseable={0}".format(unreadable_runs))
    log_step("Computed VDJdb metrics from redcea_runs: {0}".format(", ".join(summary_bits)))
    return metrics_df


def compute_density_by_length(
    processed_dir: str | Path = "data/processed",
    *,
    k: int = 5,
    max_points_per_facet: int | None = None,
    random_state: int = 17,
) -> pd.DataFrame:
    processed_dir = repo_path(processed_dir)
    rng = np.random.default_rng(random_state)
    log_step(
        "Computing density by CDR3 length from {0}; k={1}, max_points_per_facet={2}; using existing FAISS indexes only".format(
            processed_dir,
            k,
            max_points_per_facet,
        )
    )

    rows: list[dict[str, object]] = []

    def _safe_manifest_path(row_value) -> Path | None:
        if pd.isna(row_value):
            return None
        value = str(row_value).strip()
        if not value:
            return None
        return Path(value)

    def read_parquet_metadata_only(path: Path) -> tuple[pd.DataFrame, int]:
        try:
            import pyarrow.parquet as pq
        except Exception as exc:
            raise RuntimeError("pyarrow is required for metadata-only parquet reads") from exc

        parquet_file = pq.ParquetFile(path)
        row_count = int(parquet_file.metadata.num_rows)
        keep_tokens = (
            "clone",
            "clonotype",
            "cdr3",
            "junction",
            "v_gene",
            "j_gene",
            "v_call",
            "j_call",
            "v.segm",
            "j.segm",
            "trbv",
            "trbj",
            "chain",
            "locus",
        )
        columns = [
            name
            for name in parquet_file.schema.names
            if any(token in str(name).lower() for token in keep_tokens)
            and not str(name).startswith("emb_")
            and str(name) not in {"embedding", "embedding_id"}
        ]
        if not columns:
            return pd.DataFrame(index=pd.RangeIndex(row_count)), row_count
        return pd.read_parquet(path, columns=columns), row_count

    def reconstruct_index_rows(index, row_indices: np.ndarray) -> np.ndarray:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if len(row_indices) == 0:
            return np.empty((0, int(index.d)), dtype=np.float32)
        out = np.empty((len(row_indices), int(index.d)), dtype=np.float32)
        try:
            for output_position, index_position in enumerate(row_indices):
                index.reconstruct(int(index_position), out[output_position])
        except TypeError:
            for output_position, index_position in enumerate(row_indices):
                out[output_position] = np.asarray(index.reconstruct(int(index_position)), dtype=np.float32)
        return out

    def kth_distances_from_existing_index(
        *,
        index_path: Path | None,
        n_neighbors: int,
        index_expected_rows: int,
        row_indices: np.ndarray,
        batch_size: int = 100000,
    ) -> np.ndarray:
        if index_path is None or not index_path.exists():
            raise FileNotFoundError("FAISS index is missing: {0}".format(index_path))
        try:
            import faiss  # type: ignore
        except Exception as exc:
            raise RuntimeError("faiss is required to reuse existing index files") from exc

        index = faiss.read_index(str(index_path))
        if int(index.ntotal) != int(index_expected_rows):
            raise ValueError(
                "FAISS index row mismatch for {0}: index.ntotal={1}, expected_rows={2}".format(
                    index_path,
                    index.ntotal,
                    index_expected_rows,
                )
            )
        row_indices = np.asarray(row_indices, dtype=np.int64)
        kth_index = min(int(k), n_neighbors - 1)
        out = np.empty(len(row_indices), dtype=np.float32)
        for start in range(0, len(row_indices), batch_size):
            stop = min(start + batch_size, len(row_indices))
            query = reconstruct_index_rows(index, row_indices[start:stop])
            dist_sq, _ = index.search(query, n_neighbors)
            dist_sq = np.maximum(np.asarray(dist_sq[:, kth_index], dtype=np.float32), 0.0)
            out[start:stop] = np.sqrt(dist_sq)
        return out

    def process_facet(facet_label: str, cdr3_lengths: pd.Series, index_path: Path | None = None) -> None:
        cdr3_lengths = cdr3_lengths.reset_index(drop=True)
        if len(cdr3_lengths) < 2:
            log_step("Skipping density facet {0}: rows={1}".format(facet_label, len(cdr3_lengths)))
            return
        n_total = len(cdr3_lengths)
        row_indices = np.arange(n_total, dtype=np.int64)
        if max_points_per_facet is not None and n_total > max_points_per_facet:
            sampled_index = rng.choice(n_total, size=max_points_per_facet, replace=False)
            row_indices = np.sort(sampled_index).astype(np.int64, copy=False)
            cdr3_lengths = cdr3_lengths.iloc[row_indices].reset_index(drop=True)
        log_step(
            "Density facet {0}: total_rows={1}, used_rows={2}, index={3}".format(
                facet_label,
                n_total,
                len(cdr3_lengths),
                index_path,
            )
        )
        n_neighbors = min(max(2, int(k) + 1), n_total)
        try:
            knn_distances = kth_distances_from_existing_index(
                index_path=index_path,
                n_neighbors=n_neighbors,
                index_expected_rows=n_total,
                row_indices=row_indices,
            )
            log_step("Density facet {0}: reused FAISS index {1}".format(facet_label, index_path))
        except Exception as exc:
            raise RuntimeError(
                "Density facet {0} requires an existing usable FAISS index: {1}".format(
                    facet_label,
                    index_path,
                )
            ) from exc
        for cdr3_length, knn_distance in zip(cdr3_lengths.astype(int), knn_distances):
            rows.append(
                {
                    "facet_label": facet_label,
                    "cdr3_length": int(cdr3_length),
                    "knn_distance": float(knn_distance),
                    "k": int(k),
                    "n_total_facet": int(n_total),
                    "n_sampled_facet": int(len(cdr3_lengths)),
                }
            )
        del cdr3_lengths, knn_distances
        gc.collect()

    def yfv_cdr3_lengths_from_metadata(
        *,
        donor_id: str,
        sample_label: str,
        embedding_path: Path,
        airr_path: Path,
    ) -> pd.Series:
        embedding_metadata, embedding_rows = read_parquet_metadata_only(embedding_path)
        airr_frame = read_airr_like_table(airr_path).reset_index(drop=True)
        standardized_airr = standardize_metadata_frame(airr_frame, chain_default="TRB")
        standardized_embedding = standardize_metadata_frame(embedding_metadata, chain_default="TRB")
        if standardized_embedding["cdr3"].fillna("").astype(str).ne("").any():
            return standardized_embedding["cdr3"].fillna("").astype(str).str.len().astype(int)

        airr_candidates = clone_id_candidates(airr_frame, sample_label=sample_label)
        embedding_candidates = clone_id_candidates(embedding_metadata, sample_label=sample_label)
        match_columns, overlap = best_clone_id_key(embedding_candidates, airr_candidates)
        if match_columns is not None and overlap == embedding_rows:
            embedding_key, airr_key = match_columns
            airr_with_key = standardized_airr.copy()
            airr_with_key["_merge_clone_id"] = airr_candidates[airr_key].astype(str)
            embedding_order = pd.DataFrame(
                {
                    "_merge_clone_id": embedding_candidates[embedding_key].astype(str),
                    "_embedding_order": np.arange(embedding_rows, dtype=np.int64),
                }
            )
            merged = embedding_order.merge(airr_with_key[["_merge_clone_id", "cdr3"]], on="_merge_clone_id", how="left", sort=False)
            if merged["cdr3"].isna().any():
                raise ValueError("Failed to align density metadata for donor {0} {1}".format(donor_id, sample_label))
            merged = merged.sort_values("_embedding_order")
            return merged["cdr3"].fillna("").astype(str).str.len().astype(int).reset_index(drop=True)

        if len(airr_frame) == embedding_rows:
            return standardized_airr["cdr3"].fillna("").astype(str).str.len().astype(int).reset_index(drop=True)
        raise ValueError(
            "Cannot align density metadata for donor {0} {1}: embedding rows={2}, AIRR rows={3}, best clone_id overlap={4}.".format(
                donor_id,
                sample_label,
                embedding_rows,
                len(airr_frame),
                overlap,
            )
        )

    dataset_manifest_path = processed_dir / "benchmark_dataset_manifest.tsv"
    if not dataset_manifest_path.exists():
        raise FileNotFoundError("Dataset manifest is missing: {0}".format(dataset_manifest_path))
    dataset_manifest = pd.read_csv(dataset_manifest_path, sep="\t")
    log_step("Density dataset manifest rows={0}: {1}".format(len(dataset_manifest), dataset_manifest_path))
    for row in dataset_manifest.itertuples(index=False):
        dataset_mode = str(row.dataset_mode)
        dataset_label = str(row.dataset)
        sample_embedding_path = _safe_manifest_path(getattr(row, "sample_embedding_path", None))
        background_embedding_path = _safe_manifest_path(getattr(row, "background_embedding_path", None))
        sample_index_path = _safe_manifest_path(getattr(row, "sample_index_path", None))
        background_index_path = _safe_manifest_path(getattr(row, "background_index_path", None))
        sample_airr_path = _safe_manifest_path(getattr(row, "sample_airr_path", None))
        background_airr_path = _safe_manifest_path(getattr(row, "background_airr_path", None))
        required_paths = [
            sample_embedding_path,
            background_embedding_path,
            sample_index_path,
            background_index_path,
            sample_airr_path,
            background_airr_path,
        ]
        if any(path is None or not path.exists() for path in required_paths):
            log_step("Skipping density dataset {0}: required embedding/index/AIRR inputs are unavailable locally".format(dataset_label))
            continue
        if dataset_mode == "vdjdb":
            epitope_label = str(row.epitope)
            process_facet(
                f"VDJdb / {epitope_label} / sample",
                yfv_cdr3_lengths_from_metadata(
                    donor_id=dataset_label,
                    sample_label="sample",
                    embedding_path=sample_embedding_path,
                    airr_path=sample_airr_path,
                ),
                sample_index_path,
            )
            process_facet(
                f"VDJdb / {epitope_label} / background",
                yfv_cdr3_lengths_from_metadata(
                    donor_id=dataset_label,
                    sample_label="background",
                    embedding_path=background_embedding_path,
                    airr_path=background_airr_path,
                ),
                background_index_path,
            )
            continue

        donor_id = str(row.donor_id)
        process_facet(
            f"YFV / {donor_id} / sample",
            yfv_cdr3_lengths_from_metadata(
                donor_id=donor_id,
                sample_label="sample",
                embedding_path=sample_embedding_path,
                airr_path=sample_airr_path,
            ),
            sample_index_path,
        )
        process_facet(
            f"YFV / {donor_id} / background",
            yfv_cdr3_lengths_from_metadata(
                donor_id=donor_id,
                sample_label="background",
                embedding_path=background_embedding_path,
                airr_path=background_airr_path,
            ),
            background_index_path,
        )
    density_df = pd.DataFrame(rows)
    log_step("Computed density table rows={0}".format(len(density_df)))
    return density_df


def run_density_analysis(
    processed_dir: str | Path = "data/processed",
    *,
    density_path: str | Path = "results/density/knn_distance_by_length.tsv",
    figure_stem: str | Path = "figures/clustering_strategy/fig1_density_by_length",
    k: int = 5,
    max_points_per_facet: int | None = None,
) -> pd.DataFrame:
    ensure_output_dirs()
    log_step("Starting density analysis")
    density_df = compute_density_by_length(
        processed_dir,
        k=k,
        max_points_per_facet=max_points_per_facet,
    )
    density_path = repo_path(density_path)
    density_path.parent.mkdir(parents=True, exist_ok=True)
    density_df.to_csv(density_path, sep="\t", index=False)
    log_step("Wrote density table rows={0}: {1}".format(len(density_df), density_path))
    if len(density_df):
        plot_density_by_length(density_df, repo_path(figure_stem))
        log_step("Wrote density figures: {0}.png/.pdf".format(repo_path(figure_stem)))
    log_step("Finished density analysis")
    return density_df


def run_vdjdb_evaluation(
    *,
    assignments_dir: str | Path = "results/clustering_assignments",
    metadata_parts_dir: str | Path = "results/run_metadata/clustering_run_parts",
    metadata_path: str | Path | None = None,
    redcea_runs_dir: str | Path = "results/redcea_runs",
    metrics_path: str | Path = "results/metrics/vdjdb_clustering_metrics.tsv",
    fig2_stem: str | Path = "figures/clustering_strategy/fig2_vdjdb_f1_precision_recall",
    fig3_stem: str | Path = "figures/clustering_strategy/fig3_vdjdb_cluster_concentration",
    tcrvdb_path: str | Path = DEFAULT_TCRVDB_PATH,
    padj_threshold: float = DEFAULT_TCRVDB_PADJ_THRESHOLD,
) -> pd.DataFrame:
    log_step("Starting VDJdb evaluation")
    metrics_df = _compute_vdjdb_metrics_from_redcea_runs(
        assignments_dir=assignments_dir,
        redcea_runs_dir=redcea_runs_dir,
        tcrvdb_path=tcrvdb_path,
        padj_threshold=padj_threshold,
    )
    if metrics_df.empty and metadata_path is not None and not repo_path(redcea_runs_dir).exists():
        log_step(
            "Falling back to metadata-based VDJdb loading because redcea_runs is unavailable: {0}".format(
                repo_path(redcea_runs_dir)
            )
        )
        assignments, run_metadata = load_assignment_tables(
            assignments_dir=assignments_dir,
            metadata_parts_dir=metadata_parts_dir,
            metadata_path=metadata_path,
            only_success=True,
            dataset_mode="vdjdb",
            consolidate_metadata=True,
        )
        if assignments.empty or "dataset_mode" not in assignments.columns:
            metrics_df = pd.DataFrame()
        else:
            vdjdb_assignments = assignments.loc[assignments["dataset_mode"] == "vdjdb"].copy()
            metrics_df = compute_vdjdb_metrics(vdjdb_assignments, run_metadata)
    elif metrics_df.empty and metadata_path is None and not repo_path(redcea_runs_dir).exists():
        log_step("Skipping metadata fallback because metadata_path was not provided and redcea_runs is unavailable.")
    if metrics_df.empty:
        metrics_df = pd.DataFrame()
        metrics_path = repo_path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_path, sep="\t", index=False)
        log_step("No assignments available for VDJdb evaluation; wrote empty metrics: {0}".format(metrics_path))
        return metrics_df
    metrics_path = repo_path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, sep="\t", index=False)
    log_step("Wrote VDJdb metrics rows={0}: {1}".format(len(metrics_df), metrics_path))
    if len(metrics_df):
        plot_vdjdb_benchmark(metrics_df, repo_path(fig2_stem))
        log_step("Wrote VDJdb benchmark figures: {0}.png/.pdf".format(repo_path(fig2_stem)))
        plot_vdjdb_signal_concentration(metrics_df, repo_path(fig3_stem))
        log_step("Wrote VDJdb concentration figures: {0}.png/.pdf".format(repo_path(fig3_stem)))
    log_step("Finished VDJdb evaluation")
    return metrics_df


def run_yfv_enrichment_evaluation(
    *,
    assignments_dir: str | Path = "results/clustering_assignments",
    metadata_parts_dir: str | Path = "results/run_metadata/clustering_run_parts",
    metadata_path: str | Path = "results/run_metadata/clustering_runs.tsv",
    enrichment_path: str | Path = "results/enrichment/yfv_cluster_enrichment.tsv",
    metrics_path: str | Path = "results/metrics/yfv_enrichment_metrics.tsv",
    fig4_stem: str | Path = "figures/clustering_strategy/fig4_yfv_enrichment_output",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log_step("Starting YFV enrichment evaluation")
    assignments, _run_metadata = load_assignment_tables(
        assignments_dir=assignments_dir,
        metadata_parts_dir=metadata_parts_dir,
        metadata_path=metadata_path,
        only_success=True,
        dataset_mode="yfv",
    )
    if assignments.empty or "dataset_mode" not in assignments.columns:
        enrichment_df = pd.DataFrame()
        summary_df = pd.DataFrame()
        enrichment_path = repo_path(enrichment_path)
        metrics_path = repo_path(metrics_path)
        enrichment_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        enrichment_df.to_csv(enrichment_path, sep="\t", index=False)
        summary_df.to_csv(metrics_path, sep="\t", index=False)
        log_step(
            "No assignments available for YFV enrichment; wrote empty outputs: {0}, {1}".format(
                enrichment_path,
                metrics_path,
            )
        )
        return enrichment_df, summary_df
    yfv_assignments = assignments.loc[assignments["dataset_mode"] == "yfv"].copy()
    log_step(
        "YFV assignments rows={0}, runs={1}, donors={2}".format(
            len(yfv_assignments),
            yfv_assignments["run_id"].nunique() if "run_id" in yfv_assignments.columns else 0,
            yfv_assignments["donor_id"].nunique() if "donor_id" in yfv_assignments.columns else 0,
        )
    )
    enrichment_df = compute_yfv_cluster_enrichment(yfv_assignments)
    enrichment_path = repo_path(enrichment_path)
    enrichment_path.parent.mkdir(parents=True, exist_ok=True)
    enrichment_df.to_csv(enrichment_path, sep="\t", index=False)
    log_step("Wrote YFV cluster enrichment rows={0}: {1}".format(len(enrichment_df), enrichment_path))
    summary_df = summarize_yfv_enrichment(enrichment_df)
    metrics_path = repo_path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(metrics_path, sep="\t", index=False)
    log_step("Wrote YFV enrichment summary rows={0}: {1}".format(len(summary_df), metrics_path))
    if len(summary_df):
        plot_yfv_enrichment_summary(summary_df, repo_path(fig4_stem))
        log_step("Wrote YFV enrichment figures: {0}.png/.pdf".format(repo_path(fig4_stem)))
    log_step("Finished YFV enrichment evaluation")
    return enrichment_df, summary_df


def run_yfv_known_clonotype_evaluation(
    *,
    assignments_dir: str | Path = "results/clustering_assignments",
    metadata_parts_dir: str | Path = "results/run_metadata/clustering_run_parts",
    metadata_path: str | Path = "results/run_metadata/clustering_runs.tsv",
    known_yfv_path: str | Path = "data/processed/known_yfv_vdjdb_clonotypes.tsv",
    enrichment_path: str | Path = "results/enrichment/yfv_cluster_enrichment.tsv",
    output_path: str | Path = "results/metrics/yfv_known_clonotype_recovery.tsv",
    fig5_stem: str | Path = "figures/clustering_strategy/fig5_yfv_known_clonotype_recovery",
) -> pd.DataFrame:
    log_step("Starting YFV known clonotype recovery evaluation")
    assignments, _run_metadata = load_assignment_tables(
        assignments_dir=assignments_dir,
        metadata_parts_dir=metadata_parts_dir,
        metadata_path=metadata_path,
        only_success=True,
        dataset_mode="yfv",
    )
    if assignments.empty or "dataset_mode" not in assignments.columns:
        recovery_df = pd.DataFrame()
        output_path = repo_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        recovery_df.to_csv(output_path, sep="\t", index=False)
        log_step("No assignments available for YFV known recovery; wrote empty output: {0}".format(output_path))
        return recovery_df
    enrichment_df = pd.read_csv(repo_path(enrichment_path), sep="\t")
    known_yfv = pd.read_csv(repo_path(known_yfv_path), sep="\t")
    log_step(
        "Loaded YFV recovery inputs: enrichment_rows={0}, known_clonotypes={1}".format(
            len(enrichment_df),
            len(known_yfv),
        )
    )
    yfv_assignments = assignments.loc[assignments["dataset_mode"] == "yfv"].copy()
    log_step("YFV assignments for known recovery rows={0}".format(len(yfv_assignments)))
    recovery_df = compute_known_yfv_recovery(yfv_assignments, enrichment_df, known_yfv)
    if len(recovery_df):
        recovery_df = recovery_df.copy()
        recovery_df["_candidate_numeric"] = recovery_df["is_candidate_clustered"].astype(float)
        recovery_df["_significant_numeric"] = recovery_df["is_in_significant_cluster"].astype(float)
        grouped = recovery_df.groupby(["run_id", "donor_id", "method"])
        recovery_df["candidate_level_recovery_rate"] = grouped["_candidate_numeric"].transform("mean").astype(float)
        recovery_df["enrichment_level_recovery_rate"] = grouped["_significant_numeric"].transform("mean").astype(float)
        recovery_df = recovery_df.drop(columns=["_candidate_numeric", "_significant_numeric"])
    else:
        recovery_df["candidate_level_recovery_rate"] = []
        recovery_df["enrichment_level_recovery_rate"] = []
    output_path = repo_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_df.to_csv(output_path, sep="\t", index=False)
    log_step("Wrote YFV known clonotype recovery rows={0}: {1}".format(len(recovery_df), output_path))
    if len(recovery_df):
        plot_yfv_known_recovery_heatmap(recovery_df, repo_path(fig5_stem))
        log_step("Wrote YFV known recovery figures: {0}.png/.pdf".format(repo_path(fig5_stem)))
    log_step("Finished YFV known clonotype recovery evaluation")
    return recovery_df


def run_cross_donor_overlap_evaluation(
    *,
    assignments_dir: str | Path = "results/clustering_assignments",
    metadata_parts_dir: str | Path = "results/run_metadata/clustering_run_parts",
    metadata_path: str | Path = "results/run_metadata/clustering_runs.tsv",
    enrichment_path: str | Path = "results/enrichment/yfv_cluster_enrichment.tsv",
    output_path: str | Path = "results/metrics/yfv_cross_donor_overlap.tsv",
    fig6_stem: str | Path = "figures/clustering_strategy/fig6_yfv_cross_donor_overlap",
) -> pd.DataFrame:
    log_step("Starting YFV cross-donor overlap evaluation")
    assignments, _run_metadata = load_assignment_tables(
        assignments_dir=assignments_dir,
        metadata_parts_dir=metadata_parts_dir,
        metadata_path=metadata_path,
        only_success=True,
        dataset_mode="yfv",
    )
    if assignments.empty or "dataset_mode" not in assignments.columns:
        overlap_df = pd.DataFrame()
        output_path = repo_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlap_df.to_csv(output_path, sep="\t", index=False)
        log_step("No assignments available for cross-donor overlap; wrote empty output: {0}".format(output_path))
        return overlap_df
    enrichment_df = pd.read_csv(repo_path(enrichment_path), sep="\t")
    log_step("Loaded YFV enrichment rows for overlap={0}".format(len(enrichment_df)))
    yfv_assignments = assignments.loc[assignments["dataset_mode"] == "yfv"].copy()
    log_step("YFV assignments for overlap rows={0}".format(len(yfv_assignments)))
    overlap_df = compute_cross_donor_overlap(yfv_assignments, enrichment_df)
    output_path = repo_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlap_df.to_csv(output_path, sep="\t", index=False)
    log_step("Wrote YFV cross-donor overlap rows={0}: {1}".format(len(overlap_df), output_path))
    if len(overlap_df):
        plot_cross_donor_overlap(overlap_df, repo_path(fig6_stem))
        log_step("Wrote YFV cross-donor overlap figures: {0}.png/.pdf".format(repo_path(fig6_stem)))
    log_step("Finished YFV cross-donor overlap evaluation")
    return overlap_df


def build_final_method_comparison(
    *,
    vdjdb_metrics_path: str | Path = "results/metrics/vdjdb_clustering_metrics.tsv",
    yfv_metrics_path: str | Path = "results/metrics/yfv_enrichment_metrics.tsv",
    yfv_recovery_path: str | Path = "results/metrics/yfv_known_clonotype_recovery.tsv",
    overlap_path: str | Path = "results/metrics/yfv_cross_donor_overlap.tsv",
) -> pd.DataFrame:
    log_step("Building final method comparison")
    def _safe_read(path: str | Path) -> pd.DataFrame:
        path = repo_path(path)
        if not path.exists() or path.stat().st_size == 0:
            log_step("Metric input missing or empty: {0}".format(path))
            return pd.DataFrame()
        frame = pd.read_csv(path, sep="\t")
        log_step("Loaded metric input rows={0}: {1}".format(len(frame), path))
        return frame

    vdjdb_metrics = _safe_read(vdjdb_metrics_path)
    yfv_metrics = _safe_read(yfv_metrics_path)
    yfv_recovery = _safe_read(yfv_recovery_path)
    overlap_df = _safe_read(overlap_path)

    if vdjdb_metrics.empty and yfv_metrics.empty and yfv_recovery.empty and overlap_df.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "parameter_json",
                "vdjdb_mean_f1",
                "vdjdb_mean_precision",
                "vdjdb_mean_recall",
                "vdjdb_mean_purity",
                "yfv_mean_significant_clusters",
                "yfv_mean_retained_fraction",
                "yfv_mean_background_contamination",
                "yfv_mean_weighted_enrichment_score",
                "yfv_mean_candidate_recovery_rate",
                "yfv_mean_enrichment_recovery_rate",
                "yfv_mean_overlap_fold_change",
            ]
        )

    if len(vdjdb_metrics):
        vdjdb_summary = (
            vdjdb_metrics.groupby(["method", "parameter_json"])
            .agg(
                vdjdb_mean_f1=("f1", "mean"),
                vdjdb_mean_precision=("precision", "mean"),
                vdjdb_mean_recall=("recall", "mean"),
                vdjdb_mean_purity=("weighted_cluster_purity", "mean"),
            )
            .reset_index()
        )
    else:
        vdjdb_summary = pd.DataFrame(
            columns=["method", "parameter_json", "vdjdb_mean_f1", "vdjdb_mean_precision", "vdjdb_mean_recall", "vdjdb_mean_purity"]
        )
    if len(yfv_metrics):
        yfv_summary = (
            yfv_metrics.groupby(["method", "parameter_json"])
            .agg(
                yfv_mean_significant_clusters=("number_of_significant_enriched_clusters", "mean"),
                yfv_mean_retained_fraction=("retained_fraction", "mean"),
                yfv_mean_background_contamination=("background_contamination", "mean"),
                yfv_mean_weighted_enrichment_score=("weighted_enrichment_score", "mean"),
            )
            .reset_index()
        )
    else:
        yfv_summary = pd.DataFrame(
            columns=[
                "method",
                "parameter_json",
                "yfv_mean_significant_clusters",
                "yfv_mean_retained_fraction",
                "yfv_mean_background_contamination",
                "yfv_mean_weighted_enrichment_score",
            ]
        )
    if len(yfv_recovery):
        yfv_recovery = yfv_recovery.copy()
        yfv_recovery["candidate_level_recovery_rate"] = pd.to_numeric(
            yfv_recovery["candidate_level_recovery_rate"], errors="coerce"
        )
        yfv_recovery["enrichment_level_recovery_rate"] = pd.to_numeric(
            yfv_recovery["enrichment_level_recovery_rate"], errors="coerce"
        )
        recovery_summary = (
            yfv_recovery.groupby(["method", "parameter_json"])
            .agg(
                yfv_mean_candidate_recovery_rate=("candidate_level_recovery_rate", "mean"),
                yfv_mean_enrichment_recovery_rate=("enrichment_level_recovery_rate", "mean"),
            )
            .reset_index()
        )
    else:
        recovery_summary = pd.DataFrame(columns=["method", "parameter_json", "yfv_mean_candidate_recovery_rate", "yfv_mean_enrichment_recovery_rate"])
    if len(overlap_df):
        overlap_summary = (
            overlap_df.groupby(["method", "parameter_json"])
            .agg(
                yfv_mean_overlap_fold_change=("overlap_fold_change", "mean"),
            )
            .reset_index()
        )
    else:
        overlap_summary = pd.DataFrame(columns=["method", "parameter_json", "yfv_mean_overlap_fold_change"])

    final = vdjdb_summary.merge(yfv_summary, on=["method", "parameter_json"], how="outer")
    final = final.merge(recovery_summary, on=["method", "parameter_json"], how="left")
    final = final.merge(overlap_summary, on=["method", "parameter_json"], how="left")
    sort_columns = [column for column in ["vdjdb_mean_f1", "yfv_mean_enrichment_recovery_rate", "yfv_mean_weighted_enrichment_score"] if column in final.columns]
    if sort_columns:
        final = final.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    final = final.reset_index(drop=True)
    log_step("Built final method comparison rows={0}".format(len(final)))
    return final


def write_summary_report(
    comparison_df: pd.DataFrame,
    *,
    output_path: str | Path = "reports/clustering_strategy_summary.md",
) -> Path:
    output_path = repo_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Clustering Strategy Summary",
        "",
    ]
    if comparison_df.empty:
        lines.extend(["No successful benchmark runs were available.", ""])
    else:
        best_vdjdb = comparison_df.sort_values("vdjdb_mean_f1", ascending=False).iloc[0]
        best_yfv = comparison_df.sort_values(
            ["yfv_mean_enrichment_recovery_rate", "yfv_mean_weighted_enrichment_score"],
            ascending=[False, False],
        ).iloc[0]
        lines.extend(
            [
                f"- Best VDJdb method: `{best_vdjdb['method']}`",
                f"- Best YFV method: `{best_yfv['method']}`",
                "",
                "## Combined Table",
                "",
                "```text",
                comparison_df.to_string(index=False),
                "```",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log_step("Wrote summary report: {0}".format(output_path))
    return output_path


def run_summary_and_method_selection(
    *,
    vdjdb_metrics_path: str | Path = "results/metrics/vdjdb_clustering_metrics.tsv",
    yfv_metrics_path: str | Path = "results/metrics/yfv_enrichment_metrics.tsv",
    yfv_recovery_path: str | Path = "results/metrics/yfv_known_clonotype_recovery.tsv",
    overlap_path: str | Path = "results/metrics/yfv_cross_donor_overlap.tsv",
    output_table_path: str | Path = "results/metrics/final_method_comparison.tsv",
    figure_stem: str | Path = "figures/clustering_strategy/final_method_summary",
    report_path: str | Path = "reports/clustering_strategy_summary.md",
) -> pd.DataFrame:
    ensure_output_dirs()
    log_step("Starting final summary and method selection")
    comparison_df = build_final_method_comparison(
        vdjdb_metrics_path=vdjdb_metrics_path,
        yfv_metrics_path=yfv_metrics_path,
        yfv_recovery_path=yfv_recovery_path,
        overlap_path=overlap_path,
    )
    output_table_path = repo_path(output_table_path)
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_table_path, sep="\t", index=False)
    log_step("Wrote final method comparison rows={0}: {1}".format(len(comparison_df), output_table_path))
    if len(comparison_df):
        plot_final_method_summary(comparison_df, repo_path(figure_stem))
        log_step("Wrote final method summary figures: {0}.png/.pdf".format(repo_path(figure_stem)))
    write_summary_report(comparison_df, output_path=report_path)
    log_step("Finished final summary and method selection")
    return comparison_df
