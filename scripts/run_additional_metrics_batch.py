from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from redcea.auxiliary_cluster_metrics import (
    AUXILIARY_CLUSTER_METRIC_COLUMNS,
    append_auxiliary_cluster_metrics,
)


SUMMARY_SUFFIX = "_summary_tcrempnet.tsv"
CLUSTERS_SUFFIX = "_tcremp_clusters.tsv"


def setup_logging(log_path: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill additional RedCEA metrics for all run directories and build unified metric tables."
    )
    parser.add_argument("--runs-root", default="results/redcea_runs")
    parser.add_argument(
        "--execution-manifest",
        default=r"C:\Users\lizzka239\Downloads\clustering_execution_manifest.tsv",
        help="TSV manifest with run_id and parameter_json metadata.",
    )
    parser.add_argument("--cluster-output", default="results/metrics/redcea_all_cluster_metrics.tsv")
    parser.add_argument("--run-output", default="results/metrics/redcea_all_run_metrics.tsv")
    parser.add_argument("--log-path", default="results/metrics/run_additional_metrics_batch.log")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument(
        "--dataset-mode",
        choices=["all", "vdjdb", "yfv"],
        default="all",
        help="Restrict processing and output tables to one dataset mode.",
    )
    parser.add_argument(
        "--run-id-prefix",
        default=None,
        help="Optional run_id prefix filter applied after dataset-mode filtering.",
    )
    parser.add_argument(
        "--rewrite-existing",
        action="store_true",
        help="Recompute and rewrite summary files even if additional metric columns are already present.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed worker result instead of continuing.",
    )
    return parser.parse_args()


def find_run_prefix(run_dir: Path) -> str | None:
    matches = sorted(run_dir.glob(f"*{SUMMARY_SUFFIX}"))
    if not matches:
        return None
    return matches[0].name[: -len(SUMMARY_SUFFIX)]


def load_first_npy(run_dir: Path, pattern: str) -> np.ndarray | None:
    match = next(run_dir.glob(pattern), None)
    if match is None:
        return None
    return np.load(match)


def infer_source_counts(cluster_df: pd.DataFrame) -> tuple[int, int]:
    if "source" in cluster_df.columns:
        source = cluster_df["source"].fillna("").astype(str)
        total_sample = int(source.eq("sample").sum())
        total_background = int(source.eq("background").sum())
        return total_sample, total_background

    clone_ids = cluster_df["clone_id"].fillna("").astype(str)
    total_sample = int(clone_ids.str.startswith("s_").sum())
    total_background = int(clone_ids.str.startswith("b_").sum())
    return total_sample, total_background


def needs_metric_backfill(summary_df: pd.DataFrame) -> bool:
    return any(column not in summary_df.columns for column in AUXILIARY_CLUSTER_METRIC_COLUMNS)


def atomic_write_tsv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temp_path, sep="\t", index=False)
    os.replace(temp_path, destination)


def process_single_run(run_dir_str: str, rewrite_existing: bool) -> dict[str, object]:
    run_dir = Path(run_dir_str)
    started = time.perf_counter()
    result: dict[str, object] = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "status": "unknown",
        "message": "",
        "runtime_seconds": np.nan,
        "cluster_rows": 0,
        "summary_rows": 0,
        "summary_path": "",
        "metrics_added": False,
        "traceback": "",
    }
    try:
        prefix = find_run_prefix(run_dir)
        if prefix is None:
            result["status"] = "skipped"
            result["message"] = "summary file not found"
            return result

        summary_path = run_dir / f"{prefix}{SUMMARY_SUFFIX}"
        clusters_path = run_dir / f"{prefix}{CLUSTERS_SUFFIX}"
        result["summary_path"] = str(summary_path)
        if not clusters_path.exists():
            result["status"] = "skipped"
            result["message"] = "clusters file not found"
            return result

        summary_df = pd.read_csv(summary_path, sep="\t")
        result["summary_rows"] = int(len(summary_df))
        if "enrichment_fdr_zbinom" not in summary_df.columns:
            result["status"] = "skipped"
            result["message"] = "not a zbinom summary"
            return result

        if (not rewrite_existing) and (not needs_metric_backfill(summary_df)):
            result["status"] = "ok"
            result["message"] = "metrics already present"
            result["metrics_added"] = False
            return result

        cluster_df = pd.read_csv(clusters_path, sep="\t")
        result["cluster_rows"] = int(len(cluster_df))
        total_sample, total_background = infer_source_counts(cluster_df)
        sample_knn_indices = load_first_npy(run_dir, "knn_sample_sample__*.indices.npy")
        sample_knn_distances = load_first_npy(run_dir, "knn_sample_sample__*.distances.npy")

        extended_df = append_auxiliary_cluster_metrics(
            summary_df=summary_df,
            cluster_df=cluster_df,
            total_sample=total_sample,
            total_background=total_background,
            sample_knn_indices=sample_knn_indices,
            sample_knn_distances=sample_knn_distances,
        )
        atomic_write_tsv(extended_df, summary_path)
        result["status"] = "ok"
        result["message"] = "summary updated"
        result["metrics_added"] = True
        result["summary_rows"] = int(len(extended_df))
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["message"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        return result
    finally:
        result["runtime_seconds"] = float(time.perf_counter() - started)


def parse_parameter_json(value: object) -> dict[str, object]:
    if pd.isna(value):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_execution_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        logging.warning("Execution manifest not found: %s", manifest_path)
        return pd.DataFrame(columns=["run_id"])
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    if "run_id" not in manifest_df.columns:
        raise ValueError("Execution manifest must contain a run_id column: {0}".format(manifest_path))
    return manifest_df


def infer_run_metadata_from_name(run_id: str) -> dict[str, object]:
    parts = run_id.split("_")
    metadata: dict[str, object] = {
        "dataset": None,
        "dataset_mode": None,
        "method": None,
        "parameter_json": None,
        "redcea_output_dir": None,
    }
    if len(parts) < 3:
        return metadata

    grid_idx = None
    for idx, token in enumerate(parts):
        if token == "grid":
            grid_idx = idx
            break
    if grid_idx is None or grid_idx < 1:
        return metadata

    method = parts[grid_idx - 1]
    dataset_tokens = parts[: grid_idx - 1]
    if not dataset_tokens:
        return metadata

    if dataset_tokens[0] == "yfv":
        metadata["dataset_mode"] = "yfv"
        metadata["dataset"] = "_".join(dataset_tokens)
    elif dataset_tokens[0] == "vdjdb":
        metadata["dataset_mode"] = "vdjdb"
        metadata["dataset"] = "_".join(dataset_tokens)
    else:
        metadata["dataset"] = "_".join(dataset_tokens)

    metadata["method"] = method
    return metadata


def run_matches_filters(
    *,
    run_id: str,
    manifest_lookup: pd.DataFrame,
    dataset_mode: str = "all",
    run_id_prefix: str | None = None,
) -> bool:
    if run_id_prefix and not str(run_id).startswith(str(run_id_prefix)):
        return False
    if dataset_mode == "all":
        return True

    inferred = infer_run_metadata_from_name(run_id)
    inferred_mode = inferred.get("dataset_mode")
    if inferred_mode == dataset_mode:
        return True
    if manifest_lookup.empty or "run_id" not in manifest_lookup.columns:
        return False

    matched = manifest_lookup.loc[manifest_lookup["run_id"].astype(str) == str(run_id)]
    if matched.empty or "dataset_mode" not in matched.columns:
        return False
    dataset_mode_values = matched["dataset_mode"].dropna().astype(str).unique().tolist()
    return dataset_mode in dataset_mode_values


def select_run_dirs(
    *,
    runs_root: Path,
    manifest_lookup: pd.DataFrame,
    dataset_mode: str = "all",
    run_id_prefix: str | None = None,
) -> list[Path]:
    return [
        run_dir
        for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir())
        if run_matches_filters(
            run_id=run_dir.name,
            manifest_lookup=manifest_lookup,
            dataset_mode=dataset_mode,
            run_id_prefix=run_id_prefix,
        )
    ]


def resolve_run_metadata(
    *,
    run_id: str,
    run_dir: Path,
    manifest_lookup: pd.DataFrame,
) -> dict[str, object]:
    inferred = infer_run_metadata_from_name(run_id)
    inferred["run_id"] = run_id
    inferred["run_dir"] = str(run_dir)
    inferred["redcea_output_dir"] = str(run_dir)

    if manifest_lookup.empty:
        return inferred

    matched = manifest_lookup.loc[manifest_lookup["run_id"].astype(str) == run_id]
    if matched.empty:
        return inferred

    row = matched.iloc[0]
    resolved = dict(inferred)
    for column in matched.columns:
        value = row[column]
        if pd.isna(value):
            continue
        resolved[column] = value

    if not resolved.get("dataset_mode"):
        resolved["dataset_mode"] = inferred.get("dataset_mode")
    if not resolved.get("dataset"):
        resolved["dataset"] = inferred.get("dataset")
    if not resolved.get("method"):
        resolved["method"] = inferred.get("method")
    resolved["redcea_output_dir"] = str(run_dir)
    return resolved


def collect_cluster_level_table(
    *,
    run_dirs: list[Path],
    manifest_lookup: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for run_dir in run_dirs:
        prefix = find_run_prefix(run_dir)
        if prefix is None:
            continue
        summary_path = run_dir / f"{prefix}{SUMMARY_SUFFIX}"
        if not summary_path.exists():
            continue
        summary_df = pd.read_csv(summary_path, sep="\t")
        summary_df.insert(0, "run_id", run_dir.name)
        summary_df.insert(1, "run_dir", str(run_dir))
        resolved = resolve_run_metadata(run_id=run_dir.name, run_dir=run_dir, manifest_lookup=manifest_lookup)
        summary_df.insert(2, "dataset", resolved.get("dataset"))
        summary_df.insert(3, "dataset_mode", resolved.get("dataset_mode"))
        summary_df.insert(4, "method", resolved.get("method"))
        summary_df.insert(5, "parameter_json", resolved.get("parameter_json"))
        summary_df.insert(6, "redcea_output_dir", resolved.get("redcea_output_dir"))

        extra_columns = [
            column
            for column in resolved.keys()
            if column not in {"run_id", "run_dir", "dataset", "dataset_mode", "method", "parameter_json", "redcea_output_dir"}
            and column not in summary_df.columns
        ]
        for column in extra_columns:
            summary_df[column] = resolved[column]
        frames.append(summary_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    parameter_df = pd.json_normalize(combined["parameter_json"].apply(parse_parameter_json)).add_prefix("param_")
    if not parameter_df.empty:
        combined = pd.concat([combined, parameter_df], axis=1)
    return combined


def numeric_summary(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "max": np.nan,
            "q25": np.nan,
            "q75": np.nan,
        }
    return {
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "max": float(numeric.max()),
        "q25": float(numeric.quantile(0.25)),
        "q75": float(numeric.quantile(0.75)),
    }


def _prepare_cluster_metric_inputs(cluster_df: pd.DataFrame) -> pd.DataFrame:
    prepared = cluster_df.copy()
    numeric_columns = [
        "cluster_id",
        "cluster_size",
        "sample",
        "background",
        "sample_fraction_in_cluster",
        "log_fold_change",
        "enrichment_fdr_zbinom",
    ]
    for column in numeric_columns:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    if "cluster_size" not in prepared.columns:
        prepared["cluster_size"] = np.nan
    if {"sample", "background"}.issubset(prepared.columns):
        computed_cluster_size = prepared["sample"] + prepared["background"]
        prepared["cluster_size"] = prepared["cluster_size"].where(prepared["cluster_size"].notna(), computed_cluster_size)

    if "sample_fraction_in_cluster" not in prepared.columns:
        prepared["sample_fraction_in_cluster"] = np.nan
    if "sample" in prepared.columns:
        computed_fraction = pd.Series(np.nan, index=prepared.index, dtype="float64")
        valid_cluster_size = prepared["cluster_size"] > 0
        computed_fraction.loc[valid_cluster_size] = (
            prepared.loc[valid_cluster_size, "sample"] / prepared.loc[valid_cluster_size, "cluster_size"]
        )
        prepared["sample_fraction_in_cluster"] = prepared["sample_fraction_in_cluster"].where(
            prepared["sample_fraction_in_cluster"].notna(),
            computed_fraction,
        )
    return prepared


def _enriched_cluster_mask(frame: pd.DataFrame) -> pd.Series:
    if not {"log_fold_change", "enrichment_fdr_zbinom"}.issubset(frame.columns):
        return pd.Series(False, index=frame.index, dtype=bool)
    return (
        (pd.to_numeric(frame["log_fold_change"], errors="coerce") > 0)
        & (pd.to_numeric(frame["enrichment_fdr_zbinom"], errors="coerce") < 0.05)
        & (pd.to_numeric(frame["cluster_id"], errors="coerce") != -1)
    )


def _non_noise_cluster_mask(frame: pd.DataFrame) -> pd.Series:
    if "cluster_id" not in frame.columns:
        return pd.Series(True, index=frame.index, dtype=bool)
    return pd.to_numeric(frame["cluster_id"], errors="coerce") != -1


def _mean_median(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return np.nan, np.nan
    return float(numeric.mean()), float(numeric.median())


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan
    if denominator == 0:
        if numerator > 0:
            return float(np.inf)
        if numerator == 0:
            return np.nan
    return float(numerator / denominator)


def _resolve_dense_group_cols(cluster_df: pd.DataFrame, group_cols: list[str] | tuple[str, ...] | str | None) -> list[str]:
    if group_cols is not None:
        resolved = [group_cols] if isinstance(group_cols, str) else list(group_cols)
        missing = [column for column in resolved if column not in cluster_df.columns]
        if missing:
            raise ValueError("Dense score grouping columns are missing from cluster_df: {0}".format(", ".join(missing)))
        return resolved

    for candidate in (["sample_group"], ["epitope"], ["dataset"]):
        if all(column in cluster_df.columns for column in candidate):
            return candidate

    logging.warning("No sample_group/epitope/dataset columns found; computing dense RedCEA ranks across all runs together.")
    return []


def _iter_comparison_groups(frame: pd.DataFrame, group_cols: list[str]) -> list[tuple[object, pd.DataFrame]]:
    if not group_cols:
        return [(None, frame)]
    return list(frame.groupby(group_cols, sort=True, dropna=False))


def compute_dense_redcea_scores(
    cluster_df: pd.DataFrame,
    group_cols: list[str] | tuple[str, ...] | str | None = None,
) -> pd.DataFrame:
    if cluster_df.empty or "run_id" not in cluster_df.columns:
        return pd.DataFrame()

    prepared = _prepare_cluster_metric_inputs(cluster_df)
    resolved_group_cols = _resolve_dense_group_cols(prepared, group_cols)

    rows: list[dict[str, object]] = []
    for run_id, frame in prepared.groupby("run_id", sort=True):
        row: dict[str, object] = {"run_id": run_id}
        for column in resolved_group_cols:
            row[column] = frame.iloc[0][column]

        enriched_mask = _enriched_cluster_mask(frame)
        sample_all = pd.to_numeric(frame.get("sample", pd.Series(dtype=float)), errors="coerce")
        sample_enriched = pd.to_numeric(frame.loc[enriched_mask, "sample"], errors="coerce")
        cluster_size_enriched = pd.to_numeric(frame.loc[enriched_mask, "cluster_size"], errors="coerce")
        lfc_enriched = pd.to_numeric(frame.loc[enriched_mask, "log_fold_change"], errors="coerce")

        total_sample = float(sample_all.sum()) if "sample" in frame.columns else np.nan
        sample_mass_enriched = float(sample_enriched.sum()) if len(sample_enriched) else 0.0
        cluster_mass_enriched = float(cluster_size_enriched.sum()) if len(cluster_size_enriched) else 0.0
        background_mass_enriched = (
            float(cluster_mass_enriched - sample_mass_enriched)
            if pd.notna(cluster_mass_enriched) and pd.notna(sample_mass_enriched)
            else np.nan
        )
        sample_coverage_enriched = (
            float(sample_mass_enriched / total_sample) if pd.notna(total_sample) and total_sample > 0 else np.nan
        )

        effective_n = np.nan
        effective_sample_cluster_size = np.nan
        if sample_mass_enriched > 0:
            cluster_weights = sample_enriched.fillna(0.0)
            weight_fraction = cluster_weights / sample_mass_enriched
            simpson_denom = float((weight_fraction**2).sum())
            if simpson_denom > 0:
                effective_n = float(1.0 / simpson_denom)
                effective_sample_cluster_size = float(sample_mass_enriched / effective_n)

        lfc_sample_weighted = np.nan
        valid_lfc_mask = sample_enriched.notna() & lfc_enriched.notna()
        if valid_lfc_mask.any():
            lfc_weights = sample_enriched.loc[valid_lfc_mask]
            weight_total = float(lfc_weights.sum())
            if weight_total > 0:
                lfc_sample_weighted = float((lfc_enriched.loc[valid_lfc_mask] * lfc_weights).sum() / weight_total)

        row.update(
            {
                "sample_mass_enriched": sample_mass_enriched,
                "cluster_mass_enriched": cluster_mass_enriched,
                "background_mass_enriched": background_mass_enriched,
                "sample_coverage_enriched": sample_coverage_enriched,
                "effective_n_enriched_clusters_sample_weighted": effective_n,
                "effective_sample_cluster_size": effective_sample_cluster_size,
                "lfc_sample_weighted": lfc_sample_weighted,
                "_total_sample": total_sample,
                "_coverage_for_rank": float(np.log1p(sample_mass_enriched)) if pd.notna(sample_mass_enriched) else np.nan,
            }
        )
        rows.append(row)

    run_level = pd.DataFrame(rows)
    if run_level.empty:
        return run_level

    run_level["coverage_rank_pct"] = np.nan
    run_level["lfc_rank_pct"] = np.nan
    run_level["eff_low_threshold"] = np.nan
    run_level["eff_high_threshold"] = np.nan
    run_level["effective_size_penalty"] = np.nan

    for _, group_frame in _iter_comparison_groups(run_level, resolved_group_cols):
        group_index = group_frame.index
        run_level.loc[group_index, "coverage_rank_pct"] = group_frame["_coverage_for_rank"].rank(method="average", pct=True)
        run_level.loc[group_index, "lfc_rank_pct"] = group_frame["lfc_sample_weighted"].rank(method="average", pct=True)

        effective_sizes = pd.to_numeric(group_frame["effective_sample_cluster_size"], errors="coerce")
        if effective_sizes.notna().any():
            eff_low = max(float(effective_sizes.quantile(0.25)), 2.0)
            eff_high = float(effective_sizes.quantile(0.75))
        else:
            eff_low = np.nan
            eff_high = np.nan

        run_level.loc[group_index, "eff_low_threshold"] = eff_low
        run_level.loc[group_index, "eff_high_threshold"] = eff_high

        penalty = pd.Series(np.nan, index=group_index, dtype="float64")
        valid_eff_mask = effective_sizes.notna() & (effective_sizes > 0) & pd.notna(eff_low) & pd.notna(eff_high)
        if valid_eff_mask.any():
            valid_eff = effective_sizes.loc[valid_eff_mask]
            penalty.loc[valid_eff.index] = np.minimum(1.0, valid_eff / eff_low) * np.minimum(1.0, eff_high / valid_eff)
        run_level.loc[group_index, "effective_size_penalty"] = penalty

    zero_enriched_mask = run_level["sample_mass_enriched"].eq(0) & run_level["_total_sample"].gt(0)
    run_level.loc[zero_enriched_mask, "effective_size_penalty"] = 0.0
    run_level.loc[zero_enriched_mask, "lfc_rank_pct"] = 0.0

    run_level["redcea_dense_score_base"] = run_level["coverage_rank_pct"] * run_level["effective_size_penalty"]
    run_level["redcea_dense_score"] = run_level["redcea_dense_score_base"] * run_level["lfc_rank_pct"]
    run_level["redcea_dense_score_soft"] = run_level["redcea_dense_score_base"] * (0.5 + 0.5 * run_level["lfc_rank_pct"])

    output_columns = [
        "run_id",
        *resolved_group_cols,
        "redcea_dense_score",
        "redcea_dense_score_soft",
        "redcea_dense_score_base",
        "coverage_rank_pct",
        "sample_coverage_enriched",
        "sample_mass_enriched",
        "cluster_mass_enriched",
        "background_mass_enriched",
        "effective_n_enriched_clusters_sample_weighted",
        "effective_sample_cluster_size",
        "effective_size_penalty",
        "eff_low_threshold",
        "eff_high_threshold",
        "lfc_sample_weighted",
        "lfc_rank_pct",
    ]
    return run_level.loc[:, output_columns]


def compute_lfc_mass_shift(cluster_df: pd.DataFrame, alpha: float = 1.5) -> pd.DataFrame:
    if cluster_df.empty or "run_id" not in cluster_df.columns:
        return pd.DataFrame()

    prepared = _prepare_cluster_metric_inputs(cluster_df)
    rows: list[dict[str, object]] = []
    for run_id, frame in prepared.groupby("run_id", sort=True):
        non_noise_mask = _non_noise_cluster_mask(frame)
        significant_mask = non_noise_mask & (pd.to_numeric(frame["enrichment_fdr_zbinom"], errors="coerce") < 0.05)
        positive_mask = significant_mask & (pd.to_numeric(frame["log_fold_change"], errors="coerce") > 0)
        negative_mask = significant_mask & (pd.to_numeric(frame["log_fold_change"], errors="coerce") < 0)

        pos_sample = pd.to_numeric(frame.loc[positive_mask, "sample"], errors="coerce").fillna(0.0)
        neg_sample = pd.to_numeric(frame.loc[negative_mask, "sample"], errors="coerce").fillna(0.0)
        pos_cluster_size = pd.to_numeric(frame.loc[positive_mask, "cluster_size"], errors="coerce")
        neg_cluster_size = pd.to_numeric(frame.loc[negative_mask, "cluster_size"], errors="coerce")
        pos_lfc = pd.to_numeric(frame.loc[positive_mask, "log_fold_change"], errors="coerce").abs()
        neg_lfc = pd.to_numeric(frame.loc[negative_mask, "log_fold_change"], errors="coerce").abs()

        lfc_pos_mass = float((pos_lfc * (pos_sample**alpha)).sum()) if len(pos_lfc) else 0.0
        lfc_neg_mass = float((neg_lfc * (neg_sample**alpha)).sum()) if len(neg_lfc) else 0.0
        denom = lfc_pos_mass + lfc_neg_mass
        lfc_mass_shift = float((lfc_pos_mass - lfc_neg_mass) / denom) if denom > 0 else np.nan

        sample_sig_pos = float(pos_sample.sum()) if len(pos_sample) else 0.0
        sample_sig_neg = float(neg_sample.sum()) if len(neg_sample) else 0.0
        cluster_mass_sig_pos = float(pos_cluster_size.sum()) if len(pos_cluster_size) else 0.0
        cluster_mass_sig_neg = float(neg_cluster_size.sum()) if len(neg_cluster_size) else 0.0

        mean_size_sig_pos, median_size_sig_pos = _mean_median(pos_cluster_size)

        rows.append(
            {
                "run_id": run_id,
                "lfc_mass_shift": lfc_mass_shift,
                "lfc_pos_mass": lfc_pos_mass,
                "lfc_neg_mass": lfc_neg_mass,
                "n_sig_pos": int(positive_mask.sum()),
                "n_sig_neg": int(negative_mask.sum()),
                "sample_sig_pos": sample_sig_pos,
                "sample_sig_neg": sample_sig_neg,
                "sample_pos_neg_ratio": _safe_ratio(sample_sig_pos, sample_sig_neg),
                "cluster_mass_sig_pos": cluster_mass_sig_pos,
                "cluster_mass_sig_neg": cluster_mass_sig_neg,
                "cluster_mass_pos_neg_ratio": _safe_ratio(cluster_mass_sig_pos, cluster_mass_sig_neg),
                "mean_size_sig_pos": mean_size_sig_pos,
                "median_size_sig_pos": median_size_sig_pos,
            }
        )
    return pd.DataFrame(rows)


def _compute_density_contrast(frame: pd.DataFrame, enriched_mask: pd.Series) -> dict[str, float]:
    default = {
        "cohesion_enriched__mean": np.nan,
        "cohesion_non_enriched__mean": np.nan,
        "density_proxy_enriched__mean": np.nan,
        "density_proxy_non_enriched__mean": np.nan,
        "density_proxy_enriched_to_non_enriched__ratio": np.nan,
        "density_proxy_enriched_to_non_enriched__log2_ratio": np.nan,
    }
    if "cohesion" not in frame.columns:
        return default

    non_noise_mask = _non_noise_cluster_mask(frame)
    valid_cohesion = pd.to_numeric(frame["cohesion"], errors="coerce")
    valid_mask = non_noise_mask & valid_cohesion.notna() & (valid_cohesion != 0)
    if not valid_mask.any():
        return default

    enriched_cohesion = valid_cohesion.loc[valid_mask & enriched_mask]
    non_enriched_cohesion = valid_cohesion.loc[valid_mask & (~enriched_mask)]

    enriched_mean = float(enriched_cohesion.mean()) if enriched_cohesion.notna().any() else np.nan
    non_enriched_mean = float(non_enriched_cohesion.mean()) if non_enriched_cohesion.notna().any() else np.nan

    density_proxy_enriched = 1.0 / enriched_cohesion if enriched_cohesion.notna().any() else pd.Series(dtype=float)
    density_proxy_non_enriched = (
        1.0 / non_enriched_cohesion if non_enriched_cohesion.notna().any() else pd.Series(dtype=float)
    )

    density_proxy_enriched_mean = (
        float(density_proxy_enriched.mean()) if density_proxy_enriched.notna().any() else np.nan
    )
    density_proxy_non_enriched_mean = (
        float(density_proxy_non_enriched.mean()) if density_proxy_non_enriched.notna().any() else np.nan
    )

    ratio = np.nan
    log2_ratio = np.nan
    if (
        pd.notna(density_proxy_enriched_mean)
        and pd.notna(density_proxy_non_enriched_mean)
        and density_proxy_non_enriched_mean != 0
    ):
        ratio = float(density_proxy_enriched_mean / density_proxy_non_enriched_mean)
        if ratio > 0:
            log2_ratio = float(np.log2(ratio))

    default.update(
        {
            "cohesion_enriched__mean": enriched_mean,
            "cohesion_non_enriched__mean": non_enriched_mean,
            "density_proxy_enriched__mean": density_proxy_enriched_mean,
            "density_proxy_non_enriched__mean": density_proxy_non_enriched_mean,
            "density_proxy_enriched_to_non_enriched__ratio": ratio,
            "density_proxy_enriched_to_non_enriched__log2_ratio": log2_ratio,
        }
    )
    return default


def build_run_level_table(cluster_df: pd.DataFrame) -> pd.DataFrame:
    if cluster_df.empty:
        return pd.DataFrame()

    cluster_df = _prepare_cluster_metric_inputs(cluster_df)
    metadata_like_columns = [
        "run_id",
        "run_dir",
        "dataset",
        "dataset_mode",
        "method",
        "parameter_json",
        "n_points",
        "n_clusters",
        "n_noise",
        "noise_fraction",
        "runtime_seconds",
        "status",
        "error_message",
        "error_traceback",
        "redcea_output_dir",
    ]
    metadata_like_columns.extend([column for column in cluster_df.columns if column.startswith("param_")])
    metadata_like_columns = [column for column in metadata_like_columns if column in cluster_df.columns]

    metric_columns = [
        column
        for column in cluster_df.columns
        if column not in metadata_like_columns
        and column not in {"cluster_id"}
        and pd.api.types.is_numeric_dtype(cluster_df[column])
    ]

    rows: list[dict[str, object]] = []
    for run_id, frame in cluster_df.groupby("run_id", sort=True):
        row: dict[str, object] = {"run_id": run_id}
        for column in metadata_like_columns:
            row[column] = frame.iloc[0][column]
        row["n_clusters_in_summary"] = int(len(frame))
        non_noise_mask = _non_noise_cluster_mask(frame)
        enriched_mask = _enriched_cluster_mask(frame)
        all_clusters = frame.loc[non_noise_mask].copy()
        enriched_clusters = frame.loc[enriched_mask].copy()

        row["n_clusters_total"] = int(len(all_clusters))
        row["n_clusters_enriched"] = int(len(enriched_clusters))

        cluster_size_all_mean, cluster_size_all_median = _mean_median(all_clusters.get("cluster_size", pd.Series(dtype=float)))
        cluster_size_enriched_mean, cluster_size_enriched_median = _mean_median(
            enriched_clusters.get("cluster_size", pd.Series(dtype=float))
        )
        row["cluster_size_all__mean"] = cluster_size_all_mean
        row["cluster_size_all__median"] = cluster_size_all_median
        row["cluster_size_enriched__mean"] = cluster_size_enriched_mean
        row["cluster_size_enriched__median"] = cluster_size_enriched_median

        sample_fraction_all = pd.Series(dtype=float)
        sample_fraction_enriched = pd.Series(dtype=float)
        if {"sample", "background"}.issubset(frame.columns):
            all_cluster_size = pd.to_numeric(all_clusters["sample"], errors="coerce") + pd.to_numeric(
                all_clusters["background"], errors="coerce"
            )
            enriched_cluster_size = pd.to_numeric(enriched_clusters["sample"], errors="coerce") + pd.to_numeric(
                enriched_clusters["background"], errors="coerce"
            )
            sample_fraction_all = pd.Series(
                np.where(
                    all_cluster_size > 0,
                    pd.to_numeric(all_clusters["sample"], errors="coerce") / all_cluster_size,
                    np.nan,
                ),
                index=all_clusters.index,
                dtype="float64",
            )
            sample_fraction_enriched = pd.Series(
                np.where(
                    enriched_cluster_size > 0,
                    pd.to_numeric(enriched_clusters["sample"], errors="coerce") / enriched_cluster_size,
                    np.nan,
                ),
                index=enriched_clusters.index,
                dtype="float64",
            )
        sample_fraction_all_mean, sample_fraction_all_median = _mean_median(sample_fraction_all)
        sample_fraction_enriched_mean, sample_fraction_enriched_median = _mean_median(sample_fraction_enriched)
        row["sample_fraction_all__mean"] = sample_fraction_all_mean
        row["sample_fraction_all__median"] = sample_fraction_all_median
        row["sample_fraction_enriched__mean"] = sample_fraction_enriched_mean
        row["sample_fraction_enriched__median"] = sample_fraction_enriched_median
        row.update(_compute_density_contrast(frame, enriched_mask))

        if "is_good_candidate" in frame.columns:
            row["good_candidate_count"] = int(frame["is_good_candidate"].fillna(False).astype(bool).sum())
            row["good_candidate_fraction"] = float(frame["is_good_candidate"].fillna(False).astype(float).mean())
        if "is_good_candidate_relaxed" in frame.columns:
            row["good_candidate_relaxed_count"] = int(frame["is_good_candidate_relaxed"].fillna(False).astype(bool).sum())
            row["good_candidate_relaxed_fraction"] = float(
                frame["is_good_candidate_relaxed"].fillna(False).astype(float).mean()
            )
        for column in metric_columns:
            stats = numeric_summary(frame[column])
            for suffix, value in stats.items():
                row[f"{column}__{suffix}"] = value
        rows.append(row)

    run_level_df = pd.DataFrame(rows)
    dense_scores_df = compute_dense_redcea_scores(cluster_df)
    lfc_mass_shift_df = compute_lfc_mass_shift(cluster_df)

    if not dense_scores_df.empty:
        dense_metric_columns = [column for column in dense_scores_df.columns if column == "run_id" or column not in metadata_like_columns]
        run_level_df = run_level_df.merge(dense_scores_df.loc[:, dense_metric_columns], on="run_id", how="left")
    if not lfc_mass_shift_df.empty:
        run_level_df = run_level_df.merge(lfc_mass_shift_df, on="run_id", how="left")
    return run_level_df


def progress_message(
    *,
    done: int,
    total: int,
    ok: int,
    skipped: int,
    failed: int,
    started_at: float,
    run_id: str,
    status: str,
    message: str,
) -> str:
    elapsed = time.perf_counter() - started_at
    return (
        "Progress {done}/{total} | ok={ok} skipped={skipped} failed={failed} | "
        "elapsed={elapsed:.1f}s | {run_id}: {status} ({message})"
    ).format(
        done=done,
        total=total,
        ok=ok,
        skipped=skipped,
        failed=failed,
        elapsed=elapsed,
        run_id=run_id,
        status=status,
        message=message,
    )


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root)
    execution_manifest = Path(args.execution_manifest)
    cluster_output = Path(args.cluster_output)
    run_output = Path(args.run_output)
    log_path = Path(args.log_path) if args.log_path else None

    setup_logging(log_path)
    logging.info("Starting additional metrics batch run")
    logging.info(
        "runs_root=%s execution_manifest=%s workers=%d rewrite_existing=%s dataset_mode=%s run_id_prefix=%s",
        runs_root,
        execution_manifest,
        args.workers,
        args.rewrite_existing,
        args.dataset_mode,
        args.run_id_prefix,
    )

    manifest_df = load_execution_manifest(execution_manifest)
    run_dirs = select_run_dirs(
        runs_root=runs_root,
        manifest_lookup=manifest_df,
        dataset_mode=args.dataset_mode,
        run_id_prefix=args.run_id_prefix,
    )
    total = len(run_dirs)
    if total == 0:
        logging.warning(
            "No run directories matched under %s for dataset_mode=%s run_id_prefix=%s",
            runs_root,
            args.dataset_mode,
            args.run_id_prefix,
        )
        return 0

    started_at = time.perf_counter()
    results: list[dict[str, object]] = []
    ok = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_run, str(run_dir), args.rewrite_existing): run_dir.name
            for run_dir in run_dirs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            run_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "run_id": run_id,
                    "run_dir": str(runs_root / run_id),
                    "status": "failed",
                    "message": f"WorkerCrash: {exc}",
                    "runtime_seconds": np.nan,
                    "cluster_rows": 0,
                    "summary_rows": 0,
                    "summary_path": "",
                    "metrics_added": False,
                    "traceback": traceback.format_exc(),
                }
            results.append(result)

            status = str(result["status"])
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1

            logging.info(
                progress_message(
                    done=index,
                    total=total,
                    ok=ok,
                    skipped=skipped,
                    failed=failed,
                    started_at=started_at,
                    run_id=str(result["run_id"]),
                    status=status,
                    message=str(result["message"]),
                )
            )
            if result.get("traceback"):
                logging.error("Traceback for %s\n%s", result["run_id"], result["traceback"])
            if args.fail_fast and status == "failed":
                logging.error("Fail-fast enabled, stopping after first failure")
                break

    result_df = pd.DataFrame(results).sort_values(["status", "run_id"]).reset_index(drop=True)
    result_manifest_path = run_output.parent / "run_additional_metrics_batch_manifest.tsv"
    atomic_write_tsv(result_df, result_manifest_path)
    logging.info("Wrote worker manifest: %s", result_manifest_path)

    manifest_run_ids = set(manifest_df["run_id"].astype(str)) if not manifest_df.empty else set()
    existing_run_ids = {run_dir.name for run_dir in run_dirs}
    matched_runs = len(existing_run_ids & manifest_run_ids)
    logging.info(
        "Manifest coverage over existing runs: matched=%d missing=%d total_runs=%d",
        matched_runs,
        len(existing_run_ids - manifest_run_ids),
        len(existing_run_ids),
    )

    cluster_level_df = collect_cluster_level_table(run_dirs=run_dirs, manifest_lookup=manifest_df)
    run_level_df = build_run_level_table(cluster_level_df)
    atomic_write_tsv(cluster_level_df, cluster_output)
    atomic_write_tsv(run_level_df, run_output)
    logging.info("Wrote cluster-level table: %s rows=%d", cluster_output, len(cluster_level_df))
    logging.info("Wrote run-level table: %s rows=%d", run_output, len(run_level_df))

    total_elapsed = time.perf_counter() - started_at
    logging.info(
        "Finished additional metrics batch run | total=%d ok=%d skipped=%d failed=%d elapsed=%.1fs",
        total,
        ok,
        skipped,
        failed,
        total_elapsed,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
