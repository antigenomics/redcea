from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_SUMMARY_COLUMNS = {
    "cluster_id",
    "cluster_size",
    "sample",
    "background",
    "enrichment_pvalue_zbinom",
    "enrichment_fdr_zbinom",
    "log_fold_change",
}

DBCV_FAMILY_TOKENS = (
    "density_validity",
    "geometry_score",
    "dbcv_with_odds",
    "redcea_auxiliary_validity",
)


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


def summary_path_for_run(run_dir: Path) -> Path | None:
    matches = sorted(run_dir.glob("*_summary_tcrempnet.tsv"))
    return matches[0] if matches else None


def detect_summary_metric_columns(runs_root: Path) -> list[str]:
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        summary_path = summary_path_for_run(run_dir)
        if summary_path is None:
            continue
        summary_df = pd.read_csv(summary_path, sep="\t", nrows=1)
        return [column for column in summary_df.columns if column not in BASE_SUMMARY_COLUMNS]
    raise RuntimeError("Could not detect summary metric columns from results/redcea_runs")


def build_cluster_level_table(
    *,
    runs_root: Path,
    vdjdb_metrics_path: Path,
    run_metadata_path: Path,
) -> pd.DataFrame:
    vdjdb_metrics = pd.read_csv(vdjdb_metrics_path, sep="\t")
    run_metadata = pd.read_csv(run_metadata_path, sep="\t")
    run_metadata = run_metadata.loc[run_metadata["dataset_mode"] == "vdjdb"].copy()

    metric_lookup = vdjdb_metrics[
        [
            "run_id",
            "epitope",
            "precision",
            "recall",
            "f1",
            "weighted_cluster_purity",
            "positive_cluster_concentration_top1",
            "positive_cluster_concentration_top5",
            "positive_fragmentation",
            "positive_fragmentation_norm",
        ]
    ].copy()

    frames: list[pd.DataFrame] = []
    for _, row in run_metadata.iterrows():
        run_id = str(row["run_id"])
        run_dir = runs_root / run_id
        summary_path = summary_path_for_run(run_dir)
        if summary_path is None:
            continue
        summary_df = pd.read_csv(summary_path, sep="\t")
        summary_df["run_id"] = run_id
        summary_df["dataset"] = row.get("dataset")
        summary_df["dataset_mode"] = row.get("dataset_mode")
        summary_df["method"] = row.get("method")
        summary_df["parameter_json"] = row.get("parameter_json")
        summary_df["n_points"] = row.get("n_points")
        summary_df["n_clusters_run"] = row.get("n_clusters")
        summary_df["n_noise"] = row.get("n_noise")
        summary_df["noise_fraction_run"] = row.get("noise_fraction")
        summary_df["runtime_seconds"] = row.get("runtime_seconds")
        frames.append(summary_df)

    cluster_level_df = pd.concat(frames, ignore_index=True)
    cluster_level_df = cluster_level_df.merge(metric_lookup, on="run_id", how="left")

    parameter_dicts = cluster_level_df["parameter_json"].apply(parse_parameter_json)
    parameter_df = pd.json_normalize(parameter_dicts).add_prefix("param_")
    cluster_level_df = pd.concat([cluster_level_df, parameter_df], axis=1)
    return cluster_level_df


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return np.nan
    values = values.loc[mask].astype(float)
    weights = weights.loc[mask].astype(float)
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return np.nan
    return float(np.average(values, weights=weights))


def trimmed_mean(values: pd.Series, proportion: float = 0.10) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    if clean.size == 0:
        return np.nan
    if clean.size < 3 or proportion <= 0:
        return float(clean.mean())
    clean.sort()
    trim_n = int(np.floor(clean.size * proportion))
    if trim_n == 0 or (2 * trim_n) >= clean.size:
        return float(clean.mean())
    trimmed = clean[trim_n : clean.size - trim_n]
    return float(trimmed.mean()) if trimmed.size else np.nan


def quantile_value(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    if clean.size == 0:
        return np.nan
    return float(np.quantile(clean, q))


def gt_zero_fraction(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return np.nan
    return float((clean > 0).mean())


def build_run_level_table(
    cluster_level_df: pd.DataFrame,
    *,
    summary_metric_columns: list[str],
) -> pd.DataFrame:
    id_columns = [
        "run_id",
        "epitope",
        "method",
        "parameter_json",
        "precision",
        "recall",
        "f1",
        "weighted_cluster_purity",
        "positive_cluster_concentration_top1",
        "positive_cluster_concentration_top5",
        "positive_fragmentation",
        "positive_fragmentation_norm",
        "n_points",
        "n_clusters_run",
        "n_noise",
        "noise_fraction_run",
        "runtime_seconds",
    ]
    param_columns = [column for column in cluster_level_df.columns if column.startswith("param_")]
    group_columns = id_columns + param_columns

    numeric_metric_columns = [
        column
        for column in summary_metric_columns
        if column in cluster_level_df.columns and column != "is_good_candidate"
    ]

    run_rows: list[dict[str, object]] = []
    for keys, frame in cluster_level_df.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row["n_clusters_in_summary"] = int(len(frame))
        if "is_good_candidate" in frame.columns:
            row["good_candidate_fraction"] = float(frame["is_good_candidate"].fillna(False).astype(float).mean())
            row["good_candidate_count"] = int(frame["is_good_candidate"].fillna(False).astype(bool).sum())
        if "is_good_candidate_relaxed" in frame.columns:
            row["good_candidate_relaxed_fraction"] = float(
                frame["is_good_candidate_relaxed"].fillna(False).astype(float).mean()
            )
            row["good_candidate_relaxed_count"] = int(
                frame["is_good_candidate_relaxed"].fillna(False).astype(bool).sum()
            )

        for column in numeric_metric_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            row[f"{column}__mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{column}__median"] = float(values.median()) if values.notna().any() else np.nan
            row[f"{column}__max"] = float(values.max()) if values.notna().any() else np.nan
            row[f"{column}__q25"] = quantile_value(values, 0.25)
            row[f"{column}__q75"] = quantile_value(values, 0.75)
            row[f"{column}__trimmed_mean_10"] = trimmed_mean(values, proportion=0.10)
            if "sample_usage" in frame.columns:
                row[f"{column}__sample_usage_weighted_mean"] = weighted_mean(values, frame["sample_usage"])
            if any(token in column for token in DBCV_FAMILY_TOKENS):
                row[f"{column}__gt0_fraction"] = gt_zero_fraction(values)
        run_rows.append(row)
    return pd.DataFrame(run_rows)


def compute_metric_correlations(
    run_level_df: pd.DataFrame,
    *,
    target: str = "f1",
    min_points: int = 3,
) -> pd.DataFrame:
    metric_columns = [
        column
        for column in run_level_df.columns
        if column not in {"run_id", "epitope", "method", "parameter_json", target}
        and not column.startswith("param_")
        and pd.api.types.is_numeric_dtype(run_level_df[column])
    ]
    rows: list[dict[str, object]] = []
    for column in metric_columns:
        frame = run_level_df[[target, column]].dropna()
        if len(frame) < min_points:
            continue
        pearson = frame[target].corr(frame[column], method="pearson")
        spearman = frame[target].corr(frame[column], method="spearman")
        rows.append(
            {
                "metric": column,
                "n_points": int(len(frame)),
                "pearson": float(pearson) if pd.notna(pearson) else np.nan,
                "pearson_abs": float(abs(pearson)) if pd.notna(pearson) else np.nan,
                "spearman": float(spearman) if pd.notna(spearman) else np.nan,
                "spearman_abs": float(abs(spearman)) if pd.notna(spearman) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["spearman_abs", "pearson_abs"], ascending=False).reset_index(drop=True)


__all__ = [
    "BASE_SUMMARY_COLUMNS",
    "DBCV_FAMILY_TOKENS",
    "build_cluster_level_table",
    "build_run_level_table",
    "compute_metric_correlations",
    "detect_summary_metric_columns",
    "parse_parameter_json",
    "summary_path_for_run",
]
