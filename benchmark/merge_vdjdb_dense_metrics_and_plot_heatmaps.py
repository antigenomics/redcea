from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.evaluation import _compute_vdjdb_recovered_mask


METHOD_ORDER = [
    "hierarchical_leiden",
    "leiden",
    "leiden_dbscan",
    "dbscan",
    "vdbscan_leiden",
    "vdbscan",
]

DENSE_SCORE_COLUMNS = [
    "redcea_dense_score",
    "redcea_dense_score_soft",
    "redcea_dense_score_base",
]

VDJDB_QUALITY_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "metric_source",
]

VDJDB_EPITOPE_LOOKUP = {
    "glc": "GLCTLVAML",
    "ylq": "YLQPRTFLL",
}

HEATMAP_METRICS = [
    "f1",
    "redcea_dense_score",
]

PARAMETER_DISPLAY_ORDER = {
    "dbscan": ["cluster_min_samples", "eps_k_neighbors", "k_neighbors"],
    "hierarchical_leiden": ["cluster_min_samples", "k_neighbors", "leiden_resolution", "leiden_sub_resolution"],
    "leiden": ["cluster_min_samples", "k_neighbors", "leiden_resolution"],
    "leiden_dbscan": ["cluster_min_samples", "eps_k_neighbors", "k_neighbors", "leiden_resolution"],
    "vdbscan": ["cluster_min_samples", "eps_estimation_based_on", "eps_k_neighbors", "k_neighbors", "vdbscan_sym_rule"],
    "vdbscan_leiden": [
        "cluster_min_samples",
        "eps_estimation_based_on",
        "eps_k_neighbors",
        "k_neighbors",
        "leiden_resolution",
        "vdbscan_sym_rule",
    ],
}

PARAMETER_SHORT_LABELS = {
    "cluster_min_samples": "cms",
    "eps_estimation_based_on": "eps_src",
    "eps_k_neighbors": "epsk",
    "k_neighbors": "k",
    "leiden_resolution": "lr",
    "leiden_sub_resolution": "sub_lr",
    "vdbscan_sym_rule": "sym",
}

METHOD_COLORS = {
    "hierarchical_leiden": "#1f77b4",
    "leiden": "#2ca02c",
    "leiden_dbscan": "#9467bd",
    "dbscan": "#ff7f0e",
    "vdbscan_leiden": "#d62728",
    "vdbscan": "#8c564b",
}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{0:02x}{1:02x}{2:02x}".format(*rgb)


def interpolate_color(start: tuple[int, int, int], end: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = clamp01(float(t))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(start, end))


def luminance(rgb: tuple[int, int, int]) -> float:
    red, green, blue = rgb
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def sequential_color(
    value: float,
    *,
    vmin: float,
    vmax: float,
    low: tuple[int, int, int],
    high: tuple[int, int, int],
) -> tuple[int, int, int]:
    if not np.isfinite(value):
        return (238, 238, 238)
    if vmax <= vmin:
        return high
    t = (float(value) - vmin) / (vmax - vmin)
    return interpolate_color(low, high, t)


def diverging_color(
    value: float,
    *,
    limit: float,
    negative: tuple[int, int, int],
    neutral: tuple[int, int, int],
    positive: tuple[int, int, int],
) -> tuple[int, int, int]:
    if not np.isfinite(value):
        return (238, 238, 238)
    if limit <= 0:
        return neutral
    scaled = clamp01(abs(float(value)) / limit)
    if value < 0:
        return interpolate_color(neutral, negative, scaled)
    return interpolate_color(neutral, positive, scaled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge VDJdb dense-score metrics into the base run-metrics table and build "
            "method-by-epitope heatmaps for F1 and dense score."
        )
    )
    parser.add_argument(
        "--base-run-metrics",
        type=Path,
        default=Path("results/metrics/redcea_all_run_metrics.tsv"),
        help="Original consolidated run-level metrics table.",
    )
    parser.add_argument(
        "--new-run-metrics",
        type=Path,
        required=True,
        help="New run-level metrics table that contains dense-score columns.",
    )
    parser.add_argument(
        "--redcea-runs-dir",
        type=Path,
        default=Path("results/redcea_runs"),
        help="Directory with per-run REDCEA outputs and *_tcremp_clusters.tsv files.",
    )
    parser.add_argument(
        "--vdjdb-selected-runs",
        type=Path,
        default=Path("results/metrics/vdjdb_selected_runs.tsv"),
        help="Optional table with selected VDJdb runs that already contains F1.",
    )
    parser.add_argument(
        "--vdjdb-clustering-metrics",
        type=Path,
        default=Path("results/metrics/vdjdb_clustering_metrics.tsv"),
        help="Optional compact VDJdb metrics table that already contains F1.",
    )
    parser.add_argument(
        "--truth-table",
        type=Path,
        default=Path("data/01_05_2025_TCRvdb.csv"),
        help="TCRvdb truth table used to recompute VDJdb F1 for every run.",
    )
    parser.add_argument(
        "--padj-threshold",
        type=float,
        default=1e-5,
        help="Adjusted p-value threshold used to label positive vs negative VDJdb clonotypes.",
    )
    parser.add_argument(
        "--vdjdb-repaired-metadata",
        type=Path,
        default=Path("results/run_metadata/clustering_runs_vdjdb_repaired.tsv"),
        help="Metadata table with parameter_json for repaired VDJdb runs.",
    )
    parser.add_argument(
        "--output-run-metrics",
        type=Path,
        default=Path("results/metrics/redcea_all_run_metrics.tsv"),
        help="Path for the merged run-level metrics table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/vdjdb_dense_score_heatmaps"),
        help="Directory for merged VDJdb tables and heatmaps.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def normalize_segment(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
        .str.replace(r"\*.*$", "", regex=True)
        .str.replace("/", "_", regex=False)
    )


def merge_dense_score_columns(base_df: pd.DataFrame, new_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    missing_columns = [column for column in DENSE_SCORE_COLUMNS if column not in new_df.columns]
    if missing_columns:
        raise KeyError(
            "New run-metrics table is missing required dense-score columns: {0}".format(", ".join(missing_columns))
        )

    if "dataset_mode" not in base_df.columns or "dataset_mode" not in new_df.columns:
        raise KeyError("Both run-metrics tables must contain a dataset_mode column.")

    base_out = base_df.copy()
    base_vdjdb_mask = base_out["dataset_mode"].astype(str).eq("vdjdb")
    new_vdjdb_df = new_df.loc[new_df["dataset_mode"].astype(str).eq("vdjdb"), ["run_id", *DENSE_SCORE_COLUMNS]].copy()

    duplicate_mask = new_vdjdb_df["run_id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_ids = sorted(new_vdjdb_df.loc[duplicate_mask, "run_id"].astype(str).unique().tolist())
        raise ValueError("New VDJdb metrics contain duplicated run_id values: {0}".format(", ".join(duplicate_ids[:10])))

    dense_lookup = new_vdjdb_df.set_index("run_id")
    base_vdjdb_run_ids = set(base_out.loc[base_vdjdb_mask, "run_id"].astype(str))
    new_vdjdb_run_ids = set(dense_lookup.index.astype(str))
    missing_run_ids = sorted(base_vdjdb_run_ids - new_vdjdb_run_ids)

    for column in DENSE_SCORE_COLUMNS:
        base_out.loc[base_vdjdb_mask, column] = (
            base_out.loc[base_vdjdb_mask, "run_id"].astype(str).map(dense_lookup[column])
        )

    stats = {
        "base_vdjdb_rows": int(base_vdjdb_mask.sum()),
        "new_vdjdb_rows": int(len(dense_lookup)),
        "matched_vdjdb_rows": int(base_out.loc[base_vdjdb_mask, "redcea_dense_score"].notna().sum()),
        "missing_vdjdb_run_ids": int(len(missing_run_ids)),
    }
    if missing_run_ids:
        raise ValueError(
            "Dense-score merge left missing VDJdb run_ids in the base table. First examples: {0}".format(
                ", ".join(missing_run_ids[:10])
            )
        )
    return base_out, stats


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, sep="\t", index=False)
    tmp_path.replace(path)


def build_vdjdb_truth_table(truth_path: Path, *, padj_threshold: float) -> pd.DataFrame:
    truth = pd.read_csv(truth_path).drop(columns=["Unnamed: 0"], errors="ignore")
    truth["chain"] = "TRB"
    truth["cdr3"] = truth["cdr3_beta_aa"].fillna("").astype(str)
    truth["v_gene"] = normalize_segment(truth["TRBV"])
    truth["j_gene"] = normalize_segment(truth["TRBJ"])
    truth["truth_label"] = np.where(
        truth["padj"].notna(),
        np.where(truth["padj"] < float(padj_threshold), "positive", "negative"),
        "unlabeled",
    )
    return truth.loc[:, ["cdr3", "v_gene", "j_gene", "chain", "epitope_aa", "truth_label"]].copy()


def parse_parameter_json(value: object) -> dict[str, object]:
    import json

    if pd.isna(value):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def infer_algorithm(run_id: str) -> str:
    value = str(run_id)
    for algorithm in ["hierarchical_leiden", "leiden_dbscan", "vdbscan_leiden", "dbscan", "vdbscan", "leiden"]:
        if "_{0}_".format(algorithm) in value:
            return algorithm
    raise ValueError("Could not infer algorithm from run_id: {0}".format(run_id))


def load_vdjdb_f1_source(path: Path, *, source_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    required_columns = {"run_id", "epitope", "method", "f1"}
    if not required_columns.issubset(df.columns):
        return pd.DataFrame()
    keep_columns = [column for column in ["run_id", "epitope", "method", "parameter_json", "precision", "recall", "f1"] if column in df.columns]
    out = df.loc[:, keep_columns].copy()
    out["metric_source"] = source_name
    return out.drop_duplicates(subset=["run_id"], keep="first").reset_index(drop=True)


def infer_epitope_from_run_id(run_id: str) -> str:
    lowered = str(run_id).lower()
    for token, epitope in VDJDB_EPITOPE_LOOKUP.items():
        if "_{0}_".format(token) in lowered:
            return epitope
    raise ValueError("Could not infer VDJdb epitope from run_id: {0}".format(run_id))


def load_repaired_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["run_id", "parameter_json", "method"])
    df = pd.read_csv(path, sep="\t")
    keep_columns = [column for column in ["run_id", "parameter_json", "method", "status"] if column in df.columns]
    return df.loc[:, keep_columns].drop_duplicates(subset=["run_id"], keep="first").reset_index(drop=True)


def enrich_vdjdb_run_metrics_with_parameters(run_metrics_df: pd.DataFrame, repaired_metadata_df: pd.DataFrame) -> pd.DataFrame:
    vdjdb_df = run_metrics_df.loc[run_metrics_df["dataset_mode"].astype(str).eq("vdjdb")].copy()
    existing_param_columns = [column for column in vdjdb_df.columns if column.startswith("param_")]
    if existing_param_columns:
        vdjdb_df = vdjdb_df.drop(columns=existing_param_columns)
    if len(repaired_metadata_df):
        vdjdb_df = vdjdb_df.merge(
            repaired_metadata_df.rename(
                columns={
                    "parameter_json": "parameter_json_repaired",
                    "method": "method_repaired",
                    "status": "status_repaired",
                }
            ),
            on="run_id",
            how="left",
        )
    else:
        vdjdb_df["parameter_json_repaired"] = np.nan
        vdjdb_df["method_repaired"] = np.nan
        vdjdb_df["status_repaired"] = np.nan

    vdjdb_df["algorithm"] = vdjdb_df["run_id"].astype(str).apply(infer_algorithm)
    vdjdb_df["epitope"] = vdjdb_df["run_id"].astype(str).apply(infer_epitope_from_run_id)
    vdjdb_df["parameter_json_filled"] = vdjdb_df["parameter_json"]
    missing_param_mask = vdjdb_df["parameter_json_filled"].isna() | vdjdb_df["parameter_json_filled"].astype(str).str.strip().eq("")
    vdjdb_df.loc[missing_param_mask, "parameter_json_filled"] = vdjdb_df.loc[missing_param_mask, "parameter_json_repaired"]
    parameter_dicts = vdjdb_df["parameter_json_filled"].apply(parse_parameter_json)
    parameter_df = pd.json_normalize(parameter_dicts)
    if len(parameter_df):
        parameter_df = parameter_df.add_prefix("param_")
        vdjdb_df = pd.concat([vdjdb_df.reset_index(drop=True), parameter_df.reset_index(drop=True)], axis=1)
    return vdjdb_df


def parameter_label_for_row(row: pd.Series) -> str:
    algorithm = str(row["algorithm"])
    ordered_keys = PARAMETER_DISPLAY_ORDER.get(algorithm, [])
    parts: list[str] = []
    for key in ordered_keys:
        column = "param_{0}".format(key)
        value = row.get(column, np.nan)
        if pd.isna(value):
            continue
        if isinstance(value, float) and float(value).is_integer():
            value = int(value)
        label = PARAMETER_SHORT_LABELS.get(key, key)
        parts.append("{0}={1}".format(label, value))
    if not parts:
        return "unknown_params"
    return " | ".join(parts)


def build_parameter_heatmap_input(
    vdjdb_runs_df: pd.DataFrame,
    vdjdb_metric_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dense_df = vdjdb_runs_df.loc[:, ["run_id", "algorithm", "epitope", "parameter_json_filled", *DENSE_SCORE_COLUMNS] + [column for column in vdjdb_runs_df.columns if column.startswith("param_")]].copy()
    dense_df["parameter_label"] = dense_df.apply(parameter_label_for_row, axis=1)

    f1_df = vdjdb_metric_df.merge(
        vdjdb_runs_df.loc[:, ["run_id", "algorithm", "epitope", "parameter_json_filled"] + [column for column in vdjdb_runs_df.columns if column.startswith("param_")]],
        on=["run_id", "epitope"],
        how="left",
    )
    f1_df["algorithm"] = f1_df["algorithm"].fillna(f1_df["run_id"].astype(str).apply(infer_algorithm))
    f1_df["parameter_label"] = f1_df.apply(parameter_label_for_row, axis=1)
    return dense_df, f1_df


def sort_parameter_rows(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in frame.columns if column.startswith("param_")]
    if sort_columns:
        return frame.sort_values(sort_columns + ["parameter_label"]).reset_index(drop=True)
    return frame.sort_values("parameter_label").reset_index(drop=True)


def standardize_cluster_frame(cluster_df: pd.DataFrame, *, epitope: str, truth_table: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=cluster_df.index)
    out["run_id"] = ""
    out["epitope"] = epitope
    out["method"] = ""
    out["parameter_json"] = "{}"
    out["cluster_id"] = pd.to_numeric(cluster_df["cluster_id"], errors="raise").astype(int)
    out["is_noise"] = out["cluster_id"].eq(-1)
    out["cdr3"] = cluster_df["cdr3aa_beta"].fillna("").astype(str)
    out["v_gene"] = normalize_segment(cluster_df["v_beta"])
    out["j_gene"] = normalize_segment(cluster_df["j_beta"])
    out["chain"] = "TRB"
    clone_id = cluster_df["clone_id"].fillna("").astype(str)
    out["sample_label"] = np.where(
        clone_id.str.startswith("s_"),
        "sample",
        np.where(clone_id.str.startswith("b_"), "background", None),
    )
    out["truth_label"] = "unlabeled"
    sample_mask = pd.Series(out["sample_label"]).eq("sample")
    truth_subset = truth_table.loc[truth_table["epitope_aa"].astype(str) == str(epitope)].drop_duplicates(
        subset=["cdr3", "v_gene", "j_gene", "chain"],
        keep="first",
    )
    if sample_mask.any():
        merged_truth = out.loc[sample_mask, ["cdr3", "v_gene", "j_gene", "chain"]].merge(
            truth_subset.loc[:, ["cdr3", "v_gene", "j_gene", "chain", "truth_label"]],
            on=["cdr3", "v_gene", "j_gene", "chain"],
            how="left",
        )
        out.loc[sample_mask, "truth_label"] = merged_truth["truth_label"].fillna("unlabeled").to_numpy()
    return out


def compute_vdjdb_metrics_for_frame(assignments: pd.DataFrame, *, run_id: str, method: str, parameter_json: str) -> dict[str, object] | None:
    if assignments.empty:
        return None
    labeled_all = assignments.loc[assignments["truth_label"].isin(["positive", "negative"])]
    positives = labeled_all["truth_label"] == "positive"
    negatives = labeled_all["truth_label"] == "negative"
    recovered = _compute_vdjdb_recovered_mask(labeled_all)
    tp = int((positives & recovered).sum())
    fp = int((negatives & recovered).sum())
    fn = int((positives & ~recovered).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "run_id": run_id,
        "epitope": assignments["epitope"].iloc[0],
        "method": method,
        "parameter_json": parameter_json,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_vdjdb_metric_frame(
    run_metrics_df: pd.DataFrame,
    *,
    vdjdb_selected_runs_path: Path,
    vdjdb_clustering_metrics_path: Path,
    redcea_runs_dir: Path,
    truth_path: Path,
    padj_threshold: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    truth_table = build_vdjdb_truth_table(truth_path, padj_threshold=padj_threshold)
    vdjdb_runs = run_metrics_df.loc[run_metrics_df["dataset_mode"].astype(str).eq("vdjdb")].copy()
    dense_columns_df = vdjdb_runs.loc[:, ["run_id", *DENSE_SCORE_COLUMNS]].drop_duplicates(subset=["run_id"], keep="first")

    selected_runs_df = load_vdjdb_f1_source(vdjdb_selected_runs_path, source_name="vdjdb_selected_runs")
    compact_metrics_df = load_vdjdb_f1_source(vdjdb_clustering_metrics_path, source_name="vdjdb_clustering_metrics")
    combined_sources_df = pd.concat([selected_runs_df, compact_metrics_df], ignore_index=True)
    if len(combined_sources_df):
        combined_sources_df = combined_sources_df.drop_duplicates(subset=["run_id"], keep="first").reset_index(drop=True)

    rows: list[dict[str, object]] = []
    covered_run_ids = set(combined_sources_df["run_id"].astype(str)) if len(combined_sources_df) else set()
    missing_cluster_files = 0
    unreadable_runs = 0
    for row in vdjdb_runs.itertuples(index=False):
        run_id = str(row.run_id)
        if run_id in covered_run_ids:
            continue
        run_dir_value = getattr(row, "run_dir", "")
        run_dir = Path(run_dir_value) if isinstance(run_dir_value, str) and run_dir_value else redcea_runs_dir / run_id
        cluster_path = run_dir / "{0}_tcremp_clusters.tsv".format(run_id)
        if not cluster_path.exists():
            missing_cluster_files += 1
            continue
        try:
            cluster_df = pd.read_csv(cluster_path, sep="\t", low_memory=False)
            epitope = infer_epitope_from_run_id(run_id)
            assignments = standardize_cluster_frame(cluster_df, epitope=epitope, truth_table=truth_table)
            metric_row = compute_vdjdb_metrics_for_frame(
                assignments,
                run_id=run_id,
                method=str(row.method),
                parameter_json=str(getattr(row, "parameter_json", "{}") if pd.notna(getattr(row, "parameter_json", np.nan)) else "{}"),
            )
            if metric_row is None:
                continue
            for column in DENSE_SCORE_COLUMNS:
                metric_row[column] = getattr(row, column, np.nan)
            rows.append(metric_row)
        except Exception:
            unreadable_runs += 1

    recomputed_df = pd.DataFrame(rows)
    if len(recomputed_df):
        recomputed_df["metric_source"] = "recomputed_from_clusters"

    metric_df = pd.concat([combined_sources_df, recomputed_df], ignore_index=True)
    if len(metric_df):
        metric_df = metric_df.drop_duplicates(subset=["run_id"], keep="first").reset_index(drop=True)
        metric_df = metric_df.merge(dense_columns_df, on="run_id", how="left")
    if metric_df.empty:
        raise ValueError("Could not compute any VDJdb F1 rows from run cluster files.")

    metric_df["method"] = pd.Categorical(
        metric_df["method"],
        categories=[method for method in METHOD_ORDER if method in set(metric_df["method"].astype(str))],
        ordered=True,
    )
    metric_df["epitope"] = pd.Categorical(
        metric_df["epitope"],
        categories=sorted(metric_df["epitope"].dropna().astype(str).unique().tolist()),
        ordered=True,
    )
    stats = {
        "vdjdb_runs_in_metrics_table": int(len(vdjdb_runs)),
        "vdjdb_runs_with_f1": int(len(metric_df)),
        "vdjdb_f1_rows_from_selected_runs": int(len(selected_runs_df)),
        "vdjdb_f1_rows_from_compact_metrics": int(len(compact_metrics_df)),
        "vdjdb_f1_rows_recomputed_from_clusters": int(len(recomputed_df)),
        "vdjdb_missing_cluster_files": int(missing_cluster_files),
        "vdjdb_unreadable_runs": int(unreadable_runs),
    }
    return metric_df.sort_values(["method", "epitope", "run_id"]).reset_index(drop=True), stats


def merge_vdjdb_quality_columns(
    run_metrics_df: pd.DataFrame,
    vdjdb_metric_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = run_metrics_df.copy()
    vdjdb_mask = out["dataset_mode"].astype(str).eq("vdjdb")
    if "run_id" not in vdjdb_metric_df.columns:
        raise KeyError("VDJdb metric table must contain run_id before merging F1 columns.")

    metric_columns = [column for column in VDJDB_QUALITY_COLUMNS if column in vdjdb_metric_df.columns]
    metric_lookup = (
        vdjdb_metric_df.loc[:, ["run_id", *metric_columns]]
        .drop_duplicates(subset=["run_id"], keep="first")
        .set_index("run_id")
    )

    for column in metric_columns:
        out.loc[vdjdb_mask, column] = out.loc[vdjdb_mask, "run_id"].astype(str).map(metric_lookup[column])

    stats = {
        "vdjdb_rows_after_quality_merge": int(vdjdb_mask.sum()),
        "vdjdb_rows_with_f1_in_run_metrics": int(out.loc[vdjdb_mask, "f1"].notna().sum()) if "f1" in out.columns else 0,
    }
    return out, stats


def summarize_by_method_and_epitope(metric_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method, epitope), frame in metric_df.groupby(["method", "epitope"], observed=True, dropna=False):
        row: dict[str, object] = {
            "method": str(method),
            "epitope": str(epitope),
            "n_runs": int(len(frame)),
        }
        for metric in HEATMAP_METRICS:
            values = pd.to_numeric(frame[metric], errors="coerce").dropna().astype(float)
            row[f"{metric}__count"] = int(values.size)
            row[f"{metric}__mean"] = float(values.mean()) if values.size else np.nan
            row[f"{metric}__median"] = float(values.median()) if values.size else np.nan
            row[f"{metric}__q25"] = float(values.quantile(0.25)) if values.size else np.nan
            row[f"{metric}__q75"] = float(values.quantile(0.75)) if values.size else np.nan
            row[f"{metric}__iqr"] = (
                float(values.quantile(0.75) - values.quantile(0.25)) if values.size else np.nan
            )
            row[f"{metric}__min"] = float(values.min()) if values.size else np.nan
            row[f"{metric}__max"] = float(values.max()) if values.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "epitope"]).reset_index(drop=True)


def ordered_methods(summary_df: pd.DataFrame) -> list[str]:
    present = summary_df["method"].astype(str).unique().tolist()
    ordered = [method for method in METHOD_ORDER if method in present]
    leftovers = sorted(method for method in present if method not in ordered)
    return ordered + leftovers


def write_svg(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#4c4c4c")


def plot_heatmap(summary_df: pd.DataFrame, metric: str, statistic: str, output_stem: Path) -> Path:
    value_column = f"{metric}__{statistic}"
    methods = ordered_methods(summary_df)
    epitopes = sorted(summary_df["epitope"].astype(str).unique().tolist())
    heatmap_df = (
        summary_df.pivot(index="method", columns="epitope", values=value_column)
        .reindex(index=methods, columns=epitopes)
    )
    finite_values = heatmap_df.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]

    if metric == "f1":
        color_fn = lambda value: sequential_color(  # noqa: E731
            value,
            vmin=0.0,
            vmax=1.0,
            low=(247, 252, 240),
            high=(8, 64, 129),
        )
        legend_min, legend_mid, legend_max = "0.00", "0.50", "1.00"
    elif statistic == "median":
        limit = float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
        color_fn = lambda value: diverging_color(  # noqa: E731
            value,
            limit=limit,
            negative=(49, 54, 149),
            neutral=(247, 247, 247),
            positive=(165, 0, 38),
        )
        legend_min, legend_mid, legend_max = f"{-limit:.2f}", "0.00", f"{limit:.2f}"
    else:
        vmax = float(np.max(finite_values)) if finite_values.size else 1.0
        color_fn = lambda value: sequential_color(  # noqa: E731
            value,
            vmin=0.0,
            vmax=vmax,
            low=(255, 245, 235),
            high=(127, 0, 0),
        )
        legend_min, legend_mid, legend_max = "0.00", f"{(vmax / 2.0):.2f}", f"{vmax:.2f}"

    cell_width = 148
    cell_height = 58
    left_margin = 210
    top_margin = 100
    right_margin = 36
    bottom_margin = 110
    width = left_margin + (cell_width * len(epitopes)) + right_margin
    height = top_margin + (cell_height * len(methods)) + bottom_margin

    svg_lines = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" '
            'viewBox="0 0 {0} {1}" font-family="Arial, Helvetica, sans-serif">'
        ).format(width, height),
        '<rect x="0" y="0" width="{0}" height="{1}" fill="#ffffff" />'.format(width, height),
        '<text x="{0}" y="38" font-size="24" font-weight="bold" fill="#111111">{1}</text>'.format(
            width / 2,
            escape("{0} {1} by method and epitope".format(metric, statistic)),
        ),
        '<text x="{0}" y="68" font-size="13" fill="#555555">{1}</text>'.format(
            width / 2,
            escape("Cell labels show aggregated values across all runs for the method/epitope pair."),
        ),
        '<g text-anchor="middle" font-size="13" font-weight="bold" fill="#222222">',
    ]

    for column_index, epitope in enumerate(epitopes):
        x = left_margin + (column_index * cell_width) + (cell_width / 2)
        svg_lines.append('<text x="{0}" y="{1}">{2}</text>'.format(x, top_margin - 18, escape(epitope)))
    svg_lines.append("</g>")

    svg_lines.append('<g text-anchor="end" font-size="13" font-weight="bold" fill="#222222">')
    for row_index, method in enumerate(methods):
        y = top_margin + (row_index * cell_height) + (cell_height / 2) + 5
        svg_lines.append('<text x="{0}" y="{1}">{2}</text>'.format(left_margin - 16, y, escape(method)))
    svg_lines.append("</g>")

    for row_index, method in enumerate(methods):
        for column_index, epitope in enumerate(epitopes):
            value = heatmap_df.loc[method, epitope]
            fill_rgb = color_fn(value)
            fill_hex = rgb_to_hex(fill_rgb)
            text_fill = "#111111" if luminance(fill_rgb) >= 150 else "#ffffff"
            x = left_margin + (column_index * cell_width)
            y = top_margin + (row_index * cell_height)
            label = "NA" if not np.isfinite(value) else f"{float(value):.2f}"
            svg_lines.extend(
                [
                    '<rect x="{0}" y="{1}" width="{2}" height="{3}" fill="{4}" stroke="#ffffff" stroke-width="1.5" />'.format(
                        x,
                        y,
                        cell_width,
                        cell_height,
                        fill_hex,
                    ),
                    '<text x="{0}" y="{1}" text-anchor="middle" font-size="16" font-weight="bold" fill="{2}">{3}</text>'.format(
                        x + (cell_width / 2),
                        y + (cell_height / 2) + 6,
                        text_fill,
                        label,
                    ),
                ]
            )

    legend_x = left_margin
    legend_y = height - 56
    legend_width = max(180, cell_width * max(1, len(epitopes) // 2))
    legend_steps = 32
    for step in range(legend_steps):
        x = legend_x + (legend_width * step / legend_steps)
        if metric == "f1":
            value = step / (legend_steps - 1)
        elif statistic == "median":
            limit = float(legend_max) if legend_max != "0.00" else 1.0
            value = -limit + (2 * limit * step / (legend_steps - 1))
        else:
            vmax = float(legend_max) if legend_max != "0.00" else 1.0
            value = vmax * step / (legend_steps - 1)
        fill_hex = rgb_to_hex(color_fn(value))
        svg_lines.append(
            '<rect x="{0}" y="{1}" width="{2}" height="16" fill="{3}" stroke="none" />'.format(
                x,
                legend_y,
                legend_width / legend_steps,
                fill_hex,
            )
        )

    svg_lines.extend(
        [
            '<rect x="{0}" y="{1}" width="{2}" height="16" fill="none" stroke="#444444" stroke-width="1" />'.format(
                legend_x,
                legend_y,
                legend_width,
            ),
            '<text x="{0}" y="{1}" font-size="12" fill="#333333" text-anchor="start">{2}</text>'.format(
                legend_x,
                legend_y + 34,
                legend_min,
            ),
            '<text x="{0}" y="{1}" font-size="12" fill="#333333" text-anchor="middle">{2}</text>'.format(
                legend_x + (legend_width / 2),
                legend_y + 34,
                legend_mid,
            ),
            '<text x="{0}" y="{1}" font-size="12" fill="#333333" text-anchor="end">{2}</text>'.format(
                legend_x + legend_width,
                legend_y + 34,
                legend_max,
            ),
            "</svg>",
        ]
    )

    output_path = output_stem.with_suffix(".svg")
    write_svg(output_path, "\n".join(svg_lines))
    return output_path


def render_parameter_heatmap_svg(
    heatmap_df: pd.DataFrame,
    *,
    title: str,
    subtitle: str,
    metric: str,
    output_path: Path,
) -> None:
    finite_values = heatmap_df.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if metric == "f1":
        color_fn = lambda value: sequential_color(  # noqa: E731
            value,
            vmin=0.0,
            vmax=1.0,
            low=(247, 252, 240),
            high=(8, 64, 129),
        )
        legend_min, legend_mid, legend_max = "0.00", "0.50", "1.00"
    else:
        vmax = float(np.max(finite_values)) if finite_values.size else 1.0
        color_fn = lambda value: sequential_color(  # noqa: E731
            value,
            vmin=0.0,
            vmax=vmax,
            low=(255, 245, 235),
            high=(127, 0, 0),
        )
        legend_min, legend_mid, legend_max = "0.00", f"{(vmax / 2.0):.2f}", f"{vmax:.2f}"

    rows = heatmap_df.index.astype(str).tolist()
    columns = heatmap_df.columns.astype(str).tolist()
    cell_width = 168
    cell_height = 30
    left_margin = 330
    top_margin = 92
    right_margin = 36
    bottom_margin = 84
    width = left_margin + (cell_width * len(columns)) + right_margin
    height = top_margin + (cell_height * len(rows)) + bottom_margin

    svg_lines = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" '
            'viewBox="0 0 {0} {1}" font-family="Arial, Helvetica, sans-serif">'
        ).format(width, height),
        '<rect x="0" y="0" width="{0}" height="{1}" fill="#ffffff" />'.format(width, height),
        '<text x="{0}" y="34" font-size="22" font-weight="bold" fill="#111111">{1}</text>'.format(
            width / 2,
            escape(title),
        ),
        '<text x="{0}" y="58" font-size="12" fill="#555555">{1}</text>'.format(width / 2, escape(subtitle)),
        '<g text-anchor="middle" font-size="12" font-weight="bold" fill="#222222">',
    ]
    for column_index, column_name in enumerate(columns):
        x = left_margin + (column_index * cell_width) + (cell_width / 2)
        svg_lines.append('<text x="{0}" y="{1}">{2}</text>'.format(x, top_margin - 18, escape(column_name)))
    svg_lines.append("</g>")

    svg_lines.append('<g text-anchor="end" font-size="11" fill="#222222">')
    for row_index, row_name in enumerate(rows):
        y = top_margin + (row_index * cell_height) + (cell_height / 2) + 4
        svg_lines.append('<text x="{0}" y="{1}">{2}</text>'.format(left_margin - 10, y, escape(row_name)))
    svg_lines.append("</g>")

    for row_index, row_name in enumerate(rows):
        for column_index, column_name in enumerate(columns):
            value = heatmap_df.loc[row_name, column_name]
            fill_rgb = color_fn(value)
            fill_hex = rgb_to_hex(fill_rgb)
            text_fill = "#111111" if luminance(fill_rgb) >= 150 else "#ffffff"
            x = left_margin + (column_index * cell_width)
            y = top_margin + (row_index * cell_height)
            label = "NA" if not np.isfinite(value) else f"{float(value):.2f}"
            svg_lines.extend(
                [
                    '<rect x="{0}" y="{1}" width="{2}" height="{3}" fill="{4}" stroke="#ffffff" stroke-width="1.0" />'.format(
                        x,
                        y,
                        cell_width,
                        cell_height,
                        fill_hex,
                    ),
                    '<text x="{0}" y="{1}" text-anchor="middle" font-size="12" font-weight="bold" fill="{2}">{3}</text>'.format(
                        x + (cell_width / 2),
                        y + (cell_height / 2) + 4,
                        text_fill,
                        label,
                    ),
                ]
            )

    legend_x = left_margin
    legend_y = height - 50
    legend_width = max(180, cell_width)
    legend_steps = 32
    for step in range(legend_steps):
        x = legend_x + (legend_width * step / legend_steps)
        if metric == "f1":
            value = step / (legend_steps - 1)
        else:
            vmax = float(legend_max) if legend_max != "0.00" else 1.0
            value = vmax * step / (legend_steps - 1)
        fill_hex = rgb_to_hex(color_fn(value))
        svg_lines.append(
            '<rect x="{0}" y="{1}" width="{2}" height="14" fill="{3}" stroke="none" />'.format(
                x,
                legend_y,
                legend_width / legend_steps,
                fill_hex,
            )
        )
    svg_lines.extend(
        [
            '<rect x="{0}" y="{1}" width="{2}" height="14" fill="none" stroke="#444444" stroke-width="1" />'.format(
                legend_x,
                legend_y,
                legend_width,
            ),
            '<text x="{0}" y="{1}" font-size="11" fill="#333333" text-anchor="start">{2}</text>'.format(
                legend_x,
                legend_y + 28,
                legend_min,
            ),
            '<text x="{0}" y="{1}" font-size="11" fill="#333333" text-anchor="middle">{2}</text>'.format(
                legend_x + (legend_width / 2),
                legend_y + 28,
                legend_mid,
            ),
            '<text x="{0}" y="{1}" font-size="11" fill="#333333" text-anchor="end">{2}</text>'.format(
                legend_x + legend_width,
                legend_y + 28,
                legend_max,
            ),
            "</svg>",
        ]
    )
    write_svg(output_path, "\n".join(svg_lines))


def build_parameter_heatmaps(
    frame: pd.DataFrame,
    *,
    value_column: str,
    output_dir: Path,
    file_prefix: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for algorithm in [method for method in METHOD_ORDER if method in set(frame["algorithm"].astype(str))]:
        method_df = frame.loc[frame["algorithm"].astype(str) == algorithm].copy()
        if method_df.empty:
            continue
        group_columns = ["parameter_label", "epitope"] + [column for column in method_df.columns if column.startswith("param_")]
        summary = (
            method_df.groupby(group_columns, dropna=False)[value_column]
            .agg(["median", "count"])
            .reset_index()
        )
        ordered_summary = sort_parameter_rows(summary)
        heatmap_df = ordered_summary.pivot(index="parameter_label", columns="epitope", values="median")
        render_parameter_heatmap_svg(
            heatmap_df,
            title="{0}: {1}".format(algorithm, value_column),
            subtitle="Rows are parameter sets; columns are epitopes; cell values are median over matching runs.",
            metric=value_column,
            output_path=output_dir / "{0}_{1}.svg".format(file_prefix, algorithm),
        )
        export_df = ordered_summary.loc[:, ["parameter_label", "epitope", "median", "count"] + [column for column in ordered_summary.columns if column.startswith("param_")]].copy()
        export_df.insert(0, "algorithm", algorithm)
        export_df.insert(2, "metric", value_column)
        export_df.to_csv(output_dir / "{0}_{1}.tsv".format(file_prefix, algorithm), sep="\t", index=False)
        rows.append(
            {
                "algorithm": algorithm,
                "metric": value_column,
                "parameter_sets": int(ordered_summary["parameter_label"].nunique()),
                "rows_with_metric": int(method_df[value_column].notna().sum()),
                "epitopes": ",".join(sorted(method_df["epitope"].astype(str).unique().tolist())),
                "output_svg": str(output_dir / "{0}_{1}.svg".format(file_prefix, algorithm)),
            }
        )
    return pd.DataFrame(rows)


def algorithm_primary_x_column(frame: pd.DataFrame) -> str:
    if "param_k_neighbors" in frame.columns and frame["param_k_neighbors"].notna().any():
        return "param_k_neighbors"
    for candidate in [column for column in frame.columns if column.startswith("param_")]:
        if frame[candidate].notna().any():
            return candidate
    raise ValueError("Could not determine a primary x-axis parameter column.")


def drop_redundant_parameter_columns(frame: pd.DataFrame, x_column: str, parameter_columns: list[str]) -> list[str]:
    filtered: list[str] = []
    for column in parameter_columns:
        if column == x_column or column not in frame.columns:
            continue
        if frame[column].dropna().empty:
            continue
        if frame[column].nunique(dropna=True) <= 1:
            continue
        if (
            column == "param_eps_k_neighbors"
            and x_column == "param_k_neighbors"
            and frame[[column, x_column]].dropna().shape[0] > 0
            and (frame[[column, x_column]].dropna()[column] == frame[[column, x_column]].dropna()[x_column]).all()
        ):
            continue
        filtered.append(column)
    return filtered


def choose_facet_columns(frame: pd.DataFrame, algorithm: str) -> list[str]:
    preferred = {
        "vdbscan": ["param_eps_estimation_based_on", "param_vdbscan_sym_rule"],
        "vdbscan_leiden": ["param_eps_estimation_based_on", "param_vdbscan_sym_rule"],
    }.get(algorithm, [])
    facet_columns = [column for column in preferred if column in frame.columns and frame[column].nunique(dropna=True) > 1]
    return facet_columns


def format_parameter_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def sortable_parameter_value(value: object) -> tuple[int, object]:
    if pd.isna(value):
        return (2, "")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (0, float(value))
    text = str(value)
    try:
        return (0, float(text))
    except Exception:
        return (1, text)


def add_configuration_labels(frame: pd.DataFrame, *, y_columns: list[str], facet_columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    label_columns = facet_columns + y_columns
    if not y_columns:
        out["config_label"] = "all"
        out["config_sort_key"] = [tuple()] * len(out)
        return out

    def _label(row: pd.Series) -> str:
        parts: list[str] = []
        for column in y_columns:
            key = column.replace("param_", "")
            short = PARAMETER_SHORT_LABELS.get(key, key)
            parts.append("{0}={1}".format(short, format_parameter_value(row[column])))
        return " | ".join(parts)

    def _sort_key(row: pd.Series) -> tuple[tuple[int, object], ...]:
        return tuple(sortable_parameter_value(row[column]) for column in label_columns)

    out["config_label"] = out.apply(_label, axis=1)
    out["config_sort_key"] = out.apply(_sort_key, axis=1)
    return out


def plot_algorithm_metric_landscape(
    frame: pd.DataFrame,
    *,
    algorithm: str,
    metric: str,
    output_stem: Path,
) -> dict[str, object] | None:
    subset = frame.loc[frame["algorithm"].astype(str) == algorithm].copy()
    subset = subset.loc[subset[metric].notna()].copy()
    if subset.empty:
        return None

    x_column = algorithm_primary_x_column(subset)
    facet_columns = choose_facet_columns(subset, algorithm)
    ordered_parameter_columns = ["param_{0}".format(name) for name in PARAMETER_DISPLAY_ORDER.get(algorithm, [])]
    y_columns = drop_redundant_parameter_columns(
        subset,
        x_column=x_column,
        parameter_columns=[column for column in ordered_parameter_columns if column not in facet_columns],
    )
    labeled = add_configuration_labels(subset, y_columns=y_columns, facet_columns=facet_columns)

    epitopes = sorted(labeled["epitope"].astype(str).unique().tolist())
    if facet_columns:
        facet_values = (
            labeled.loc[:, facet_columns]
            .drop_duplicates()
            .sort_values(facet_columns)
            .itertuples(index=False, name=None)
        )
        facet_values = list(facet_values)
    else:
        facet_values = [tuple()]

    n_rows = len(epitopes)
    n_cols = len(facet_values)
    max_config_count = max(
        labeled.loc[labeled["epitope"].astype(str) == epitope, "config_label"].nunique()
        for epitope in epitopes
    )
    max_x_count = max(
        labeled.loc[labeled["epitope"].astype(str) == epitope, x_column].dropna().nunique()
        for epitope in epitopes
    )
    panel_width = max(5.6, 1.15 * max_x_count + 2.8)
    panel_height = max(4.2, 0.46 * max_config_count + 1.8)
    annot_size = 9 if max_config_count >= 12 else 10
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_width * n_cols, panel_height * n_rows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    cmap = "YlGnBu" if metric == "f1" else "YlOrRd"
    vmin = 0.0
    vmax = 1.0 if metric == "f1" else float(labeled[metric].max())
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    total_panels = 0
    for row_index, epitope in enumerate(epitopes):
        for col_index, facet_value in enumerate(facet_values):
            ax = axes[row_index][col_index]
            panel_df = labeled.loc[labeled["epitope"].astype(str) == epitope].copy()
            title_suffix = ""
            if facet_columns:
                for column, value in zip(facet_columns, facet_value):
                    panel_df = panel_df.loc[panel_df[column].astype(str) == str(value)].copy()
                title_suffix = ", ".join(
                    "{0}={1}".format(
                        PARAMETER_SHORT_LABELS.get(column.replace("param_", ""), column.replace("param_", "")),
                        value,
                    )
                    for column, value in zip(facet_columns, facet_value)
                )
            if panel_df.empty:
                ax.axis("off")
                continue

            order_df = (
                panel_df.loc[:, ["config_label", "config_sort_key"]]
                .drop_duplicates()
                .sort_values("config_sort_key")
            )
            row_order = order_df["config_label"].tolist()
            x_values = sorted(panel_df[x_column].dropna().unique().tolist())
            heatmap_df = (
                panel_df.groupby(["config_label", x_column], observed=True)[metric]
                .median()
                .unstack(x_column)
                .reindex(index=row_order, columns=x_values)
            )
            sns.heatmap(
                heatmap_df,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0.5,
                annot_kws={"size": annot_size},
                cbar=col_index == n_cols - 1,
                ax=ax,
            )
            x_label = x_column.replace("param_", "")
            ax.set_xlabel(x_label)
            ax.set_ylabel(epitope if col_index == 0 else "")
            title = epitope if not title_suffix else "{0}\n{1}".format(epitope, title_suffix)
            ax.set_title(title)
            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)
            total_panels += 1

    fig.suptitle("{0}: {1} over parameter grid".format(algorithm, metric), y=1.02)
    fig.tight_layout()
    save_figure(fig, output_stem)
    return {
        "algorithm": algorithm,
        "metric": metric,
        "parameter_sets": int(labeled["config_label"].nunique()),
        "rows_with_metric": int(len(labeled)),
        "epitopes": ",".join(epitopes),
        "facet_count": int(len(facet_values)),
        "output_png": str(output_stem.with_suffix(".png")),
        "output_pdf": str(output_stem.with_suffix(".pdf")),
    }


def plot_epitope_f1_vs_dense(
    metric_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> pd.DataFrame:
    required_columns = {"epitope", "method", "f1", "redcea_dense_score"}
    if not required_columns.issubset(metric_df.columns):
        return pd.DataFrame()

    plot_df = metric_df.loc[
        metric_df["f1"].notna() & metric_df["redcea_dense_score"].notna(),
        ["run_id", "method", "epitope", "f1", "redcea_dense_score"],
    ].copy()
    if plot_df.empty:
        return pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for epitope in sorted(plot_df["epitope"].astype(str).unique().tolist()):
        epitope_df = plot_df.loc[plot_df["epitope"].astype(str) == epitope].copy()
        if epitope_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(9.4, 7.2))
        present_methods = epitope_df["method"].astype(str).unique().tolist()
        method_order = [method for method in METHOD_ORDER if method in present_methods]
        method_order.extend(sorted(method for method in present_methods if method not in method_order))

        for method in method_order:
            method_df = epitope_df.loc[epitope_df["method"].astype(str) == method].copy()
            if method_df.empty:
                continue
            ax.scatter(
                method_df["redcea_dense_score"],
                method_df["f1"],
                s=60,
                alpha=0.78,
                linewidths=0.7,
                edgecolors="white",
                color=method_color(method),
                label=method,
            )

        pearson_r = np.nan
        x = epitope_df["redcea_dense_score"].to_numpy(dtype=float)
        y = epitope_df["f1"].to_numpy(dtype=float)
        if len(epitope_df) >= 2 and np.unique(x).size >= 2 and np.unique(y).size >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
            ax.plot(
                x_line,
                (slope * x_line) + intercept,
                color="#202020",
                linewidth=2.0,
                linestyle="--",
                label="linear trend",
            )
            pearson_r = float(np.corrcoef(x, y)[0, 1])

        dense_min = float(epitope_df["redcea_dense_score"].min())
        dense_max = float(epitope_df["redcea_dense_score"].max())
        ax.set_xlim(max(0.0, dense_min - 0.02), min(1.0, dense_max + 0.02))
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlabel("redcea_dense_score")
        ax.set_ylabel("f1")
        ax.set_title("{0}: f1 vs redcea dense score".format(epitope))
        ax.grid(True, alpha=0.18, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="best", fontsize=9, frameon=True)

        annotation = "n={0}".format(len(epitope_df))
        if np.isfinite(pearson_r):
            annotation += " | pearson r={0:.2f}".format(pearson_r)
        ax.text(
            0.02,
            0.98,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.3"},
        )

        output_stem = output_dir / "f1_vs_redcea_dense_{0}".format(epitope.lower())
        fig.tight_layout()
        save_figure(fig, output_stem)
        rows.append(
            {
                "epitope": epitope,
                "n_points": int(len(epitope_df)),
                "n_methods": int(epitope_df["method"].nunique()),
                "pearson_r": pearson_r,
                "f1_min": float(epitope_df["f1"].min()),
                "f1_max": float(epitope_df["f1"].max()),
                "dense_min": dense_min,
                "dense_max": dense_max,
                "output_png": str(output_stem.with_suffix(".png")),
                "output_pdf": str(output_stem.with_suffix(".pdf")),
            }
        )
    return pd.DataFrame(rows)


def plot_overall_f1_vs_dense(
    metric_df: pd.DataFrame,
    *,
    output_stem: Path,
) -> pd.DataFrame:
    required_columns = {"method", "f1", "redcea_dense_score"}
    if not required_columns.issubset(metric_df.columns):
        return pd.DataFrame()

    plot_df = metric_df.loc[
        metric_df["f1"].notna() & metric_df["redcea_dense_score"].notna(),
        ["run_id", "method", "epitope", "f1", "redcea_dense_score"],
    ].copy()
    if plot_df.empty:
        return pd.DataFrame()

    fig, ax = plt.subplots(figsize=(10.2, 7.8))
    present_methods = plot_df["method"].astype(str).unique().tolist()
    method_order = [method for method in METHOD_ORDER if method in present_methods]
    method_order.extend(sorted(method for method in present_methods if method not in method_order))

    for method in method_order:
        method_df = plot_df.loc[plot_df["method"].astype(str) == method].copy()
        if method_df.empty:
            continue
        ax.scatter(
            method_df["redcea_dense_score"],
            method_df["f1"],
            s=54,
            alpha=0.72,
            linewidths=0.65,
            edgecolors="white",
            color=method_color(method),
            label=method,
        )

    pearson_r = np.nan
    x = plot_df["redcea_dense_score"].to_numpy(dtype=float)
    y = plot_df["f1"].to_numpy(dtype=float)
    if len(plot_df) >= 2 and np.unique(x).size >= 2 and np.unique(y).size >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 300)
        ax.plot(
            x_line,
            (slope * x_line) + intercept,
            color="#202020",
            linewidth=2.2,
            linestyle="--",
            label="linear trend",
        )
        pearson_r = float(np.corrcoef(x, y)[0, 1])

    dense_min = float(plot_df["redcea_dense_score"].min())
    dense_max = float(plot_df["redcea_dense_score"].max())
    ax.set_xlim(max(0.0, dense_min - 0.02), min(1.0, dense_max + 0.02))
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("redcea_dense_score")
    ax.set_ylabel("f1")
    ax.set_title("All epitopes: f1 vs redcea dense score")
    ax.grid(True, alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=9, frameon=True)

    annotation = "n={0} | epitopes={1}".format(len(plot_df), plot_df["epitope"].nunique())
    if np.isfinite(pearson_r):
        annotation += " | pearson r={0:.2f}".format(pearson_r)
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.3"},
    )

    fig.tight_layout()
    save_figure(fig, output_stem)
    return pd.DataFrame(
        [
            {
                "scope": "all_epitopes",
                "n_points": int(len(plot_df)),
                "n_methods": int(plot_df["method"].nunique()),
                "n_epitopes": int(plot_df["epitope"].nunique()) if "epitope" in plot_df.columns else 0,
                "pearson_r": pearson_r,
                "f1_min": float(plot_df["f1"].min()),
                "f1_max": float(plot_df["f1"].max()),
                "dense_min": dense_min,
                "dense_max": dense_max,
                "output_png": str(output_stem.with_suffix(".png")),
                "output_pdf": str(output_stem.with_suffix(".pdf")),
            }
        ]
    )


def write_summary(
    summary_path: Path,
    *,
    merge_stats: dict[str, int],
    summary_df: pd.DataFrame,
    parameter_heatmap_manifest_df: pd.DataFrame,
    landscape_manifest_df: pd.DataFrame,
    epitope_scatter_manifest_df: pd.DataFrame,
    overall_scatter_manifest_df: pd.DataFrame,
) -> None:
    lines = [
        "# VDJdb dense-score merge and heatmap summary",
        "",
        "## Merge audit",
        "",
    ]
    for key, value in merge_stats.items():
        lines.append("- {0}: {1}".format(key, value))

    top_f1 = summary_df.loc[:, ["method", "epitope", "f1__median"]].sort_values("f1__median", ascending=False).head(10)
    top_dense = (
        summary_df.loc[:, ["method", "epitope", "redcea_dense_score__median"]]
        .sort_values("redcea_dense_score__median", ascending=False)
        .head(10)
    )

    lines.extend(
        [
            "",
            "## Parameter heatmaps",
            "",
            "```text",
            parameter_heatmap_manifest_df.to_string(index=False) if len(parameter_heatmap_manifest_df) else "No parameter heatmaps generated.",
            "```",
            "",
            "## PNG landscapes",
            "",
            "```text",
            landscape_manifest_df.to_string(index=False) if len(landscape_manifest_df) else "No PNG landscapes generated.",
            "```",
            "",
            "## Epitope F1 vs dense plots",
            "",
            "```text",
            epitope_scatter_manifest_df.to_string(index=False) if len(epitope_scatter_manifest_df) else "No epitope scatter plots generated.",
            "```",
            "",
            "## Overall F1 vs dense plot",
            "",
            "```text",
            overall_scatter_manifest_df.to_string(index=False) if len(overall_scatter_manifest_df) else "No overall scatter plot generated.",
            "```",
            "",
            "## Top median F1 cells",
            "",
            "```text",
            top_f1.to_string(index=False),
            "```",
            "",
            "## Top median redcea_dense_score cells",
            "",
            "```text",
            top_dense.to_string(index=False),
            "```",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if hasattr(sns, "set_theme"):
        sns.set_theme(style="whitegrid", context="talk")
    else:
        sns.set_style("whitegrid")

    args = parse_args()

    base_df = load_table(args.base_run_metrics)
    new_df = load_table(args.new_run_metrics)

    merged_run_metrics_df, merge_stats = merge_dense_score_columns(base_df, new_df)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    vdjdb_metric_df, vdjdb_stats = build_vdjdb_metric_frame(
        merged_run_metrics_df,
        vdjdb_selected_runs_path=args.vdjdb_selected_runs,
        vdjdb_clustering_metrics_path=args.vdjdb_clustering_metrics,
        redcea_runs_dir=args.redcea_runs_dir,
        truth_path=args.truth_table,
        padj_threshold=args.padj_threshold,
    )
    merge_stats.update(vdjdb_stats)
    merged_run_metrics_df, quality_merge_stats = merge_vdjdb_quality_columns(merged_run_metrics_df, vdjdb_metric_df)
    merge_stats.update(quality_merge_stats)
    save_table(merged_run_metrics_df, args.output_run_metrics)

    summary_df = summarize_by_method_and_epitope(vdjdb_metric_df)
    repaired_metadata_df = load_repaired_metadata(args.vdjdb_repaired_metadata)
    vdjdb_runs_with_params_df = enrich_vdjdb_run_metrics_with_parameters(merged_run_metrics_df, repaired_metadata_df)
    dense_param_df, f1_param_df = build_parameter_heatmap_input(vdjdb_runs_with_params_df, vdjdb_metric_df)

    vdjdb_metric_df.to_csv(output_dir / "vdjdb_f1_and_dense_score_runs.tsv", sep="\t", index=False)
    summary_df.to_csv(output_dir / "vdjdb_method_epitope_metric_summary.tsv", sep="\t", index=False)
    dense_param_df.to_csv(output_dir / "vdjdb_dense_score_parameter_runs.tsv", sep="\t", index=False)
    f1_param_df.to_csv(output_dir / "vdjdb_f1_parameter_runs.tsv", sep="\t", index=False)

    for metric in HEATMAP_METRICS:
        plot_heatmap(summary_df, metric, "median", output_dir / f"{metric}_median_heatmap")
        plot_heatmap(summary_df, metric, "iqr", output_dir / f"{metric}_iqr_heatmap")

    parameter_heatmaps_dir = output_dir / "per_algorithm_parameter_heatmaps"
    parameter_heatmaps_dir.mkdir(parents=True, exist_ok=True)
    dense_manifest_df = build_parameter_heatmaps(
        dense_param_df,
        value_column="redcea_dense_score",
        output_dir=parameter_heatmaps_dir,
        file_prefix="redcea_dense_score",
    )
    f1_manifest_df = build_parameter_heatmaps(
        f1_param_df.loc[f1_param_df["f1"].notna()].copy(),
        value_column="f1",
        output_dir=parameter_heatmaps_dir,
        file_prefix="f1",
    )
    parameter_heatmap_manifest_df = pd.concat([dense_manifest_df, f1_manifest_df], ignore_index=True)
    parameter_heatmap_manifest_df.to_csv(output_dir / "parameter_heatmap_manifest.tsv", sep="\t", index=False)

    landscape_dir = output_dir / "per_algorithm_landscapes"
    landscape_dir.mkdir(parents=True, exist_ok=True)
    landscape_rows: list[dict[str, object]] = []
    for algorithm in sorted(dense_param_df["algorithm"].astype(str).unique().tolist(), key=lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 999):
        dense_result = plot_algorithm_metric_landscape(
            dense_param_df,
            algorithm=algorithm,
            metric="redcea_dense_score",
            output_stem=landscape_dir / "redcea_dense_score_{0}".format(algorithm),
        )
        if dense_result is not None:
            landscape_rows.append(dense_result)
    for algorithm in sorted(f1_param_df["algorithm"].astype(str).unique().tolist(), key=lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 999):
        f1_result = plot_algorithm_metric_landscape(
            f1_param_df,
            algorithm=algorithm,
            metric="f1",
            output_stem=landscape_dir / "f1_{0}".format(algorithm),
        )
        if f1_result is not None:
            landscape_rows.append(f1_result)
    landscape_manifest_df = pd.DataFrame(landscape_rows)
    landscape_manifest_df.to_csv(output_dir / "landscape_manifest.tsv", sep="\t", index=False)

    epitope_scatter_dir = output_dir / "per_epitope_f1_vs_dense"
    epitope_scatter_manifest_df = plot_epitope_f1_vs_dense(vdjdb_metric_df, output_dir=epitope_scatter_dir)
    epitope_scatter_manifest_df.to_csv(output_dir / "epitope_f1_vs_dense_manifest.tsv", sep="\t", index=False)

    overall_scatter_manifest_df = plot_overall_f1_vs_dense(
        vdjdb_metric_df,
        output_stem=output_dir / "f1_vs_redcea_dense_all_epitopes",
    )
    overall_scatter_manifest_df.to_csv(output_dir / "overall_f1_vs_dense_manifest.tsv", sep="\t", index=False)

    write_summary(
        output_dir / "summary.md",
        merge_stats=merge_stats,
        summary_df=summary_df,
        parameter_heatmap_manifest_df=parameter_heatmap_manifest_df,
        landscape_manifest_df=landscape_manifest_df,
        epitope_scatter_manifest_df=epitope_scatter_manifest_df,
        overall_scatter_manifest_df=overall_scatter_manifest_df,
    )

    print("Merged run metrics saved to: {0}".format(args.output_run_metrics))
    print("Per-run VDJdb metric table saved to: {0}".format(output_dir / "vdjdb_f1_and_dense_score_runs.tsv"))
    print("Method-by-epitope summary saved to: {0}".format(output_dir / "vdjdb_method_epitope_metric_summary.tsv"))
    print("Heatmaps saved to: {0}".format(output_dir))


if __name__ == "__main__":
    main()
