from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from redcea.auxiliary_cluster_metrics import (  # noqa: E402
    AUXILIARY_CLUSTER_METRIC_COLUMNS,
    append_auxiliary_cluster_metrics,
)
from scripts.run_additional_metrics_batch import (  # noqa: E402
    CLUSTERS_SUFFIX,
    SUMMARY_SUFFIX,
    _prepare_cluster_metric_inputs,
    build_run_level_table,
    find_run_prefix,
    infer_source_counts,
    load_first_npy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate RedCEA grid runs for selected rheum patients and export wide/long "
            "tables for downstream visualization."
        )
    )
    parser.add_argument(
        "--runs-root",
        required=True,
        help="Directory that contains one subdirectory per patient, e.g. runs/as_Abd_PB_F.",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        required=True,
        help="Patient directory names to include, e.g. as_Abd_PB_F as_Abr_PB_F.",
    )
    parser.add_argument(
        "--output-wide",
        required=True,
        help="Output TSV with one row per run.",
    )
    parser.add_argument(
        "--output-long",
        required=True,
        help="Output TSV in long format with metric_name / metric_value columns.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--skip-auxiliary-backfill",
        action="store_true",
        help=(
            "Do not attempt in-memory auxiliary metric backfill from *_tcremp_clusters.tsv "
            "and knn npy files."
        ),
    )
    parser.add_argument(
        "--metric-columns",
        nargs="*",
        default=None,
        help=(
            "Optional explicit metric columns for the long table. "
            "If omitted, all numeric run-level metrics are exported."
        ),
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_tagged_run_name(run_name: str) -> dict[str, object]:
    parsed: dict[str, object] = {"raw_run_name": run_name}
    parts = [part for part in run_name.split("__") if part]
    if not parts:
        return parsed

    first = parts[0]
    if first.startswith("algo_"):
        parsed["method"] = first.removeprefix("algo_")
    else:
        parsed["method"] = first

    token_map = {
        "kn": ("k_neighbors", int),
        "ekn": ("eps_k_neighbors", int),
        "ms": ("cluster_min_samples", int),
        "eps": ("eps_estimation_based_on", str),
        "sym": ("vdbscan_sym_rule", str),
        "lr": ("leiden_resolution", lambda value: float(value.replace("p", "."))),
        "pc": ("cluster_pc_components", int),
        "test": ("enrichment_test", str),
    }
    for token in parts[1:]:
        key, _, raw_value = token.partition("_")
        if not raw_value:
            continue
        mapped = token_map.get(key)
        if mapped is None:
            parsed[token] = raw_value
            continue
        output_key, caster = mapped
        try:
            parsed[output_key] = caster(raw_value)
        except Exception:
            parsed[output_key] = raw_value
    return parsed


def needs_auxiliary_backfill(summary_df: pd.DataFrame) -> bool:
    return any(column not in summary_df.columns for column in AUXILIARY_CLUSTER_METRIC_COLUMNS)


def load_summary_with_optional_backfill(run_dir: Path, *, skip_auxiliary_backfill: bool) -> pd.DataFrame | None:
    prefix = find_run_prefix(run_dir)
    if prefix is None:
        logging.warning("Skipping %s: summary file not found", run_dir)
        return None

    summary_path = run_dir / f"{prefix}{SUMMARY_SUFFIX}"
    if not summary_path.exists():
        logging.warning("Skipping %s: summary file missing", run_dir)
        return None

    summary_df = pd.read_csv(summary_path, sep="\t")
    if skip_auxiliary_backfill or (not needs_auxiliary_backfill(summary_df)):
        return summary_df

    clusters_path = run_dir / f"{prefix}{CLUSTERS_SUFFIX}"
    if not clusters_path.exists():
        logging.warning("Auxiliary backfill skipped for %s: clusters file missing", run_dir)
        return summary_df

    try:
        cluster_df = pd.read_csv(clusters_path, sep="\t")
        total_sample, total_background = infer_source_counts(cluster_df)
        sample_knn_indices = load_first_npy(run_dir, "knn_sample_sample__*.indices.npy")
        sample_knn_distances = load_first_npy(run_dir, "knn_sample_sample__*.distances.npy")
        background_knn_indices = load_first_npy(run_dir, "knn_bg_bg__*.indices.npy")
        background_knn_distances = load_first_npy(run_dir, "knn_bg_bg__*.distances.npy")
        return append_auxiliary_cluster_metrics(
            summary_df=summary_df,
            cluster_df=cluster_df,
            total_sample=total_sample,
            total_background=total_background,
            sample_knn_indices=sample_knn_indices,
            sample_knn_distances=sample_knn_distances,
            background_knn_indices=background_knn_indices,
            background_knn_distances=background_knn_distances,
        )
    except Exception as exc:
        logging.warning("Auxiliary backfill failed for %s: %s", run_dir, exc)
        return summary_df


def collect_cluster_level_table(
    *,
    runs_root: Path,
    patients: list[str],
    skip_auxiliary_backfill: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for patient in patients:
        patient_dir = runs_root / patient
        if not patient_dir.exists():
            logging.warning("Patient directory not found: %s", patient_dir)
            continue

        run_dirs = sorted(path for path in patient_dir.iterdir() if path.is_dir())
        if not run_dirs:
            logging.warning("No run directories found under %s", patient_dir)
            continue

        for run_dir in run_dirs:
            summary_df = load_summary_with_optional_backfill(
                run_dir,
                skip_auxiliary_backfill=skip_auxiliary_backfill,
            )
            if summary_df is None or summary_df.empty:
                continue

            summary_df = _prepare_cluster_metric_inputs(summary_df)
            summary_df.insert(0, "run_id", run_dir.name)
            summary_df.insert(1, "patient", patient)
            summary_df.insert(2, "run_dir", str(run_dir))

            parsed = parse_tagged_run_name(run_dir.name)
            for column, value in parsed.items():
                if column in summary_df.columns:
                    continue
                summary_df[column] = value
            frames.append(summary_df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_long_table(
    run_level_df: pd.DataFrame,
    *,
    metric_columns: list[str] | None,
) -> pd.DataFrame:
    id_columns = [
        column
        for column in [
            "patient",
            "run_id",
            "run_dir",
            "raw_run_name",
            "method",
            "k_neighbors",
            "eps_k_neighbors",
            "cluster_min_samples",
            "eps_estimation_based_on",
            "vdbscan_sym_rule",
            "leiden_resolution",
            "cluster_pc_components",
            "enrichment_test",
            "parameter_json",
            "status",
        ]
        if column in run_level_df.columns
    ]

    if metric_columns is None:
        metric_columns = [
            column
            for column in run_level_df.columns
            if column not in id_columns and pd.api.types.is_numeric_dtype(run_level_df[column])
        ]
    else:
        missing = [column for column in metric_columns if column not in run_level_df.columns]
        if missing:
            raise ValueError("Requested metric columns are missing: {0}".format(", ".join(missing)))

    long_df = run_level_df.melt(
        id_vars=id_columns,
        value_vars=metric_columns,
        var_name="metric_name",
        value_name="metric_value",
    )
    long_df["metric_value"] = pd.to_numeric(long_df["metric_value"], errors="coerce")
    long_df = long_df.loc[long_df["metric_value"].notna()].copy()
    return long_df.sort_values(["patient", "metric_name", "run_id"]).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    runs_root = Path(args.runs_root)
    output_wide = Path(args.output_wide)
    output_long = Path(args.output_long)

    cluster_level_df = collect_cluster_level_table(
        runs_root=runs_root,
        patients=args.patients,
        skip_auxiliary_backfill=args.skip_auxiliary_backfill,
    )
    if cluster_level_df.empty:
        raise SystemExit("No valid run summaries were found for the requested patients.")

    run_level_df = build_run_level_table(cluster_level_df)
    if run_level_df.empty:
        raise SystemExit("Run-level metrics table is empty.")

    preferred_order = [
        "patient",
        "run_id",
        "run_dir",
        "method",
        "k_neighbors",
        "eps_k_neighbors",
        "cluster_min_samples",
        "eps_estimation_based_on",
        "vdbscan_sym_rule",
        "leiden_resolution",
        "cluster_pc_components",
        "enrichment_test",
        "n_clusters_total",
        "n_clusters_enriched",
        "sample_mass_enriched",
        "sample_coverage_enriched",
        "lfc_sample_weighted",
        "lfc_mass_shift",
        "lfc_pos_mass",
        "lfc_neg_mass",
        "n_sig_pos",
        "n_sig_neg",
        "sample_sig_pos",
        "sample_sig_neg",
        "sample_pos_neg_ratio",
        "redcea_dense_score",
        "redcea_dense_score_soft",
        "redcea_dense_score_base",
    ]
    ordered_columns = [column for column in preferred_order if column in run_level_df.columns]
    ordered_columns.extend(column for column in run_level_df.columns if column not in ordered_columns)
    run_level_df = run_level_df.loc[:, ordered_columns].sort_values(
        [
            column
            for column in [
                "patient",
                "k_neighbors",
                "eps_k_neighbors",
                "cluster_min_samples",
                "eps_estimation_based_on",
                "vdbscan_sym_rule",
                "leiden_resolution",
                "run_id",
            ]
            if column in run_level_df.columns
        ]
    ).reset_index(drop=True)

    long_df = build_long_table(run_level_df, metric_columns=args.metric_columns)

    output_wide.parent.mkdir(parents=True, exist_ok=True)
    output_long.parent.mkdir(parents=True, exist_ok=True)
    run_level_df.to_csv(output_wide, sep="\t", index=False)
    long_df.to_csv(output_long, sep="\t", index=False)

    logging.info("Wrote wide table: %s", output_wide)
    logging.info("Wrote long table: %s", output_long)
    logging.info("Patients: %s", ", ".join(args.patients))
    logging.info("Runs aggregated: %d", run_level_df["run_id"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
