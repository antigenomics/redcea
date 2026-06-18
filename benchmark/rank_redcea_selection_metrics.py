from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_METRICS = [
    "redcea_dense_score",
    "redcea_dense_score_soft",
    "redcea_dense_score_base",
    "lfc_mass_shift",
    "sample_sig_pos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank RedCEA runs by dense-benchmark and longitudinal run-level selection metrics."
    )
    parser.add_argument("--input", default="results/metrics/redcea_all_run_metrics.tsv")
    parser.add_argument("--output", default="results/metrics/redcea_selection_metric_ranking.tsv")
    parser.add_argument("--dataset-mode", choices=["all", "vdjdb", "yfv"], default="all")
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def build_comparison_table(run_level_df: pd.DataFrame) -> pd.DataFrame:
    available_metrics = [column for column in DEFAULT_METRICS if column in run_level_df.columns]
    if not available_metrics:
        raise ValueError("Input table does not contain any of the expected selection metrics.")

    ordered = run_level_df.copy()
    for metric in available_metrics:
        ordered[f"rank__{metric}"] = ordered[metric].rank(method="min", ascending=False, na_option="bottom")

    sort_columns = available_metrics + ["run_id"]
    ascending = [False] * len(available_metrics) + [True]
    ordered = ordered.sort_values(sort_columns, ascending=ascending, na_position="last")

    leading_columns = [
        column
        for column in ["run_id", "dataset_mode", "dataset", "epitope", "sample_group", "method", "parameter_json"]
        if column in ordered.columns
    ]
    metric_columns: list[str] = []
    for metric in available_metrics:
        metric_columns.extend([metric, f"rank__{metric}"])
    diagnostic_columns = [
        column
        for column in [
            "coverage_rank_pct",
            "effective_size_penalty",
            "lfc_rank_pct",
            "sample_coverage_enriched",
            "sample_mass_enriched",
            "sample_sig_pos",
            "mean_size_sig_pos",
            "cluster_mass_pos_neg_ratio",
        ]
        if column in ordered.columns and column not in metric_columns
    ]
    return ordered.loc[:, leading_columns + metric_columns + diagnostic_columns]


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    run_level_df = pd.read_csv(input_path, sep="\t")
    if args.dataset_mode != "all" and "dataset_mode" in run_level_df.columns:
        run_level_df = run_level_df.loc[run_level_df["dataset_mode"].astype(str) == args.dataset_mode].copy()

    comparison_df = build_comparison_table(run_level_df)
    if args.top_n > 0:
        comparison_df = comparison_df.head(args.top_n)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, sep="\t", index=False)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
