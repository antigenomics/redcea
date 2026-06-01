from __future__ import annotations

import argparse

from benchmark.workflows import run_vdjdb_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan all VDJdb redcea_runs directories and compute precision/recall/F1."
    )
    parser.add_argument("--redcea-runs-dir", default="results/redcea_runs")
    parser.add_argument("--assignments-dir", default="results/clustering_assignments")
    parser.add_argument("--metrics-path", default="results/metrics/vdjdb_clustering_metrics.tsv")
    parser.add_argument("--tcrvdb-path", default=None)
    parser.add_argument("--padj-threshold", type=float, default=None)
    parser.add_argument(
        "--fig2-stem",
        default="figures/clustering_strategy/fig2_vdjdb_f1_precision_recall",
    )
    parser.add_argument(
        "--fig3-stem",
        default="figures/clustering_strategy/fig3_vdjdb_cluster_concentration",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kwargs = {
        "redcea_runs_dir": args.redcea_runs_dir,
        "assignments_dir": args.assignments_dir,
        "metrics_path": args.metrics_path,
        "fig2_stem": args.fig2_stem,
        "fig3_stem": args.fig3_stem,
    }
    if args.tcrvdb_path is not None:
        kwargs["tcrvdb_path"] = args.tcrvdb_path
    if args.padj_threshold is not None:
        kwargs["padj_threshold"] = args.padj_threshold
    metrics_df = run_vdjdb_evaluation(
        **kwargs,
    )
    print("Finished VDJdb-only evaluation; rows={0}".format(len(metrics_df)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
