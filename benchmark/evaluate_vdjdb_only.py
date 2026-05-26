from __future__ import annotations

import argparse

from benchmark.workflows import run_vdjdb_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only VDJdb evaluation with an explicit metadata file.")
    parser.add_argument("--assignments-dir", default="results/clustering_assignments")
    parser.add_argument("--metadata-parts-dir", default="results/run_metadata/clustering_run_parts")
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--metrics-path", default="results/metrics/vdjdb_clustering_metrics.tsv")
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
    metrics_df = run_vdjdb_evaluation(
        assignments_dir=args.assignments_dir,
        metadata_parts_dir=args.metadata_parts_dir,
        metadata_path=args.metadata_path,
        metrics_path=args.metrics_path,
        fig2_stem=args.fig2_stem,
        fig3_stem=args.fig3_stem,
    )
    print("Finished VDJdb-only evaluation; rows={0}".format(len(metrics_df)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
