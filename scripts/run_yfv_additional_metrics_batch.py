from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run auxiliary cluster-metric backfill and aggregation only for YFV redcea runs."
    )
    parser.add_argument("--runs-root", default="results/redcea_runs")
    parser.add_argument(
        "--execution-manifest",
        default="results/run_metadata/clustering_execution_manifest.tsv",
        help="TSV manifest with run_id and parameter_json metadata.",
    )
    parser.add_argument("--cluster-output", default="results/metrics/yfv_cluster_auxiliary_metrics.tsv")
    parser.add_argument("--run-output", default="results/metrics/yfv_run_auxiliary_metrics.tsv")
    parser.add_argument("--log-path", default="results/metrics/run_yfv_additional_metrics_batch.log")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument(
        "--run-id-prefix",
        default="yfv_",
        help="Optional run_id prefix filter inside YFV runs. Defaults to `yfv_`.",
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


def main() -> int:
    args = parse_args()
    batch_script = Path(__file__).with_name("run_additional_metrics_batch.py")
    command = [
        sys.executable,
        str(batch_script),
        "--runs-root",
        str(args.runs_root),
        "--execution-manifest",
        str(args.execution_manifest),
        "--cluster-output",
        str(args.cluster_output),
        "--run-output",
        str(args.run_output),
        "--log-path",
        str(args.log_path),
        "--workers",
        str(args.workers),
        "--dataset-mode",
        "yfv",
    ]
    if args.run_id_prefix:
        command.extend(["--run-id-prefix", str(args.run_id_prefix)])
    if args.rewrite_existing:
        command.append("--rewrite-existing")
    if args.fail_fast:
        command.append("--fail-fast")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
