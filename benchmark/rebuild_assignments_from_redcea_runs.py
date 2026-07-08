from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.import_archived_redcea_results import parse_run_identity
from benchmark.prepare_datasets import DEFAULT_TCRVDB_PADJ_THRESHOLD, build_vdjdb_truth_table, log_step
from benchmark.run_benchmark import standardize_redcea_assignments


def rebuild_assignments(
    *,
    metadata_path: Path,
    results_dir: Path,
    tcrvdb_path: Path,
    padj_threshold: float,
    dataset_mode: str,
) -> tuple[int, int]:
    metadata = pd.read_csv(metadata_path, sep="\t")
    if "dataset_mode" in metadata.columns:
        metadata = metadata.loc[metadata["dataset_mode"].astype(str) == dataset_mode].copy()
    if "status" in metadata.columns:
        metadata = metadata.loc[metadata["status"].astype(str) == "success"].copy()
    metadata = metadata.drop_duplicates(subset=["run_id"], keep="last").reset_index(drop=True)

    assignments_dir = results_dir / "clustering_assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)
    truth_table = build_vdjdb_truth_table(tcrvdb_path, padj_threshold=padj_threshold)

    written = 0
    missing = 0
    for row in metadata.itertuples(index=False):
        run_id = str(row.run_id)
        run_info = parse_run_identity(run_id)
        run_dir_value = getattr(row, "redcea_output_dir", "")
        run_dir = Path(str(run_dir_value)) if str(run_dir_value).strip() else (results_dir / "redcea_runs" / run_id)
        cluster_path = run_dir / f"{run_id}_tcremp_clusters.tsv"
        if not cluster_path.exists():
            log_step(f"Missing cluster table for run_id={run_id}: {cluster_path}")
            missing += 1
            continue

        log_step(f"Rebuilding assignment parquet for run_id={run_id} from {cluster_path}")
        cluster_df = pd.read_csv(cluster_path, sep="\t")
        assignments = standardize_redcea_assignments(
            cluster_df,
            run_id=run_id,
            dataset=str(getattr(row, "dataset", run_info["dataset"])),
            dataset_mode=str(getattr(row, "dataset_mode", run_info["dataset_mode"])),
            method=str(getattr(row, "method", run_info["method"])),
            parameter_json=str(getattr(row, "parameter_json", "{}")),
            epitope=run_info["epitope"],
            donor_id=run_info["donor_id"],
            truth_table=truth_table,
        )
        assignments.to_parquet(assignments_dir / f"{run_id}.parquet", index=False)
        written += 1

    log_step(
        "Finished rebuilding assignments from redcea_runs: dataset_mode={0}, written={1}, missing_cluster_tables={2}".format(
            dataset_mode,
            written,
            missing,
        )
    )
    return written, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild normalized assignment parquet files from redcea_runs outputs.")
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tcrvdb-path", default=str(ROOT_DIR / "data" / "01_05_2025_TCRvdb.csv"))
    parser.add_argument("--padj-threshold", type=float, default=DEFAULT_TCRVDB_PADJ_THRESHOLD)
    parser.add_argument("--dataset-mode", choices=["vdjdb", "yfv"], default="vdjdb")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rebuild_assignments(
        metadata_path=Path(args.metadata_path),
        results_dir=Path(args.results_dir),
        tcrvdb_path=Path(args.tcrvdb_path),
        padj_threshold=args.padj_threshold,
        dataset_mode=args.dataset_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
