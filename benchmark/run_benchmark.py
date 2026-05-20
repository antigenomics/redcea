from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.grids import get_enabled_methods, get_method_grid
from benchmark.runner import ClusteringBenchmarkRunner


def consolidate_run_metadata(
    *,
    metadata_parts_dir: str | Path = "results/run_metadata/clustering_run_parts",
    metadata_path: str | Path = "results/run_metadata/clustering_runs.tsv",
) -> pd.DataFrame:
    metadata_parts_dir = Path(metadata_parts_dir)
    part_files = sorted(metadata_parts_dir.glob("*.tsv"))
    if not part_files:
        return pd.DataFrame(
            columns=[
                "run_id",
                "dataset",
                "method",
                "parameter_json",
                "n_points",
                "n_clusters",
                "n_noise",
                "noise_fraction",
                "runtime_seconds",
                "status",
                "error_message",
            ]
        )
    frames = [pd.read_csv(path, sep="\t") for path in part_files]
    consolidated = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["dataset", "method", "run_id"])
        .drop_duplicates(subset=["run_id"], keep="last")
        .reset_index(drop=True)
    )
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated.to_csv(metadata_path, sep="\t", index=False)
    return consolidated


def build_grid_manifest(include_extended=False):
    rows = []
    for method in get_enabled_methods(include_extended=include_extended):
        for params in get_method_grid(method, include_extended=include_extended):
            rows.append(
                {
                    "method": method,
                    "parameter_json": json.dumps(params, sort_keys=True, separators=(",", ":")),
                }
            )
    manifest = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    manifest.insert(0, "grid_id", ["grid_{0:04d}".format(i) for i in range(len(manifest))])
    return manifest


def _load_processed_inputs(processed_dir):
    processed_dir = Path(processed_dir)
    required = [
        processed_dir / "vdjdb_glc.parquet",
        processed_dir / "vdjdb_ylq.parquet",
        processed_dir / "yfv_repertoires.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed benchmark inputs are missing. Expected files: {0}".format(", ".join(missing))
        )
    glc = pd.read_parquet(processed_dir / "vdjdb_glc.parquet")
    ylq = pd.read_parquet(processed_dir / "vdjdb_ylq.parquet")
    yfv = pd.read_parquet(processed_dir / "yfv_repertoires.parquet")
    return glc, ylq, yfv


def build_execution_manifest(processed_dir, include_extended=False):
    glc, ylq, yfv = _load_processed_inputs(processed_dir)
    grid_manifest = build_grid_manifest(include_extended=include_extended)
    rows = []

    vdjdb_datasets = [
        ("vdjdb_glc", "GLC", glc),
        ("vdjdb_ylq", "YLQ", ylq),
    ]
    for _, row in grid_manifest.iterrows():
        method = row["method"]
        for dataset_name, epitope, frame in vdjdb_datasets:
            del frame
            rows.append(
                {
                    "grid_id": row["grid_id"],
                    "run_id": "{0}_{1}_{2}".format(dataset_name, method, row["grid_id"]),
                    "dataset": dataset_name,
                    "dataset_mode": "vdjdb",
                    "epitope": epitope,
                    "donor_id": None,
                    "method": method,
                    "parameter_json": row["parameter_json"],
                    "status": "pending",
                }
            )

        for donor_id, donor_frame in yfv.groupby("donor_id"):
            del donor_frame
            rows.append(
                {
                    "grid_id": row["grid_id"],
                    "run_id": "yfv_{0}_{1}_{2}".format(donor_id, method, row["grid_id"]),
                    "dataset": "yfv_repertoires",
                    "dataset_mode": "yfv",
                    "epitope": None,
                    "donor_id": str(donor_id),
                    "method": method,
                    "parameter_json": row["parameter_json"],
                    "status": "pending",
                }
            )
    return pd.DataFrame(rows)


def _load_dataset_frame(processed_dir, dataset_mode, dataset, donor_id=None):
    glc, ylq, yfv = _load_processed_inputs(processed_dir)
    if dataset_mode == "vdjdb":
        if dataset == "vdjdb_glc":
            return glc
        if dataset == "vdjdb_ylq":
            return ylq
        raise KeyError("Unknown vdjdb dataset: {0}".format(dataset))
    if dataset_mode == "yfv":
        if donor_id is None:
            raise ValueError("donor_id is required for yfv execution rows")
        return yfv.loc[yfv["donor_id"].astype(str) == str(donor_id)].reset_index(drop=True)
    raise KeyError("Unknown dataset_mode: {0}".format(dataset_mode))


def execute_single_manifest_row(manifest_row, processed_dir, runner):
    params = json.loads(manifest_row["parameter_json"])
    frame = _load_dataset_frame(
        processed_dir,
        dataset_mode=manifest_row["dataset_mode"],
        dataset=manifest_row["dataset"],
        donor_id=manifest_row.get("donor_id"),
    )
    return runner.run(
        frame,
        dataset=manifest_row["dataset"],
        dataset_mode=manifest_row["dataset_mode"],
        method=manifest_row["method"],
        parameters=params,
        epitope=None if pd.isna(manifest_row.get("epitope")) else manifest_row.get("epitope"),
        donor_id=None if pd.isna(manifest_row.get("donor_id")) else str(manifest_row.get("donor_id")),
        run_id=manifest_row["run_id"],
    )


def execute_manifest(manifest, processed_dir, runner):
    executed_rows = []
    for _, row in manifest.iterrows():
        result = execute_single_manifest_row(row, processed_dir, runner)
        executed_rows.append(
            {
                "grid_id": row["grid_id"],
                "run_id": result.run_id,
                "dataset": row["dataset"],
                "dataset_mode": row["dataset_mode"],
                "epitope": row.get("epitope"),
                "donor_id": row.get("donor_id"),
                "method": row["method"],
                "parameter_json": row["parameter_json"],
                "status": result.metadata["status"],
            }
        )
    return pd.DataFrame(executed_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the RedCEA clustering proposal benchmark grid.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--manifest-path", default="results/run_metadata/clustering_grid_manifest.tsv")
    parser.add_argument("--vdjdb-manifest-path", default="results/run_metadata/clustering_manifest_vdjdb.tsv")
    parser.add_argument("--yfv-manifest-path", default="results/run_metadata/clustering_manifest_yfv.tsv")
    parser.add_argument("--execution-path", default="results/run_metadata/clustering_execution_manifest.tsv")
    parser.add_argument("--assignments-dir", default="results/clustering_assignments")
    parser.add_argument("--run-metadata-path", default="results/run_metadata/clustering_runs.tsv")
    parser.add_argument("--metadata-parts-dir", default="results/run_metadata/clustering_run_parts")
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--include-extended", action="store_true")
    parser.add_argument("--mode", choices=["manifest", "single", "all", "consolidate"], default="all")
    parser.add_argument("--single-manifest-path", default=None)
    parser.add_argument("--single-row-index", type=int, default=None, help="1-based manifest row index for Slurm arrays.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "consolidate":
        consolidate_run_metadata(
            metadata_parts_dir=args.metadata_parts_dir,
            metadata_path=args.run_metadata_path,
        )
        return 0

    if args.mode == "single":
        if args.single_manifest_path is None or args.single_row_index is None:
            raise ValueError("--single-manifest-path and --single-row-index are required for mode=single")
        runner = ClusteringBenchmarkRunner(
            assignments_dir=args.assignments_dir,
            metadata_path=args.run_metadata_path,
            metadata_parts_dir=args.metadata_parts_dir,
            pca_components=args.pca_components,
        )
        manifest = pd.read_csv(args.single_manifest_path, sep="\t")
        row = manifest.iloc[int(args.single_row_index) - 1]
        execute_single_manifest_row(row, args.processed_dir, runner)
        return 0

    grid_manifest = build_grid_manifest(include_extended=args.include_extended)
    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    grid_manifest.to_csv(manifest_path, sep="\t", index=False)

    execution_manifest = build_execution_manifest(args.processed_dir, include_extended=args.include_extended)
    execution_path = Path(args.execution_path)
    execution_path.parent.mkdir(parents=True, exist_ok=True)
    execution_manifest.to_csv(execution_path, sep="\t", index=False)
    execution_manifest.loc[execution_manifest["dataset_mode"] == "vdjdb"].to_csv(
        Path(args.vdjdb_manifest_path),
        sep="\t",
        index=False,
    )
    execution_manifest.loc[execution_manifest["dataset_mode"] == "yfv"].to_csv(
        Path(args.yfv_manifest_path),
        sep="\t",
        index=False,
    )

    if args.mode == "manifest":
        return 0

    runner = ClusteringBenchmarkRunner(
        assignments_dir=args.assignments_dir,
        metadata_path=args.run_metadata_path,
        metadata_parts_dir=args.metadata_parts_dir,
        pca_components=args.pca_components,
    )
    executed = execute_manifest(execution_manifest, args.processed_dir, runner)
    executed.to_csv(execution_path, sep="\t", index=False)
    consolidate_run_metadata(
        metadata_parts_dir=args.metadata_parts_dir,
        metadata_path=args.run_metadata_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
