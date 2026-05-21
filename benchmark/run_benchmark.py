from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.grids import get_enabled_methods, get_method_grid
from benchmark.prepare_datasets import align_yfv_embedding_with_airr
from benchmark.runner import ClusteringBenchmarkRunner


def log_step(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[{0}] {1}".format(timestamp, message), flush=True)


def consolidate_run_metadata(
    *,
    metadata_parts_dir: str | Path = "results/run_metadata/clustering_run_parts",
    metadata_path: str | Path = "results/run_metadata/clustering_runs.tsv",
) -> pd.DataFrame:
    metadata_parts_dir = Path(metadata_parts_dir)
    part_files = sorted(metadata_parts_dir.glob("*.tsv"))
    log_step("Consolidating run metadata from {0}; part_files={1}".format(metadata_parts_dir, len(part_files)))
    if not part_files:
        empty = pd.DataFrame(
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
                "error_traceback",
            ]
        )
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(metadata_path, sep="\t", index=False)
        log_step("No metadata parts found; wrote empty metadata table: {0}".format(metadata_path))
        return empty
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
    status_counts = consolidated["status"].value_counts(dropna=False).to_dict() if "status" in consolidated.columns else {}
    log_step(
        "Wrote consolidated metadata rows={0}, unique_runs={1}, status_counts={2}: {3}".format(
            len(consolidated),
            consolidated["run_id"].nunique() if "run_id" in consolidated.columns else "NA",
            status_counts,
            metadata_path,
        )
    )
    return consolidated


def build_grid_manifest(include_extended=False):
    log_step("Building grid manifest; include_extended={0}".format(include_extended))
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
    method_counts = manifest["method"].value_counts().to_dict() if len(manifest) else {}
    log_step("Built grid manifest rows={0}, method_counts={1}".format(len(manifest), method_counts))
    return manifest


def _load_vdjdb_processed_inputs(processed_dir):
    processed_dir = Path(processed_dir)
    required = [
        processed_dir / "vdjdb_glc.parquet",
        processed_dir / "vdjdb_ylq.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed benchmark inputs are missing. Expected files: {0}".format(", ".join(missing))
        )
    glc = pd.read_parquet(processed_dir / "vdjdb_glc.parquet")
    ylq = pd.read_parquet(processed_dir / "vdjdb_ylq.parquet")
    log_step("Loaded VDJdb processed inputs: GLC rows={0}, YLQ rows={1}".format(len(glc), len(ylq)))
    return glc, ylq


def _load_vdjdb_processed_input(processed_dir, dataset):
    processed_dir = Path(processed_dir)
    if dataset == "vdjdb_glc":
        path = processed_dir / "vdjdb_glc.parquet"
    elif dataset == "vdjdb_ylq":
        path = processed_dir / "vdjdb_ylq.parquet"
    else:
        raise KeyError("Unknown vdjdb dataset: {0}".format(dataset))
    if not path.exists():
        raise FileNotFoundError("Processed VDJdb input is missing: {0}".format(path))
    frame = pd.read_parquet(path)
    log_step("Loaded VDJdb processed input dataset={0}, rows={1}: {2}".format(dataset, len(frame), path))
    return frame


def _validate_vdjdb_processed_inputs(processed_dir):
    processed_dir = Path(processed_dir)
    required = [
        processed_dir / "vdjdb_glc.parquet",
        processed_dir / "vdjdb_ylq.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed benchmark inputs are missing. Expected files: {0}".format(", ".join(missing))
        )


def _load_yfv_manifest(processed_dir):
    manifest_path = Path(processed_dir) / "yfv_repertoires_manifest.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError("Processed YFV manifest is missing: {0}".format(manifest_path))
    manifest = pd.read_csv(manifest_path, sep="\t")
    log_step("Loaded YFV processed manifest rows={0}: {1}".format(len(manifest), manifest_path))
    return manifest


def build_execution_manifest(processed_dir, include_extended=False):
    log_step("Building execution manifest from processed_dir={0}".format(processed_dir))
    _validate_vdjdb_processed_inputs(processed_dir)
    yfv_manifest = _load_yfv_manifest(processed_dir)
    grid_manifest = build_grid_manifest(include_extended=include_extended)
    rows = []

    vdjdb_datasets = [
        ("vdjdb_glc", "GLC"),
        ("vdjdb_ylq", "YLQ"),
    ]
    for _, row in grid_manifest.iterrows():
        method = row["method"]
        for dataset_name, epitope in vdjdb_datasets:
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

        for donor_id in yfv_manifest["donor_id"].astype(str).tolist():
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
    manifest = pd.DataFrame(rows)
    mode_counts = manifest["dataset_mode"].value_counts().to_dict() if len(manifest) else {}
    log_step("Built execution manifest rows={0}, dataset_mode_counts={1}".format(len(manifest), mode_counts))
    return manifest


def _load_dataset_frame(processed_dir, dataset_mode, dataset, donor_id=None):
    if dataset_mode == "vdjdb":
        return _load_vdjdb_processed_input(processed_dir, dataset)
    if dataset_mode == "yfv":
        if donor_id is None:
            raise ValueError("donor_id is required for yfv execution rows")
        yfv_manifest = _load_yfv_manifest(processed_dir)
        matched = yfv_manifest.loc[yfv_manifest["donor_id"].astype(str) == str(donor_id)]
        if matched.empty:
            raise KeyError("Unknown yfv donor_id in manifest: {0}".format(donor_id))
        row = matched.iloc[0]
        sample = align_yfv_embedding_with_airr(
            str(donor_id),
            sample_label="sample",
            embedding_path=Path(row["sample_embedding_path"]),
            airr_path=Path(row["sample_airr_path"]),
        )
        background = align_yfv_embedding_with_airr(
            str(donor_id),
            sample_label="background",
            embedding_path=Path(row["background_embedding_path"]),
            airr_path=Path(row["background_airr_path"]),
        )
        return pd.concat([sample, background], ignore_index=True)
    raise KeyError("Unknown dataset_mode: {0}".format(dataset_mode))


def execute_single_manifest_row(manifest_row, processed_dir, runner):
    params = json.loads(manifest_row["parameter_json"])
    print(
        "Starting run_id={0} dataset_mode={1} dataset={2} donor_id={3} epitope={4} method={5} params={6}".format(
            manifest_row["run_id"],
            manifest_row["dataset_mode"],
            manifest_row["dataset"],
            manifest_row.get("donor_id"),
            manifest_row.get("epitope"),
            manifest_row["method"],
            manifest_row["parameter_json"],
        ),
        flush=True,
    )
    frame = _load_dataset_frame(
        processed_dir,
        dataset_mode=manifest_row["dataset_mode"],
        dataset=manifest_row["dataset"],
        donor_id=manifest_row.get("donor_id"),
    )
    print("Loaded input frame for {0}: n_rows={1}, n_cols={2}".format(manifest_row["run_id"], len(frame), len(frame.columns)), flush=True)
    result = runner.run(
        frame,
        dataset=manifest_row["dataset"],
        dataset_mode=manifest_row["dataset_mode"],
        method=manifest_row["method"],
        parameters=params,
        epitope=None if pd.isna(manifest_row.get("epitope")) else manifest_row.get("epitope"),
        donor_id=None if pd.isna(manifest_row.get("donor_id")) else str(manifest_row.get("donor_id")),
        run_id=manifest_row["run_id"],
    )
    print(
        "Finished run_id={0} status={1} n_points={2} n_clusters={3} n_noise={4} runtime_seconds={5}".format(
            result.run_id,
            result.metadata.get("status"),
            result.metadata.get("n_points"),
            result.metadata.get("n_clusters"),
            result.metadata.get("n_noise"),
            result.metadata.get("runtime_seconds"),
        ),
        flush=True,
    )
    if result.metadata.get("status") != "success":
        print("ERROR run_id={0}: {1}".format(result.run_id, result.metadata.get("error_message", "")), flush=True)
        traceback_text = str(result.metadata.get("error_traceback", "") or "").strip()
        if traceback_text:
            print("TRACEBACK run_id={0}:\n{1}".format(result.run_id, traceback_text), flush=True)
    return result


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
        consolidated = consolidate_run_metadata(
            metadata_parts_dir=args.metadata_parts_dir,
            metadata_path=args.run_metadata_path,
        )
        print(
            "Consolidated run metadata: {0} rows -> {1}".format(
                len(consolidated),
                args.run_metadata_path,
            ),
            flush=True,
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
        print(
            "Array task row={0} manifest={1} manifest_rows={2}".format(
                args.single_row_index,
                args.single_manifest_path,
                len(manifest),
            ),
            flush=True,
        )
        row = manifest.iloc[int(args.single_row_index) - 1]
        execute_single_manifest_row(row, args.processed_dir, runner)
        return 0

    grid_manifest = build_grid_manifest(include_extended=args.include_extended)
    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    grid_manifest.to_csv(manifest_path, sep="\t", index=False)
    print("Wrote grid manifest: {0} ({1} parameter rows)".format(manifest_path, len(grid_manifest)), flush=True)

    execution_manifest = build_execution_manifest(args.processed_dir, include_extended=args.include_extended)
    execution_path = Path(args.execution_path)
    execution_path.parent.mkdir(parents=True, exist_ok=True)
    execution_manifest.to_csv(execution_path, sep="\t", index=False)
    vdjdb_manifest = execution_manifest.loc[execution_manifest["dataset_mode"] == "vdjdb"]
    yfv_manifest = execution_manifest.loc[execution_manifest["dataset_mode"] == "yfv"]
    vdjdb_manifest.to_csv(
        Path(args.vdjdb_manifest_path),
        sep="\t",
        index=False,
    )
    yfv_manifest.to_csv(
        Path(args.yfv_manifest_path),
        sep="\t",
        index=False,
    )
    print("Wrote execution manifest: {0} ({1} total rows)".format(execution_path, len(execution_manifest)), flush=True)
    print("Wrote VDJdb manifest: {0} ({1} array tasks)".format(args.vdjdb_manifest_path, len(vdjdb_manifest)), flush=True)
    print("Wrote YFV manifest: {0} ({1} array tasks)".format(args.yfv_manifest_path, len(yfv_manifest)), flush=True)
    print("Methods: {0}".format(", ".join(grid_manifest["method"].drop_duplicates().astype(str).tolist())), flush=True)

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
