from __future__ import annotations

import argparse
import re
import sys
import tarfile
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.data_sources import VDJDB_TARGETS
from benchmark.airr_utils import infer_chain
from benchmark.prepare_datasets import DEFAULT_TCRVDB_PADJ_THRESHOLD, build_known_yfv_clonotypes, build_vdjdb_truth_table, log_step
from benchmark.run_benchmark import consolidate_run_metadata, standardize_redcea_assignments


def extract_archive(archive_path: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    redcea_runs_root = results_dir / "redcea_runs"
    if redcea_runs_root.exists() and any(redcea_runs_root.iterdir()):
        log_step("Reusing existing extracted redcea_runs: {0}".format(redcea_runs_root))
        return
    log_step("Extracting archive {0} -> {1}".format(archive_path, results_dir))
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=results_dir)
    log_step("Finished extracting archive")


def parse_run_identity(run_id: str) -> dict[str, object]:
    if run_id.startswith("yfv_"):
        match = re.fullmatch(r"yfv_([A-Z][0-9]+_F[0-9]+)_(.+)_grid_([0-9]+)", run_id)
        if match is None:
            raise ValueError("Cannot parse YFV run id: {0}".format(run_id))
        donor_id, method, _grid = match.groups()
        return {
            "dataset": "yfv_repertoires",
            "dataset_mode": "yfv",
            "donor_id": donor_id,
            "epitope": None,
            "method": method,
        }
    match = re.fullmatch(r"vdjdb_([a-z]+)_(.+)_grid_([0-9]+)", run_id)
    if match is None:
        raise ValueError("Cannot parse VDJdb run id: {0}".format(run_id))
    dataset_key_raw, method, _grid = match.groups()
    dataset_key = dataset_key_raw.upper()
    if dataset_key not in VDJDB_TARGETS:
        raise ValueError("Unknown VDJdb dataset key in run id {0}: {1}".format(run_id, dataset_key))
    return {
        "dataset": "vdjdb_{0}".format(dataset_key.lower()),
        "dataset_mode": "vdjdb",
        "donor_id": None,
        "epitope": VDJDB_TARGETS[dataset_key]["epitope_sequence"],
        "method": method,
    }


def import_redcea_runs(
    *,
    results_dir: Path,
    tcrvdb_path: Path,
    padj_threshold: float,
) -> None:
    redcea_runs_dir = results_dir / "redcea_runs"
    assignments_dir = results_dir / "clustering_assignments"
    metadata_parts_dir = results_dir / "run_metadata" / "clustering_run_parts"
    metadata_path = results_dir / "run_metadata" / "clustering_runs.tsv"
    known_yfv_path = ROOT_DIR / "data" / "processed" / "known_yfv_vdjdb_clonotypes.tsv"

    assignments_dir.mkdir(parents=True, exist_ok=True)
    metadata_parts_dir.mkdir(parents=True, exist_ok=True)
    known_yfv_path.parent.mkdir(parents=True, exist_ok=True)

    truth_table = build_vdjdb_truth_table(tcrvdb_path, padj_threshold=padj_threshold)
    known_yfv = build_known_yfv_clonotypes(tcrvdb_path)
    known_yfv.to_csv(known_yfv_path, sep="\t", index=False)
    log_step("Wrote known YFV clonotypes rows={0}: {1}".format(len(known_yfv), known_yfv_path))

    for run_dir in sorted(redcea_runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        cluster_path = run_dir / "{0}_tcremp_clusters.tsv".format(run_id)
        if not cluster_path.exists():
            log_step("Skipping run without cluster table: {0}".format(run_dir))
            continue
        info = parse_run_identity(run_id)
        log_step("Importing archived run: {0}".format(run_id))
        cluster_df = pd.read_csv(cluster_path, sep="\t")
        if "junction_aa" not in cluster_df.columns and "cdr3aa_beta" in cluster_df.columns:
            cluster_df["junction_aa"] = cluster_df["cdr3aa_beta"]
        if "v_call" not in cluster_df.columns and "v_beta" in cluster_df.columns:
            cluster_df["v_call"] = cluster_df["v_beta"]
        if "j_call" not in cluster_df.columns and "j_beta" in cluster_df.columns:
            cluster_df["j_call"] = cluster_df["j_beta"]
        if "locus" not in cluster_df.columns:
            chain = infer_chain(cluster_df, default="TRB").astype(str)
            cluster_df["locus"] = chain.replace({"TRB": "beta", "TRA": "alpha"})
        parameter_json = "{}"
        assignments = standardize_redcea_assignments(
            cluster_df,
            run_id=run_id,
            dataset=str(info["dataset"]),
            dataset_mode=str(info["dataset_mode"]),
            method=str(info["method"]),
            parameter_json=parameter_json,
            epitope=info["epitope"],
            donor_id=info["donor_id"],
            truth_table=truth_table,
        )
        assignments.to_parquet(assignments_dir / "{0}.parquet".format(run_id), index=False)
        metadata = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "dataset": str(info["dataset"]),
                    "dataset_mode": str(info["dataset_mode"]),
                    "method": str(info["method"]),
                    "parameter_json": parameter_json,
                    "n_points": int(len(assignments)),
                    "n_clusters": int(assignments.loc[~assignments["is_noise"], "cluster_id"].nunique()),
                    "n_noise": int(assignments["is_noise"].sum()),
                    "noise_fraction": float(assignments["is_noise"].mean()) if len(assignments) else 0.0,
                    "runtime_seconds": 0.0,
                    "status": "success",
                    "error_message": "",
                    "error_traceback": "",
                    "redcea_output_dir": str(run_dir),
                }
            ]
        )
        metadata.to_csv(metadata_parts_dir / "{0}.tsv".format(run_id), sep="\t", index=False)

    consolidate_run_metadata(metadata_parts_dir=metadata_parts_dir, metadata_path=metadata_path)
    log_step("Imported archived runs into benchmark results directories")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import archived redcea_runs into local benchmark outputs.")
    parser.add_argument("--archive-path", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tcrvdb-path", default=str(ROOT_DIR / "data" / "01_05_2025_TCRvdb.csv"))
    parser.add_argument("--padj-threshold", type=float, default=DEFAULT_TCRVDB_PADJ_THRESHOLD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_path = Path(args.archive_path)
    results_dir = Path(args.results_dir)
    extract_archive(archive_path, results_dir)
    import_redcea_runs(
        results_dir=results_dir,
        tcrvdb_path=Path(args.tcrvdb_path),
        padj_threshold=args.padj_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
