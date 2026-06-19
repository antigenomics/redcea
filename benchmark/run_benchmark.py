from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.airr_utils import standardize_metadata_frame, to_tcremp_airr_frame
from benchmark.data_sources import DEFAULT_VDJDB_BENCHMARK_TARGETS, resolve_vdjdb_target_keys
from benchmark.grids import get_enabled_methods, get_method_grid
from benchmark.prepare_datasets import (
    DEFAULT_TCRVDB_PADJ_THRESHOLD,
    build_vdjdb_truth_table,
)
from redcea.config import PipelineConfig


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
                "dataset_mode",
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
                "redcea_output_dir",
            ]
        )
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(metadata_path, sep="\t", index=False)
        return empty
    frames = [pd.read_csv(path, sep="\t") for path in part_files]
    consolidated = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["dataset_mode", "dataset", "method", "run_id"])
        .drop_duplicates(subset=["run_id"], keep="last")
        .reset_index(drop=True)
    )
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated.to_csv(metadata_path, sep="\t", index=False)
    return consolidated


def build_grid_manifest(grid_size="small"):
    rows = []
    for method in get_enabled_methods(grid_size=grid_size):
        for params in get_method_grid(method, grid_size=grid_size):
            rows.append(
                {
                    "method": method,
                    "parameter_json": json.dumps(params, sort_keys=True, separators=(",", ":")),
                }
            )
    manifest = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    manifest.insert(0, "grid_id", ["grid_{0:04d}".format(i) for i in range(len(manifest))])
    log_step(
        "Built grid manifest for grid_size={0}; methods={1}; rows={2}".format(
            grid_size,
            ",".join(get_enabled_methods(grid_size=grid_size)),
            len(manifest),
        )
    )
    return manifest


def _load_dataset_manifest(processed_dir: str | Path) -> pd.DataFrame:
    path = Path(processed_dir) / "benchmark_dataset_manifest.tsv"
    if not path.exists():
        raise FileNotFoundError("Dataset manifest is missing: {0}".format(path))
    return pd.read_csv(path, sep="\t")


def _normalize_csv_tokens(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    tokens = [token.strip() for token in str(raw_value).split(",") if token.strip()]
    return tokens or None


def _filter_dataset_manifest(
    dataset_manifest: pd.DataFrame,
    *,
    dataset_mode_filter: str = "all",
    yfv_donor_ids: list[str] | None = None,
    vdjdb_targets: list[str] | None = None,
) -> pd.DataFrame:
    filtered = dataset_manifest.copy()
    if dataset_mode_filter != "all":
        filtered = filtered.loc[filtered["dataset_mode"].astype(str) == str(dataset_mode_filter)].copy()
    if vdjdb_targets:
        selected_targets = set(resolve_vdjdb_target_keys(vdjdb_targets))
        vdjdb_mask = filtered["dataset_mode"].astype(str) == "vdjdb"
        filtered = filtered.loc[
            ~vdjdb_mask | filtered["dataset_key"].astype(str).isin(selected_targets)
        ].copy()
    if yfv_donor_ids:
        donor_set = {str(donor_id) for donor_id in yfv_donor_ids}
        yfv_mask = filtered["dataset_mode"].astype(str) == "yfv"
        filtered = filtered.loc[~yfv_mask | filtered["donor_id"].astype(str).isin(donor_set)].copy()
    return filtered.reset_index(drop=True)


def build_execution_manifest(
    processed_dir,
    grid_size="small",
    *,
    dataset_mode_filter: str = "all",
    yfv_donor_ids: list[str] | None = None,
    vdjdb_targets: list[str] | None = None,
):
    dataset_manifest = _load_dataset_manifest(processed_dir)
    dataset_manifest = _filter_dataset_manifest(
        dataset_manifest,
        dataset_mode_filter=dataset_mode_filter,
        yfv_donor_ids=yfv_donor_ids,
        vdjdb_targets=vdjdb_targets,
    )
    grid_manifest = build_grid_manifest(grid_size=grid_size)
    rows = []
    for _, dataset_row in dataset_manifest.iterrows():
        for _, grid_row in grid_manifest.iterrows():
            dataset_mode = str(dataset_row["dataset_mode"])
            dataset = str(dataset_row["dataset"])
            method = str(grid_row["method"])
            donor_id = None if pd.isna(dataset_row.get("donor_id")) else str(dataset_row.get("donor_id"))
            epitope = None if pd.isna(dataset_row.get("epitope")) else str(dataset_row.get("epitope"))
            if dataset_mode == "vdjdb":
                run_id = "{0}_{1}_{2}".format(dataset, method, grid_row["grid_id"])
            else:
                run_id = "yfv_{0}_{1}_{2}".format(donor_id, method, grid_row["grid_id"])
            row = dataset_row.to_dict()
            row.update(
                {
                    "grid_id": grid_row["grid_id"],
                    "config_id": grid_row["grid_id"],
                    "run_id": run_id,
                    "method": method,
                    "parameter_json": grid_row["parameter_json"],
                    "status": "pending",
                    "epitope": epitope,
                    "donor_id": donor_id,
                }
            )
            rows.append(row)
    execution_manifest = pd.DataFrame(rows)
    log_step(
        "Built execution manifest for grid_size={0}; dataset_mode_filter={1}; yfv_donors={2}; dataset_rows={3}; grid_rows={4}; execution_rows={5}".format(
            grid_size,
            dataset_mode_filter,
            ",".join(yfv_donor_ids) if yfv_donor_ids else "all",
            len(dataset_manifest),
            len(grid_manifest),
            len(execution_manifest),
        )
    )
    return execution_manifest


def _sample_label_from_clone_id(clone_id: pd.Series) -> pd.Series:
    clone_id = clone_id.fillna("").astype(str)
    return pd.Series(
        ["sample" if value.startswith("s_") else "background" if value.startswith("b_") else None for value in clone_id],
        index=clone_id.index,
        dtype="object",
    )


def _build_yfv_annotations(assignments: pd.DataFrame, donor_id: str) -> pd.DataFrame:
    out = assignments.copy()
    out["sample_label"] = _sample_label_from_clone_id(out["clone_id"])
    donor_id = str(donor_id)
    if "_" in donor_id:
        subject, replicate = donor_id.split("_", 1)
    else:
        subject, replicate = donor_id, None
    out["donor_id"] = donor_id
    out["subject_id"] = subject
    out["replicate_id"] = replicate
    out["timepoint"] = out["sample_label"].map({"sample": "15", "background": "0"})
    out["sample_type"] = out["sample_label"].map({"sample": "post", "background": "pre"})
    out["truth_label"] = None
    return out


def _build_vdjdb_annotations(
    assignments: pd.DataFrame,
    *,
    epitope: str,
    truth_table: pd.DataFrame,
) -> pd.DataFrame:
    out = assignments.copy()
    out["sample_label"] = _sample_label_from_clone_id(out["clone_id"])
    out["truth_label"] = "unlabeled"
    sample_mask = out["sample_label"].eq("sample")
    truth_ep = truth_table.loc[(truth_table["epitope_aa"] == epitope) & (truth_table["chain"] == "TRB")].copy()
    truth_ep = truth_ep.drop_duplicates(subset=["cdr3", "v_gene", "j_gene", "chain"], keep="first")
    if sample_mask.any():
        merged = out.loc[sample_mask, ["cdr3", "v_gene", "j_gene", "chain"]].merge(
            truth_ep[["cdr3", "v_gene", "j_gene", "chain", "truth_label"]],
            on=["cdr3", "v_gene", "j_gene", "chain"],
            how="left",
        )
        out.loc[sample_mask, "truth_label"] = merged["truth_label"].fillna("unlabeled").to_numpy()
    return out


def standardize_redcea_assignments(
    cluster_df: pd.DataFrame,
    *,
    run_id: str,
    dataset: str,
    dataset_mode: str,
    method: str,
    parameter_json: str,
    epitope: str | None,
    donor_id: str | None,
    truth_table: pd.DataFrame,
) -> pd.DataFrame:
    standardized = standardize_metadata_frame(cluster_df, chain_default="TRB").reset_index(drop=True)
    airr_like = to_tcremp_airr_frame(cluster_df, chain_default="TRB").reset_index(drop=True)
    out = cluster_df.copy().reset_index(drop=True)
    for column in ("junction_aa", "v_call", "j_call", "locus"):
        if column not in out.columns:
            out[column] = airr_like[column]

    out["cdr3"] = standardized["cdr3"]
    out["v_gene"] = standardized["v_gene"]
    out["j_gene"] = standardized["j_gene"]
    out["chain"] = standardized["chain"]
    out["clonotype_id"] = out["clone_id"].astype(str)
    out["cdr3_length"] = out["cdr3"].fillna("").astype(str).str.len()
    out["cluster_id"] = out["cluster_id"].astype(int)
    out["is_noise"] = out["cluster_id"].eq(-1)

    if dataset_mode == "vdjdb":
        out = _build_vdjdb_annotations(out, epitope=str(epitope), truth_table=truth_table)
        out["sample_type"] = out["sample_label"].map({"sample": "vdjdb_epitope", "background": "background"})
        out["timepoint"] = None
        out["subject_id"] = None
        out["replicate_id"] = None
        out["donor_id"] = None
    else:
        out = _build_yfv_annotations(out, str(donor_id))

    out["run_id"] = run_id
    out["dataset"] = dataset
    out["dataset_mode"] = dataset_mode
    out["epitope"] = epitope
    out["method"] = method
    out["parameter_json"] = parameter_json

    keep_columns = [
        "run_id",
        "dataset",
        "dataset_mode",
        "epitope",
        "donor_id",
        "method",
        "parameter_json",
        "clonotype_id",
        "junction_aa",
        "v_call",
        "j_call",
        "locus",
        "cdr3",
        "cdr3_length",
        "v_gene",
        "j_gene",
        "chain",
        "sample_label",
        "sample_type",
        "timepoint",
        "subject_id",
        "replicate_id",
        "truth_label",
        "cluster_id",
        "is_noise",
    ]
    return out[keep_columns].copy()


def _build_pipeline_config(manifest_row, params, output_dir: Path, nproc: int | None) -> PipelineConfig:
    env_nproc = os.environ.get("SLURM_CPUS_PER_TASK")
    resolved_nproc = nproc if nproc is not None else (int(env_nproc) if env_nproc else None)

    def _optional_str(value):
        if pd.isna(value):
            return None
        text = str(value).strip()
        return text or None

    return PipelineConfig(
        sample=str(manifest_row["sample_airr_path"]),
        background=str(manifest_row["background_airr_path"]),
        output=str(output_dir),
        prefix=str(manifest_row["run_id"]),
        index_col=None,
        chain=str(manifest_row.get("chain", "TRB")),
        species=str(manifest_row.get("species", "HomoSapiens")),
        prototypes_path=None,
        nproc=resolved_nproc,
        lower_len_cdr3=None,
        higher_len_cdr3=None,
        metrics="euclidean",
        sample_embedding=_optional_str(manifest_row.get("sample_embedding_path")),
        background_embedding=_optional_str(manifest_row.get("background_embedding_path")),
        n_bg_points=None,
        n_clonotypes=None,
        sample_random_clonotypes=False,
        random_seed=int(params.get("random_seed", 17)),
        cluster_pc_components=int(params.get("cluster_pc_components", 50)),
        core_min_samples=int(params.get("core_min_samples", params.get("cluster_min_samples", 5))),
        k_neighbors=int(params.get("k_neighbors", 4)),
        eps_k_neighbors=int(params.get("eps_k_neighbors", params.get("k_neighbors", 4))),
        leiden_resolution=float(params.get("leiden_resolution", 1.0)),
        leiden_sub_resolution=float(params.get("leiden_sub_resolution", 1.0)),
        cluster_algo=str(manifest_row["method"]),
        eps_estimation_based_on=str(params.get("eps_estimation_based_on", "sample")),
        vdbscan_sym_rule=str(params.get("vdbscan_sym_rule", "asymmetric")),
        enrichment_test=str(params.get("enrichment_test", "zbinom")),
        debug_save_intermediate=False,
        debug_output_dir=None,
        add_auxiliary_cluster_metrics=bool(params.get("add_auxiliary_cluster_metrics", False)),
    )


def _save_run_metadata(metadata_parts_dir: Path, metadata: dict[str, object]) -> None:
    metadata_parts_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metadata]).to_csv(
        metadata_parts_dir / "{0}.tsv".format(metadata["run_id"]),
        sep="\t",
        index=False,
    )


def execute_single_manifest_row(
    manifest_row,
    *,
    assignments_dir: str | Path,
    metadata_parts_dir: str | Path,
    redcea_runs_dir: str | Path,
    tcrvdb_path: str | Path,
    padj_threshold: float,
    nproc: int | None,
):
    run_id = str(manifest_row["run_id"])
    dataset_mode = str(manifest_row["dataset_mode"])
    dataset = str(manifest_row["dataset"])
    method = str(manifest_row["method"])
    params = json.loads(manifest_row["parameter_json"])
    redcea_output_dir = Path(redcea_runs_dir) / run_id
    redcea_output_dir.mkdir(parents=True, exist_ok=True)
    if dataset_mode == "vdjdb":
        truth_table = build_vdjdb_truth_table(tcrvdb_path, padj_threshold=padj_threshold)
    else:
        truth_table = pd.DataFrame(columns=["epitope_aa", "chain", "cdr3", "v_gene", "j_gene", "truth_label"])
    started = time.perf_counter()
    error_traceback = ""
    assignments = None
    log_step(
        "Starting benchmark run run_id={0} dataset_mode={1} dataset={2} method={3} params={4}".format(
            run_id,
            dataset_mode,
            dataset,
            method,
            json.dumps(params, sort_keys=True),
        )
    )
    try:
        from redcea.pipeline import run_redcea_pipeline

        config = _build_pipeline_config(manifest_row, params, redcea_output_dir, nproc=nproc)
        artifacts = run_redcea_pipeline(config)
        assignments = standardize_redcea_assignments(
            artifacts.cluster_df,
            run_id=run_id,
            dataset=dataset,
            dataset_mode=dataset_mode,
            method=method,
            parameter_json=str(manifest_row["parameter_json"]),
            epitope=None if pd.isna(manifest_row.get("epitope")) else str(manifest_row.get("epitope")),
            donor_id=None if pd.isna(manifest_row.get("donor_id")) else str(manifest_row.get("donor_id")),
            truth_table=truth_table,
        )
        status = "success"
        error_message = ""
    except Exception as exc:
        status = "error"
        error_message = "{0}: {1}".format(type(exc).__name__, str(exc))
        error_traceback = traceback.format_exc()

    runtime_seconds = time.perf_counter() - started
    metadata = {
        "run_id": run_id,
        "dataset": dataset,
        "dataset_mode": dataset_mode,
        "method": method,
        "parameter_json": str(manifest_row["parameter_json"]),
        "n_points": int(len(assignments)) if assignments is not None else 0,
        "n_clusters": int(assignments.loc[~assignments["is_noise"], "cluster_id"].nunique()) if assignments is not None else 0,
        "n_noise": int(assignments["is_noise"].sum()) if assignments is not None else 0,
        "noise_fraction": float(assignments["is_noise"].mean()) if assignments is not None and len(assignments) else 0.0,
        "runtime_seconds": float(runtime_seconds),
        "status": status,
        "error_message": error_message,
        "error_traceback": error_traceback,
        "redcea_output_dir": str(redcea_output_dir),
    }
    if assignments is not None:
        assignments_dir = Path(assignments_dir)
        assignments_dir.mkdir(parents=True, exist_ok=True)
        assignments.to_parquet(assignments_dir / "{0}.parquet".format(run_id), index=False)
    _save_run_metadata(Path(metadata_parts_dir), metadata)
    if status == "success":
        log_step(
            "Finished benchmark run run_id={0} status=success runtime_seconds={1:.2f} n_points={2} n_clusters={3} n_noise={4}".format(
                run_id,
                runtime_seconds,
                metadata["n_points"],
                metadata["n_clusters"],
                metadata["n_noise"],
            )
        )
    else:
        log_step(
            "Finished benchmark run run_id={0} status=error runtime_seconds={1:.2f} error={2}".format(
                run_id,
                runtime_seconds,
                error_message,
            )
        )
    return metadata


def execute_manifest(
    manifest: pd.DataFrame,
    *,
    assignments_dir: str | Path,
    metadata_parts_dir: str | Path,
    redcea_runs_dir: str | Path,
    tcrvdb_path: str | Path,
    padj_threshold: float,
    nproc: int | None,
) -> pd.DataFrame:
    rows = []
    for _, row in manifest.iterrows():
        rows.append(
            execute_single_manifest_row(
                row,
                assignments_dir=assignments_dir,
                metadata_parts_dir=metadata_parts_dir,
                redcea_runs_dir=redcea_runs_dir,
                tcrvdb_path=tcrvdb_path,
                padj_threshold=padj_threshold,
                nproc=nproc,
            )
        )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the RedCEA clustering benchmark over a hyperparameter grid.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--manifest-path", default="results/run_metadata/clustering_grid_manifest.tsv")
    parser.add_argument("--vdjdb-manifest-path", default="results/run_metadata/clustering_manifest_vdjdb.tsv")
    parser.add_argument("--yfv-manifest-path", default="results/run_metadata/clustering_manifest_yfv.tsv")
    parser.add_argument("--execution-path", default="results/run_metadata/clustering_execution_manifest.tsv")
    parser.add_argument("--assignments-dir", default="results/clustering_assignments")
    parser.add_argument("--run-metadata-path", default="results/run_metadata/clustering_runs.tsv")
    parser.add_argument("--metadata-parts-dir", default="results/run_metadata/clustering_run_parts")
    parser.add_argument("--redcea-runs-dir", default="results/redcea_runs")
    parser.add_argument("--tcrvdb-path", default=str(Path.home() / "01_05_2025_TCRvdb.csv"))
    parser.add_argument("--padj-threshold", type=float, default=DEFAULT_TCRVDB_PADJ_THRESHOLD)
    parser.add_argument("--nproc", type=int, default=None)
    parser.add_argument(
        "--grid-size",
        choices=["small", "large", "focused_vdbscan_leiden", "yfv_vdbscan_leiden_lowres", "yfv_redcea_benchmark"],
        default="small",
    )
    parser.add_argument("--dataset-mode-filter", choices=["all", "vdjdb", "yfv"], default="all")
    parser.add_argument(
        "--yfv-donor-ids",
        default=None,
        help="Comma-separated YFV donor IDs to keep in the execution manifest, e.g. P1_F1,P2_F1.",
    )
    parser.add_argument(
        "--vdjdb-targets",
        default=None,
        help=(
            "Comma-separated VDJdb targets to keep in the execution manifest. "
            "Tokens may be short keys like 'GLC,YLQ' or full epitope sequences like "
            "'GILGFVFTL,NLVPMVATV'. Default: {0}".format(",".join(DEFAULT_VDJDB_BENCHMARK_TARGETS))
        ),
    )
    parser.add_argument("--mode", choices=["manifest", "single", "all", "consolidate"], default="all")
    parser.add_argument("--single-manifest-path", default=None)
    parser.add_argument("--single-row-index", type=int, default=None, help="1-based manifest row index for Slurm arrays.")
    return parser.parse_args()


def main():
    args = parse_args()
    yfv_donor_ids = _normalize_csv_tokens(args.yfv_donor_ids)
    vdjdb_targets = _normalize_csv_tokens(args.vdjdb_targets)
    if args.mode == "consolidate":
        consolidate_run_metadata(
            metadata_parts_dir=args.metadata_parts_dir,
            metadata_path=args.run_metadata_path,
        )
        return 0

    if args.mode == "single":
        if args.single_manifest_path is None or args.single_row_index is None:
            raise ValueError("--single-manifest-path and --single-row-index are required for mode=single")
        manifest = pd.read_csv(args.single_manifest_path, sep="\t")
        row = manifest.iloc[int(args.single_row_index) - 1]
        execute_single_manifest_row(
            row,
            assignments_dir=args.assignments_dir,
            metadata_parts_dir=args.metadata_parts_dir,
            redcea_runs_dir=args.redcea_runs_dir,
            tcrvdb_path=args.tcrvdb_path,
            padj_threshold=args.padj_threshold,
            nproc=args.nproc,
        )
        return 0

    grid_manifest = build_grid_manifest(grid_size=args.grid_size)
    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    grid_manifest.to_csv(manifest_path, sep="\t", index=False)

    execution_manifest = build_execution_manifest(
        args.processed_dir,
        grid_size=args.grid_size,
        dataset_mode_filter=args.dataset_mode_filter,
        yfv_donor_ids=yfv_donor_ids,
        vdjdb_targets=vdjdb_targets,
    )
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

    execute_manifest(
        execution_manifest,
        assignments_dir=args.assignments_dir,
        metadata_parts_dir=args.metadata_parts_dir,
        redcea_runs_dir=args.redcea_runs_dir,
        tcrvdb_path=args.tcrvdb_path,
        padj_threshold=args.padj_threshold,
        nproc=args.nproc,
    )
    consolidate_run_metadata(
        metadata_parts_dir=args.metadata_parts_dir,
        metadata_path=args.run_metadata_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
