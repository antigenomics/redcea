from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.airr_utils import read_airr_like_table, standardize_metadata_frame, to_tcremp_airr_frame
from benchmark.grids import get_enabled_methods, get_method_grid
from benchmark.run_benchmark import build_execution_manifest, build_grid_manifest, consolidate_run_metadata, execute_manifest
from benchmark.workflows import load_assignment_tables, log_step
from scripts.run_additional_metrics_batch import collect_cluster_level_table, select_run_dirs


if hasattr(sns, "set_theme"):
    sns.set_theme(style="whitegrid", context="talk")


QVALUE_CLIP_MIN = 1e-300
GRID_SIZE = "yfv_redcea_benchmark"
TWIN_PAIRS = {"P1-P2", "Q1-Q2", "S1-S2"}

DONOR_RUN_COLUMNS = [
    "run_id",
    "donor",
    "config_id",
    "algorithm",
    "hyperparameters",
    "n_candidate_clusters",
    "n_enriched_clusters",
    "n_sample_clonotypes",
    "n_background_clonotypes",
    "n_sample_clonotypes_in_enriched_clusters",
    "fraction_sample_clonotypes_retained",
    "median_enriched_log2fc",
    "max_enriched_log2fc",
    "median_enriched_qvalue",
    "n_llw_matches_in_post",
    "n_llw_matches_in_enriched_clonotypes",
    "fraction_llw_matches_recovered",
    "n_clusters_with_llw",
    "n_enriched_clusters_with_llw",
    "fraction_llw_clusters_enriched",
]

CLUSTER_SUMMARY_COLUMNS = [
    "run_id",
    "donor",
    "config_id",
    "cluster_id",
    "n_sample",
    "n_background",
    "n_total",
    "log2fc",
    "pvalue",
    "qvalue",
    "is_enriched",
    "has_llw_match",
    "n_llw_matches",
]

LLW_SUMMARY_COLUMNS = [
    "run_id",
    "donor",
    "config_id",
    "n_llw_matches_in_post",
    "n_llw_matches_in_background",
    "n_llw_matches_in_enriched_clonotypes",
    "fraction_llw_matches_recovered",
    "n_clusters_with_llw",
    "n_enriched_clusters_with_llw",
    "fraction_llw_clusters_enriched",
]

OVERLAP_SUMMARY_COLUMNS = [
    "config_id",
    "donor_i",
    "donor_j",
    "is_twin_pair",
    "raw_day15_overlap_count",
    "raw_day15_overlap_fraction",
    "redcea_overlap_count",
    "redcea_overlap_fraction",
    "overlap_fold_change",
]

CONFIG_SUMMARY_COLUMNS = [
    "config_id",
    "algorithm",
    "hyperparameters",
    "n_donors_completed",
    "median_n_enriched_clusters",
    "median_fraction_sample_retained",
    "median_fraction_llw_recovered",
    "median_fraction_llw_clusters_enriched",
    "median_raw_overlap_fraction",
    "median_redcea_overlap_fraction",
    "median_overlap_fold_change",
    "median_lfc_mass_shift",
    "fragmentation_score",
]

BEST_CONFIG_COLUMNS = [
    "ranking_metric",
    "config_id",
    "algorithm",
    "hyperparameters",
    "metric_value",
]


@dataclass(frozen=True)
class BenchmarkPaths:
    outdir: Path
    runs_dir: Path
    assignments_dir: Path
    summaries_dir: Path
    figures_dir: Path
    logs_dir: Path
    normalized_airr_dir: Path
    dataset_manifest_path: Path
    llw_reference_path: Path
    execution_manifest_path: Path
    grid_manifest_path: Path
    run_metadata_parts_dir: Path
    run_metadata_path: Path
    clonotype_match_path: Path
    cluster_summary_path: Path
    donor_run_summary_path: Path
    llw_summary_path: Path
    overlap_summary_path: Path
    config_summary_path: Path
    best_config_summary_path: Path
    benchmark_summary_path: Path


def benchmark_paths(outdir: str | Path) -> BenchmarkPaths:
    outdir = Path(outdir)
    runs_dir = outdir / "runs"
    return BenchmarkPaths(
        outdir=outdir,
        runs_dir=runs_dir,
        assignments_dir=runs_dir / "assignments",
        summaries_dir=outdir / "summaries",
        figures_dir=outdir / "figures",
        logs_dir=outdir / "logs",
        normalized_airr_dir=outdir / "logs" / "normalized_airr",
        dataset_manifest_path=outdir / "logs" / "benchmark_dataset_manifest.tsv",
        llw_reference_path=outdir / "logs" / "known_llw_vdjdb.tsv",
        execution_manifest_path=outdir / "logs" / "yfv_execution_manifest.tsv",
        grid_manifest_path=outdir / "logs" / "yfv_grid_manifest.tsv",
        run_metadata_parts_dir=outdir / "logs" / "clustering_run_parts",
        run_metadata_path=outdir / "logs" / "clustering_runs.tsv",
        clonotype_match_path=outdir / "logs" / "yfv_llw_clonotype_matches.tsv",
        cluster_summary_path=outdir / "summaries" / "yfv_cluster_summary.tsv",
        donor_run_summary_path=outdir / "summaries" / "yfv_donor_run_summary.tsv",
        llw_summary_path=outdir / "summaries" / "yfv_llw_summary.tsv",
        overlap_summary_path=outdir / "summaries" / "yfv_overlap_summary.tsv",
        config_summary_path=outdir / "summaries" / "yfv_config_summary.tsv",
        best_config_summary_path=outdir / "summaries" / "yfv_best_config_summary.tsv",
        benchmark_summary_path=outdir / "YFV_BENCHMARK_SUMMARY.md",
    )


def ensure_output_dirs(outdir: str | Path) -> BenchmarkPaths:
    paths = benchmark_paths(outdir)
    for path in [
        paths.outdir,
        paths.runs_dir,
        paths.assignments_dir,
        paths.summaries_dir,
        paths.figures_dir,
        paths.logs_dir,
        paths.normalized_airr_dir,
        paths.run_metadata_parts_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_benchmark_config(config_path: str | Path) -> dict[str, object]:
    text = Path(config_path).read_text(encoding="utf-8")
    try:
        config = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "YFV benchmark config must be JSON-compatible YAML so it can be parsed without PyYAML: {0}".format(config_path)
        ) from exc
    donors = config.get("donors")
    if not isinstance(donors, dict) or not donors:
        raise ValueError("Config must define at least one donor under `donors`.")
    return config


def _normalize_chain_name(value: object, default: str = "TRB") -> str:
    text = str(value).strip().upper() if value is not None else default
    if text in {"TRB", "BETA"}:
        return "TRB"
    if text in {"TRA", "ALPHA"}:
        return "TRA"
    return default


def _grid_definition(config: dict[str, object]) -> dict[str, list[object]]:
    grid = config.get("grid")
    if isinstance(grid, dict):
        return {str(key): list(value) for key, value in grid.items()}
    return {
        "algorithm": ["vdbscan_leiden"],
        "k_neighbors": [8, 12, 16],
        "eps_k_neighbors": [4, 8, 12],
        "core_min_samples": [3],
        "leiden_resolution": [0.25, 0.5, 1.0],
        "eps_mode": ["sample"],
        "sym_rule": ["asymmetric"],
    }


def validate_grid_config(config: dict[str, object]) -> dict[str, object]:
    grid = _grid_definition(config)
    algorithms = [str(value) for value in grid.get("algorithm", ["vdbscan_leiden"])]
    if algorithms != ["vdbscan_leiden"]:
        raise ValueError("This YFV benchmark wrapper currently supports only `vdbscan_leiden`.")

    k_neighbors = [int(value) for value in grid.get("k_neighbors", [])]
    eps_k_neighbors = [int(value) for value in grid.get("eps_k_neighbors", [])]
    core_min_samples = [int(value) for value in grid.get("core_min_samples", [3])]
    leiden_resolution = [float(value) for value in grid.get("leiden_resolution", [])]
    eps_mode = [str(value) for value in grid.get("eps_mode", ["sample"])]
    sym_rule = [str(value) for value in grid.get("sym_rule", ["asymmetric"])]

    if not k_neighbors or not eps_k_neighbors or not leiden_resolution:
        raise ValueError("Grid must define non-empty `k_neighbors`, `eps_k_neighbors`, and `leiden_resolution`.")
    if core_min_samples != [3]:
        raise ValueError("Current YFV benchmark expects `core_min_samples` to contain only `3`.")
    if eps_mode != ["sample"]:
        raise ValueError("Current YFV benchmark expects `eps_mode` to contain only `sample`.")
    if sym_rule != ["asymmetric"]:
        raise ValueError("Current YFV benchmark expects `sym_rule` to contain only `asymmetric`.")

    invalid_pairs = [(k_value, eps_value) for k_value in k_neighbors for eps_value in eps_k_neighbors if eps_value > k_value]
    return {
        "invalid_pairs": invalid_pairs,
        "k_neighbors": k_neighbors,
        "eps_k_neighbors": eps_k_neighbors,
        "leiden_resolution": leiden_resolution,
    }


def _rename_source_columns(frame: pd.DataFrame, columns_cfg: dict[str, object], chain_default: str) -> pd.DataFrame:
    renamed = pd.DataFrame(index=frame.index)
    required = {
        "cdr3": str(columns_cfg["cdr3"]),
        "v_gene": str(columns_cfg["v_gene"]),
        "j_gene": str(columns_cfg["j_gene"]),
    }
    missing = [source for source in required.values() if source not in frame.columns]
    if missing:
        raise KeyError("Input table is missing required columns: {0}".format(", ".join(missing)))

    renamed["cdr3"] = frame[required["cdr3"]]
    renamed["v_gene"] = frame[required["v_gene"]]
    renamed["j_gene"] = frame[required["j_gene"]]
    chain_column = columns_cfg.get("chain")
    if chain_column and str(chain_column) in frame.columns:
        renamed["chain"] = frame[str(chain_column)]
    else:
        renamed["chain"] = chain_default

    if "clone_id" in frame.columns:
        renamed["clone_id"] = frame["clone_id"].astype(str)
    elif "clonotype_id" in frame.columns:
        renamed["clone_id"] = frame["clonotype_id"].astype(str)
    else:
        renamed["clone_id"] = pd.Series(np.arange(len(frame)), index=frame.index).astype(str)

    count_column = columns_cfg.get("count")
    if count_column and str(count_column) in frame.columns:
        renamed["count"] = pd.to_numeric(frame[str(count_column)], errors="coerce")
    return renamed


def _normalize_airr_file(
    source_path: str | Path,
    destination_path: Path,
    *,
    columns_cfg: dict[str, object],
    chain_default: str,
) -> Path:
    frame = read_airr_like_table(Path(source_path))
    renamed = _rename_source_columns(frame, columns_cfg, chain_default)
    tcremp_frame = to_tcremp_airr_frame(renamed, chain_default=chain_default)
    if "count" in renamed.columns:
        tcremp_frame["count"] = renamed["count"]
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    tcremp_frame.to_csv(destination_path, sep="\t", index=False)
    return destination_path


def _load_llw_reference(config: dict[str, object], *, allow_missing: bool) -> pd.DataFrame:
    vdjdb_cfg = dict(config.get("vdjdb", {}))
    path = Path(str(vdjdb_cfg.get("path", "")))
    if not path.exists():
        if allow_missing:
            return pd.DataFrame(columns=["cdr3", "v_gene", "j_gene", "chain", "epitope", "match_key"])
        raise FileNotFoundError("LLW/VDJdb file is missing: {0}".format(path))

    frame = read_airr_like_table(path)
    epitope_column = str(vdjdb_cfg.get("epitope_col", "epitope"))
    cdr3_col = str(vdjdb_cfg.get("cdr3_col", "cdr3"))
    v_col = str(vdjdb_cfg.get("v_col", "v"))
    j_col = str(vdjdb_cfg.get("j_col", "j"))
    chain_col = str(vdjdb_cfg.get("chain_col", "chain"))
    if epitope_column not in frame.columns:
        raise KeyError("LLW/VDJdb table is missing epitope column: {0}".format(epitope_column))

    llw = pd.DataFrame(index=frame.index)
    llw["cdr3"] = frame[cdr3_col]
    llw["v_gene"] = frame[v_col] if v_col in frame.columns else ""
    llw["j_gene"] = frame[j_col] if j_col in frame.columns else ""
    llw["chain"] = frame[chain_col] if chain_col in frame.columns else config.get("dataset", {}).get("chain", "TRB")
    standardized = standardize_metadata_frame(llw, chain_default=_normalize_chain_name(config.get("dataset", {}).get("chain", "TRB")))
    standardized["epitope"] = frame[epitope_column].astype(str)
    standardized = standardized.loc[standardized["epitope"] == str(vdjdb_cfg.get("epitope", "LLWNGPMAV"))].copy()
    standardized = standardized.loc[standardized["chain"].map(_normalize_chain_name) == "TRB"].copy()
    standardized["match_key"] = build_match_key(standardized, str(vdjdb_cfg.get("match_mode", "cdr3")))
    standardized = standardized.drop_duplicates(subset=["match_key", "epitope"], keep="first").reset_index(drop=True)
    return standardized


def build_match_key(frame: pd.DataFrame, match_mode: str) -> pd.Series:
    cdr3 = frame["cdr3"].fillna("").astype(str)
    if match_mode in {"cdr3", "cdr3_only"}:
        return cdr3
    if match_mode in {"cdr3_vj", "cdr3_v_gene_j_gene"}:
        v_gene = frame["v_gene"].fillna("").astype(str)
        j_gene = frame["j_gene"].fillna("").astype(str)
        return cdr3 + "|" + v_gene + "|" + j_gene
    raise ValueError("Unsupported LLW match_mode: {0}".format(match_mode))


def is_enriched_cluster_value(qvalue: object, log2fc: object, *, qvalue_threshold: float = 0.05, log2fc_threshold: float = 0.0) -> bool:
    qvalue_numeric = pd.to_numeric(pd.Series([qvalue]), errors="coerce").iloc[0]
    log2fc_numeric = pd.to_numeric(pd.Series([log2fc]), errors="coerce").iloc[0]
    if pd.isna(qvalue_numeric) or pd.isna(log2fc_numeric):
        return False
    return bool((qvalue_numeric < qvalue_threshold) and (log2fc_numeric > log2fc_threshold))


def compute_overlap_fraction(values_a: set[str], values_b: set[str]) -> float:
    if not values_a or not values_b:
        return 0.0
    return float(len(values_a & values_b) / min(len(values_a), len(values_b)))


def prepare_yfv_inputs(config_path: str | Path, outdir: str | Path, *, dry_run: bool = False) -> dict[str, object]:
    config = load_benchmark_config(config_path)
    validation = validate_grid_config(config)
    paths = ensure_output_dirs(outdir)
    chain_default = _normalize_chain_name(config.get("dataset", {}).get("chain", "TRB"))
    columns_cfg = dict(config.get("columns", {}))
    if not {"cdr3", "v_gene", "j_gene"} <= set(columns_cfg):
        raise ValueError("Config `columns` must define at least `cdr3`, `v_gene`, and `j_gene`.")

    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    for donor_id, donor_cfg_obj in dict(config.get("donors", {})).items():
        donor_cfg = dict(donor_cfg_obj)
        for key in ["pre", "post"]:
            if key not in donor_cfg:
                raise ValueError("Donor `{0}` is missing `{1}` path.".format(donor_id, key))

        pre_path = Path(str(donor_cfg["pre"]))
        post_path = Path(str(donor_cfg["post"]))
        normalized_pre = paths.normalized_airr_dir / "{0}_day0.tsv".format(donor_id)
        normalized_post = paths.normalized_airr_dir / "{0}_day15.tsv".format(donor_id)

        for source_path, normalized_path, label in [
            (pre_path, normalized_pre, "pre"),
            (post_path, normalized_post, "post"),
        ]:
            if source_path.exists():
                _normalize_airr_file(
                    source_path,
                    normalized_path,
                    columns_cfg=columns_cfg,
                    chain_default=chain_default,
                )
            elif dry_run:
                warnings.append("Missing donor file for dry-run validation: donor={0} label={1} path={2}".format(donor_id, label, source_path))
            else:
                raise FileNotFoundError("Missing donor file: donor={0} label={1} path={2}".format(donor_id, label, source_path))

        rows.append(
            {
                "dataset": str(config.get("dataset", {}).get("name", "yfv")),
                "dataset_mode": "yfv",
                "dataset_key": donor_id,
                "epitope": str(config.get("vdjdb", {}).get("epitope", "LLWNGPMAV")),
                "donor_id": donor_id,
                "chain": chain_default,
                "species": "HomoSapiens",
                "sample_airr_path": str(normalized_post if normalized_post.exists() else post_path),
                "background_airr_path": str(normalized_pre if normalized_pre.exists() else pre_path),
                "sample_embedding_path": str(donor_cfg.get("post_embedding", "")),
                "background_embedding_path": str(donor_cfg.get("pre_embedding", "")),
                "sample_index_path": str(donor_cfg.get("post_index", "")),
                "background_index_path": str(donor_cfg.get("pre_index", "")),
                "background_kind": "paired_pre_vaccination_repertoire",
                "source_pre_airr_path": str(pre_path),
                "source_post_airr_path": str(post_path),
            }
        )

    dataset_manifest = pd.DataFrame(rows)
    dataset_manifest.to_csv(paths.dataset_manifest_path, sep="\t", index=False)
    llw_reference = _load_llw_reference(config, allow_missing=dry_run)
    llw_reference.to_csv(paths.llw_reference_path, sep="\t", index=False)
    log_step("Prepared YFV dataset manifest rows={0}: {1}".format(len(dataset_manifest), paths.dataset_manifest_path))
    if validation["invalid_pairs"]:
        warnings.append(
            "Grid contains invalid eps/k pairs and they will be filtered by the named grid: {0}".format(validation["invalid_pairs"])
        )
    return {
        "dataset_manifest_path": paths.dataset_manifest_path,
        "llw_reference_path": paths.llw_reference_path,
        "warnings": warnings,
    }


def _build_yfv_manifests(config: dict[str, object], paths: BenchmarkPaths) -> pd.DataFrame:
    grid_manifest = build_grid_manifest(grid_size=str(config.get("grid_size", GRID_SIZE)))
    grid_manifest.to_csv(paths.grid_manifest_path, sep="\t", index=False)
    execution_manifest = build_execution_manifest(
        paths.logs_dir,
        grid_size=str(config.get("grid_size", GRID_SIZE)),
        dataset_mode_filter="yfv",
        yfv_donor_ids=list(dict(config.get("donors", {})).keys()),
    )
    execution_manifest.to_csv(paths.execution_manifest_path, sep="\t", index=False)
    return execution_manifest


def run_yfv_redcea_grid(
    config_path: str | Path,
    outdir: str | Path,
    *,
    dry_run: bool = False,
    nproc: int | None = None,
) -> pd.DataFrame:
    config = load_benchmark_config(config_path)
    paths = ensure_output_dirs(outdir)
    if not paths.dataset_manifest_path.exists():
        prepare_yfv_inputs(config_path, outdir, dry_run=dry_run)

    execution_manifest = _build_yfv_manifests(config, paths)
    if dry_run:
        return execution_manifest

    execute_manifest(
        execution_manifest,
        assignments_dir=paths.assignments_dir,
        metadata_parts_dir=paths.run_metadata_parts_dir,
        redcea_runs_dir=paths.runs_dir,
        tcrvdb_path=str(config.get("vdjdb", {}).get("path", "")),
        padj_threshold=1e-5,
        nproc=nproc,
    )
    consolidate_run_metadata(
        metadata_parts_dir=paths.run_metadata_parts_dir,
        metadata_path=paths.run_metadata_path,
    )
    return execution_manifest


def _load_execution_manifest(paths: BenchmarkPaths) -> pd.DataFrame:
    if not paths.execution_manifest_path.exists():
        return pd.DataFrame()
    manifest = pd.read_csv(paths.execution_manifest_path, sep="\t")
    if "config_id" not in manifest.columns and "grid_id" in manifest.columns:
        manifest["config_id"] = manifest["grid_id"]
    return manifest


def _load_clustering_outputs(paths: BenchmarkPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = _load_execution_manifest(paths)
    run_metadata = pd.read_csv(paths.run_metadata_path, sep="\t") if paths.run_metadata_path.exists() else pd.DataFrame()
    assignments, _ = load_assignment_tables(
        assignments_dir=paths.assignments_dir,
        metadata_parts_dir=paths.run_metadata_parts_dir,
        metadata_path=paths.run_metadata_path,
        only_success=True,
        dataset_mode="yfv",
        consolidate_metadata=False,
    )
    return manifest, run_metadata, assignments


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _load_cluster_metrics(paths: BenchmarkPaths, manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame()
    run_dirs = select_run_dirs(
        runs_root=paths.runs_dir,
        manifest_lookup=manifest,
        dataset_mode="yfv",
        run_id_prefix="yfv_",
    )
    return collect_cluster_level_table(run_dirs=run_dirs, manifest_lookup=manifest)


def annotate_yfv_llw(config_path: str | Path, outdir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_benchmark_config(config_path)
    paths = ensure_output_dirs(outdir)
    manifest, _run_metadata, assignments = _load_clustering_outputs(paths)
    cluster_metrics = _load_cluster_metrics(paths, manifest)

    if assignments.empty or cluster_metrics.empty:
        empty_matches = _empty_frame(
            ["run_id", "donor", "config_id", "clonotype_id", "sample_label", "cluster_id", "cdr3", "has_llw_match", "vdjdb_epitopes", "is_enriched_clonotype"]
        )
        empty_cluster_summary = _empty_frame(CLUSTER_SUMMARY_COLUMNS + ["algorithm", "hyperparameters", "vdjdb_epitopes"])
        empty_matches.to_csv(paths.clonotype_match_path, sep="\t", index=False)
        empty_cluster_summary.to_csv(paths.cluster_summary_path, sep="\t", index=False)
        return empty_matches, empty_cluster_summary

    if "config_id" not in cluster_metrics.columns and "grid_id" in cluster_metrics.columns:
        cluster_metrics["config_id"] = cluster_metrics["grid_id"]
    cluster_metrics["algorithm"] = cluster_metrics["method"].astype(str)
    cluster_metrics["hyperparameters"] = cluster_metrics["parameter_json"].astype(str)
    cluster_metrics["donor"] = cluster_metrics["donor_id"].astype(str)
    cluster_metrics["is_enriched"] = [
        is_enriched_cluster_value(
            qvalue,
            log2fc,
            qvalue_threshold=float(config.get("enrichment", {}).get("qvalue_threshold", 0.05)),
            log2fc_threshold=float(config.get("enrichment", {}).get("log2fc_threshold", 0.0)),
        )
        for qvalue, log2fc in zip(cluster_metrics["enrichment_fdr_zbinom"], cluster_metrics["log_fold_change"])
    ]

    llw_reference = _load_llw_reference(config, allow_missing=False)
    match_mode = str(dict(config.get("vdjdb", {})).get("match_mode", "cdr3"))
    enriched_lookup = cluster_metrics.loc[cluster_metrics["is_enriched"], ["run_id", "cluster_id"]].drop_duplicates()
    enriched_lookup["is_enriched_clonotype"] = True

    clonotype_matches = assignments.copy()
    clonotype_matches = clonotype_matches.merge(
        manifest[["run_id", "config_id"]] if len(manifest) else pd.DataFrame(columns=["run_id", "config_id"]),
        on="run_id",
        how="left",
    )
    clonotype_matches["donor"] = clonotype_matches["donor_id"].astype(str)
    clonotype_matches["match_key"] = build_match_key(clonotype_matches, match_mode)
    llw_hits = (
        llw_reference.groupby("match_key", as_index=False)
        .agg(
            vdjdb_epitopes=("epitope", lambda values: ";".join(sorted(set(map(str, values))))),
            n_reference_matches=("epitope", "size"),
        )
    )
    clonotype_matches = clonotype_matches.merge(llw_hits, on="match_key", how="left")
    clonotype_matches["has_llw_match"] = clonotype_matches["n_reference_matches"].fillna(0).astype(int) > 0
    clonotype_matches["vdjdb_epitopes"] = clonotype_matches["vdjdb_epitopes"].fillna("")
    clonotype_matches = clonotype_matches.merge(enriched_lookup, on=["run_id", "cluster_id"], how="left")
    clonotype_matches["is_enriched_clonotype"] = clonotype_matches["is_enriched_clonotype"].fillna(False).astype(bool)

    cluster_llw = (
        clonotype_matches.loc[clonotype_matches["has_llw_match"]]
        .groupby(["run_id", "cluster_id"], as_index=False)
        .agg(
            n_llw_matches=("clonotype_id", "nunique"),
            vdjdb_epitopes=("vdjdb_epitopes", lambda values: ";".join(sorted({token for value in values for token in str(value).split(";") if token}))),
        )
    )

    cluster_summary = cluster_metrics.merge(cluster_llw, on=["run_id", "cluster_id"], how="left")
    cluster_summary["n_llw_matches"] = cluster_summary["n_llw_matches"].fillna(0).astype(int)
    cluster_summary["has_llw_match"] = cluster_summary["n_llw_matches"] > 0
    cluster_summary["vdjdb_epitopes"] = cluster_summary["vdjdb_epitopes"].fillna("")
    cluster_summary = cluster_summary.rename(
        columns={
            "sample": "n_sample",
            "background": "n_background",
            "cluster_size": "n_total",
            "log_fold_change": "log2fc",
            "enrichment_pvalue_zbinom": "pvalue",
            "enrichment_fdr_zbinom": "qvalue",
        }
    )
    cluster_summary = cluster_summary[
        CLUSTER_SUMMARY_COLUMNS + ["algorithm", "hyperparameters", "vdjdb_epitopes"]
    ].sort_values(["donor", "config_id", "cluster_id"]).reset_index(drop=True)

    clonotype_matches = clonotype_matches[
        [
            "run_id",
            "donor",
            "config_id",
            "clonotype_id",
            "sample_label",
            "cluster_id",
            "cdr3",
            "v_gene",
            "j_gene",
            "chain",
            "has_llw_match",
            "vdjdb_epitopes",
            "is_enriched_clonotype",
        ]
    ].sort_values(["donor", "config_id", "sample_label", "clonotype_id"]).reset_index(drop=True)

    clonotype_matches.to_csv(paths.clonotype_match_path, sep="\t", index=False)
    cluster_summary.to_csv(paths.cluster_summary_path, sep="\t", index=False)
    return clonotype_matches, cluster_summary


def compute_requested_lfc_mass_shift(cluster_summary: pd.DataFrame, *, eps: float = 1e-12) -> float:
    if cluster_summary.empty:
        return 0.0
    qvalue = pd.to_numeric(cluster_summary["qvalue"], errors="coerce").clip(lower=QVALUE_CLIP_MIN, upper=1.0)
    log2fc = pd.to_numeric(cluster_summary["log2fc"], errors="coerce")
    valid = qvalue.notna() & log2fc.notna()
    if not valid.any():
        return 0.0
    weights = -np.log10(qvalue.loc[valid])
    lfc = log2fc.loc[valid]
    numerator = float((np.maximum(lfc, 0.0) * weights).sum())
    denominator = float((np.abs(lfc) * weights).sum()) + eps
    return numerator / denominator


def _run_fragmentation_metrics(cluster_summary: pd.DataFrame) -> tuple[float, float, float, float]:
    enriched = cluster_summary.loc[cluster_summary["is_enriched"]].copy()
    if enriched.empty:
        return 0.0, 1.0, 0.0, 1.0
    sizes = pd.to_numeric(enriched["n_total"], errors="coerce").fillna(0.0)
    median_size = float(sizes.median()) if len(sizes) else 0.0
    fraction_tiny = float((sizes <= 3).mean()) if len(sizes) else 1.0
    max_size = float(sizes.max()) if len(sizes) else 0.0
    giant_fraction = float(max_size / max(1.0, float(sizes.sum()))) if len(sizes) else 1.0
    fragmentation_score = 0.5 * fraction_tiny + 0.5 * giant_fraction
    return median_size, fraction_tiny, max_size, fragmentation_score


def summarize_yfv_benchmark(config_path: str | Path, outdir: str | Path) -> dict[str, pd.DataFrame]:
    _config = load_benchmark_config(config_path)
    paths = ensure_output_dirs(outdir)
    if not paths.cluster_summary_path.exists() or not paths.clonotype_match_path.exists():
        annotate_yfv_llw(config_path, outdir)

    cluster_summary = pd.read_csv(paths.cluster_summary_path, sep="\t") if paths.cluster_summary_path.exists() else _empty_frame(CLUSTER_SUMMARY_COLUMNS)
    clonotype_matches = pd.read_csv(paths.clonotype_match_path, sep="\t") if paths.clonotype_match_path.exists() else pd.DataFrame()

    if cluster_summary.empty or clonotype_matches.empty:
        donor_run_summary = _empty_frame(DONOR_RUN_COLUMNS + ["lfc_mass_shift", "median_enriched_cluster_size", "fraction_enriched_clusters_size_le_3", "max_enriched_cluster_size", "fragmentation_score"])
        llw_summary = _empty_frame(LLW_SUMMARY_COLUMNS)
        donor_run_summary.to_csv(paths.donor_run_summary_path, sep="\t", index=False)
        llw_summary.to_csv(paths.llw_summary_path, sep="\t", index=False)
        return {"cluster_summary": cluster_summary, "clonotype_matches": clonotype_matches, "donor_run_summary": donor_run_summary, "llw_summary": llw_summary}

    run_counts = (
        clonotype_matches.groupby(["run_id", "donor", "config_id", "sample_label"], as_index=False)
        .agg(n_clonotypes=("clonotype_id", "nunique"))
    )
    run_count_pivot = run_counts.pivot_table(
        index=["run_id", "donor", "config_id"],
        columns="sample_label",
        values="n_clonotypes",
        fill_value=0,
    ).reset_index()
    run_count_pivot.columns.name = None
    run_count_pivot = run_count_pivot.rename(columns={"sample": "n_sample_clonotypes", "background": "n_background_clonotypes"})

    llw_counts = (
        clonotype_matches.loc[clonotype_matches["has_llw_match"]]
        .groupby(["run_id", "donor", "config_id", "sample_label"], as_index=False)
        .agg(n_llw_matches=("clonotype_id", "nunique"))
    )
    if llw_counts.empty:
        llw_pivot = _empty_frame(["run_id", "donor", "config_id", "n_llw_matches_in_post", "n_llw_matches_in_background"])
    else:
        llw_pivot = llw_counts.pivot_table(
            index=["run_id", "donor", "config_id"],
            columns="sample_label",
            values="n_llw_matches",
            fill_value=0,
        ).reset_index()
        llw_pivot.columns.name = None
        llw_pivot = llw_pivot.rename(columns={"sample": "n_llw_matches_in_post", "background": "n_llw_matches_in_background"})

    enriched_llw_counts = (
        clonotype_matches.loc[clonotype_matches["has_llw_match"] & clonotype_matches["is_enriched_clonotype"] & clonotype_matches["sample_label"].eq("sample")]
        .groupby(["run_id", "donor", "config_id"], as_index=False)
        .agg(n_llw_matches_in_enriched_clonotypes=("clonotype_id", "nunique"))
    )
    if enriched_llw_counts.empty:
        enriched_llw_counts = _empty_frame(["run_id", "donor", "config_id", "n_llw_matches_in_enriched_clonotypes"])

    llw_cluster_counts = (
        cluster_summary.groupby(["run_id", "donor", "config_id"], as_index=False)
        .agg(
            n_clusters_with_llw=("has_llw_match", "sum"),
            n_enriched_clusters_with_llw=("has_llw_match", lambda values: int(values.index.size)),
        )
    )
    llw_cluster_counts["n_enriched_clusters_with_llw"] = (
        cluster_summary.loc[cluster_summary["is_enriched"] & cluster_summary["has_llw_match"]]
        .groupby(["run_id", "donor", "config_id"])["cluster_id"]
        .nunique()
        .reindex(llw_cluster_counts.set_index(["run_id", "donor", "config_id"]).index, fill_value=0)
        .to_numpy()
    )

    llw_summary = run_count_pivot.merge(llw_pivot, on=["run_id", "donor", "config_id"], how="left")
    llw_summary = llw_summary.merge(enriched_llw_counts, on=["run_id", "donor", "config_id"], how="left")
    llw_summary = llw_summary.merge(llw_cluster_counts, on=["run_id", "donor", "config_id"], how="left")
    for column in [
        "n_llw_matches_in_post",
        "n_llw_matches_in_background",
        "n_llw_matches_in_enriched_clonotypes",
        "n_clusters_with_llw",
        "n_enriched_clusters_with_llw",
    ]:
        llw_summary[column] = llw_summary[column].fillna(0).astype(int)
    llw_summary["fraction_llw_matches_recovered"] = llw_summary["n_llw_matches_in_enriched_clonotypes"] / llw_summary["n_llw_matches_in_post"].clip(lower=1)
    llw_summary["fraction_llw_clusters_enriched"] = llw_summary["n_enriched_clusters_with_llw"] / llw_summary["n_clusters_with_llw"].clip(lower=1)
    llw_summary = llw_summary[LLW_SUMMARY_COLUMNS].sort_values(["donor", "config_id"]).reset_index(drop=True)

    donor_rows: list[dict[str, object]] = []
    llw_lookup = llw_summary.set_index(["run_id", "donor", "config_id"], drop=False)
    count_lookup = run_count_pivot.set_index(["run_id", "donor", "config_id"], drop=False)
    for (run_id, donor, config_id), frame in cluster_summary.groupby(["run_id", "donor", "config_id"], sort=True):
        enriched = frame.loc[frame["is_enriched"]].copy()
        llw_row = llw_lookup.loc[(run_id, donor, config_id)] if (run_id, donor, config_id) in llw_lookup.index else {}
        count_row = count_lookup.loc[(run_id, donor, config_id)] if (run_id, donor, config_id) in count_lookup.index else {}
        retained = int(pd.to_numeric(enriched["n_sample"], errors="coerce").fillna(0).sum())
        median_size, fraction_tiny, max_size, fragmentation_score = _run_fragmentation_metrics(frame)
        donor_rows.append(
            {
                "run_id": run_id,
                "donor": donor,
                "config_id": config_id,
                "algorithm": frame["algorithm"].iloc[0],
                "hyperparameters": frame["hyperparameters"].iloc[0],
                "n_candidate_clusters": int(len(frame)),
                "n_enriched_clusters": int(enriched["cluster_id"].nunique()),
                "n_sample_clonotypes": int(count_row.get("n_sample_clonotypes", 0)),
                "n_background_clonotypes": int(count_row.get("n_background_clonotypes", 0)),
                "n_sample_clonotypes_in_enriched_clusters": retained,
                "fraction_sample_clonotypes_retained": float(retained / max(1, int(count_row.get("n_sample_clonotypes", 0)))),
                "median_enriched_log2fc": float(pd.to_numeric(enriched["log2fc"], errors="coerce").median()) if len(enriched) else 0.0,
                "max_enriched_log2fc": float(pd.to_numeric(enriched["log2fc"], errors="coerce").max()) if len(enriched) else 0.0,
                "median_enriched_qvalue": float(pd.to_numeric(enriched["qvalue"], errors="coerce").median()) if len(enriched) else 1.0,
                "n_llw_matches_in_post": int(llw_row.get("n_llw_matches_in_post", 0)),
                "n_llw_matches_in_enriched_clonotypes": int(llw_row.get("n_llw_matches_in_enriched_clonotypes", 0)),
                "fraction_llw_matches_recovered": float(llw_row.get("fraction_llw_matches_recovered", 0.0)),
                "n_clusters_with_llw": int(llw_row.get("n_clusters_with_llw", 0)),
                "n_enriched_clusters_with_llw": int(llw_row.get("n_enriched_clusters_with_llw", 0)),
                "fraction_llw_clusters_enriched": float(llw_row.get("fraction_llw_clusters_enriched", 0.0)),
                "lfc_mass_shift": compute_requested_lfc_mass_shift(frame),
                "median_enriched_cluster_size": median_size,
                "fraction_enriched_clusters_size_le_3": fraction_tiny,
                "max_enriched_cluster_size": max_size,
                "fragmentation_score": fragmentation_score,
            }
        )

    donor_run_summary = pd.DataFrame(donor_rows).sort_values(["donor", "config_id"]).reset_index(drop=True)
    donor_run_summary.to_csv(paths.donor_run_summary_path, sep="\t", index=False)
    llw_summary.to_csv(paths.llw_summary_path, sep="\t", index=False)
    return {
        "cluster_summary": cluster_summary,
        "clonotype_matches": clonotype_matches,
        "donor_run_summary": donor_run_summary,
        "llw_summary": llw_summary,
    }


def compute_yfv_overlap(config_path: str | Path, outdir: str | Path) -> pd.DataFrame:
    _config = load_benchmark_config(config_path)
    paths = ensure_output_dirs(outdir)
    if not paths.clonotype_match_path.exists() or not paths.cluster_summary_path.exists():
        annotate_yfv_llw(config_path, outdir)

    clonotype_matches = pd.read_csv(paths.clonotype_match_path, sep="\t") if paths.clonotype_match_path.exists() else pd.DataFrame()
    cluster_summary = pd.read_csv(paths.cluster_summary_path, sep="\t") if paths.cluster_summary_path.exists() else pd.DataFrame()
    if clonotype_matches.empty or cluster_summary.empty:
        empty = _empty_frame(OVERLAP_SUMMARY_COLUMNS)
        empty.to_csv(paths.overlap_summary_path, sep="\t", index=False)
        update_config_summaries(outdir)
        return empty

    enriched_clusters = (
        cluster_summary.loc[cluster_summary["is_enriched"], ["run_id", "cluster_id"]]
        .drop_duplicates()
        .assign(is_enriched_cluster=True)
    )
    matches = clonotype_matches.merge(enriched_clusters, on=["run_id", "cluster_id"], how="left")
    matches["is_enriched_cluster"] = matches["is_enriched_cluster"].fillna(False).astype(bool)
    post = matches.loc[matches["sample_label"].eq("sample")].copy()

    rows: list[dict[str, object]] = []
    for config_id, config_frame in post.groupby("config_id", sort=True):
        donor_sets: dict[str, set[str]] = {}
        enriched_sets: dict[str, set[str]] = {}
        for donor, donor_frame in config_frame.groupby("donor", sort=True):
            donor_sets[donor] = set(donor_frame["cdr3"].dropna().astype(str))
            enriched_sets[donor] = set(donor_frame.loc[donor_frame["is_enriched_cluster"], "cdr3"].dropna().astype(str))

        for donor_i, donor_j in combinations(sorted(donor_sets), 2):
            raw_i = donor_sets[donor_i]
            raw_j = donor_sets[donor_j]
            redcea_i = enriched_sets[donor_i]
            redcea_j = enriched_sets[donor_j]
            raw_overlap_count = len(raw_i & raw_j)
            redcea_overlap_count = len(redcea_i & redcea_j)
            raw_fraction = compute_overlap_fraction(raw_i, raw_j)
            redcea_fraction = compute_overlap_fraction(redcea_i, redcea_j)
            pair_key = "-".join(sorted([donor_i, donor_j]))
            rows.append(
                {
                    "config_id": config_id,
                    "donor_i": donor_i,
                    "donor_j": donor_j,
                    "is_twin_pair": pair_key in TWIN_PAIRS,
                    "raw_day15_overlap_count": raw_overlap_count,
                    "raw_day15_overlap_fraction": raw_fraction,
                    "redcea_overlap_count": redcea_overlap_count,
                    "redcea_overlap_fraction": redcea_fraction,
                    "overlap_fold_change": float(redcea_fraction / raw_fraction) if raw_fraction > 0 else 0.0,
                }
            )

    overlap_summary = pd.DataFrame(rows, columns=OVERLAP_SUMMARY_COLUMNS).sort_values(["config_id", "donor_i", "donor_j"]).reset_index(drop=True)
    overlap_summary.to_csv(paths.overlap_summary_path, sep="\t", index=False)
    update_config_summaries(outdir)
    return overlap_summary


def update_config_summaries(outdir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = ensure_output_dirs(outdir)
    donor_run_summary = pd.read_csv(paths.donor_run_summary_path, sep="\t") if paths.donor_run_summary_path.exists() else pd.DataFrame()
    overlap_summary = pd.read_csv(paths.overlap_summary_path, sep="\t") if paths.overlap_summary_path.exists() else pd.DataFrame()

    if donor_run_summary.empty:
        empty_config = _empty_frame(CONFIG_SUMMARY_COLUMNS)
        empty_best = _empty_frame(BEST_CONFIG_COLUMNS)
        empty_config.to_csv(paths.config_summary_path, sep="\t", index=False)
        empty_best.to_csv(paths.best_config_summary_path, sep="\t", index=False)
        return empty_config, empty_best

    config_rows: list[dict[str, object]] = []
    overlap_lookup = overlap_summary.groupby("config_id", sort=True) if not overlap_summary.empty else []
    overlap_map = {config_id: frame for config_id, frame in overlap_lookup}
    for config_id, frame in donor_run_summary.groupby("config_id", sort=True):
        overlap_frame = overlap_map.get(config_id, pd.DataFrame())
        config_rows.append(
            {
                "config_id": config_id,
                "algorithm": frame["algorithm"].iloc[0],
                "hyperparameters": frame["hyperparameters"].iloc[0],
                "n_donors_completed": int(frame["donor"].nunique()),
                "median_n_enriched_clusters": float(frame["n_enriched_clusters"].median()),
                "median_fraction_sample_retained": float(frame["fraction_sample_clonotypes_retained"].median()),
                "median_fraction_llw_recovered": float(frame["fraction_llw_matches_recovered"].median()),
                "median_fraction_llw_clusters_enriched": float(frame["fraction_llw_clusters_enriched"].median()),
                "median_raw_overlap_fraction": float(overlap_frame["raw_day15_overlap_fraction"].median()) if len(overlap_frame) else np.nan,
                "median_redcea_overlap_fraction": float(overlap_frame["redcea_overlap_fraction"].median()) if len(overlap_frame) else np.nan,
                "median_overlap_fold_change": float(overlap_frame["overlap_fold_change"].median()) if len(overlap_frame) else np.nan,
                "median_lfc_mass_shift": float(frame["lfc_mass_shift"].median()),
                "fragmentation_score": float(frame["fragmentation_score"].median()),
            }
        )
    config_summary = pd.DataFrame(config_rows, columns=CONFIG_SUMMARY_COLUMNS).sort_values("config_id").reset_index(drop=True)
    config_summary.to_csv(paths.config_summary_path, sep="\t", index=False)

    ranking_specs = [
        ("median_lfc_mass_shift", False),
        ("median_overlap_fold_change", False),
        ("median_fraction_sample_retained", False),
        ("fragmentation_score", True),
    ]
    best_rows: list[dict[str, object]] = []
    for metric, ascending in ranking_specs:
        valid = config_summary.loc[config_summary[metric].notna()].copy()
        if valid.empty:
            continue
        best = valid.sort_values(metric, ascending=ascending).iloc[0]
        best_rows.append(
            {
                "ranking_metric": metric,
                "config_id": best["config_id"],
                "algorithm": best["algorithm"],
                "hyperparameters": best["hyperparameters"],
                "metric_value": best[metric],
            }
        )
    best_summary = pd.DataFrame(best_rows, columns=BEST_CONFIG_COLUMNS)
    best_summary.to_csv(paths.best_config_summary_path, sep="\t", index=False)
    return config_summary, best_summary


def _selected_config(config_summary: pd.DataFrame) -> str | None:
    if config_summary.empty:
        return None
    sort_df = config_summary.copy()
    sort_df["median_overlap_fold_change"] = sort_df["median_overlap_fold_change"].fillna(-np.inf)
    sort_df = sort_df.sort_values(
        ["median_lfc_mass_shift", "median_overlap_fold_change", "median_fraction_sample_retained", "fragmentation_score"],
        ascending=[False, False, False, True],
    )
    return str(sort_df.iloc[0]["config_id"])


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _placeholder_figure(stem: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    _save_figure(fig, stem)


def plot_yfv_benchmark(config_path: str | Path, outdir: str | Path) -> str | None:
    config = load_benchmark_config(config_path)
    paths = ensure_output_dirs(outdir)
    donor_run_summary = pd.read_csv(paths.donor_run_summary_path, sep="\t") if paths.donor_run_summary_path.exists() else pd.DataFrame()
    cluster_summary = pd.read_csv(paths.cluster_summary_path, sep="\t") if paths.cluster_summary_path.exists() else pd.DataFrame()
    llw_summary = pd.read_csv(paths.llw_summary_path, sep="\t") if paths.llw_summary_path.exists() else pd.DataFrame()
    overlap_summary = pd.read_csv(paths.overlap_summary_path, sep="\t") if paths.overlap_summary_path.exists() else pd.DataFrame()
    config_summary = pd.read_csv(paths.config_summary_path, sep="\t") if paths.config_summary_path.exists() else pd.DataFrame()

    selected_config = _selected_config(config_summary)

    fig1 = paths.figures_dir / "yfv_enriched_clusters_by_donor"
    fig2 = paths.figures_dir / "yfv_llw_recovery_by_donor"
    fig3 = paths.figures_dir / "yfv_redcea_vs_raw_overlap"
    fig4 = paths.figures_dir / "yfv_volcano_grid"

    if selected_config is None:
        for stem, title in [
            (fig1, "Enriched Clusters by Donor"),
            (fig2, "LLW Recovery by Donor"),
            (fig3, "Raw vs RedCEA Overlap"),
            (fig4, "YFV Volcano Grid"),
        ]:
            _placeholder_figure(stem, title, "No successful YFV runs were available.")
        return None

    donor_order = list(dict(config.get("donors", {})).keys())

    donor_plot = donor_run_summary.loc[donor_run_summary["config_id"].astype(str) == selected_config].copy()
    if donor_plot.empty:
        _placeholder_figure(fig1, "Enriched Clusters by Donor", "No donor summaries were available for the selected config.")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=donor_plot, x="donor", y="n_enriched_clusters", order=donor_order, color="#d97a38", ax=ax)
        ax.set_xlabel("Donor")
        ax.set_ylabel("Enriched clusters")
        ax.set_title("Enriched clusters by donor")
        _save_figure(fig, fig1)

    llw_plot = llw_summary.loc[llw_summary["config_id"].astype(str) == selected_config].copy()
    if llw_plot.empty:
        _placeholder_figure(fig2, "LLW Recovery by Donor", "No LLW summaries were available for the selected config.")
    else:
        llw_long = llw_plot.melt(
            id_vars=["donor"],
            value_vars=["fraction_llw_matches_recovered", "fraction_llw_clusters_enriched"],
            var_name="metric",
            value_name="value",
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=llw_long, x="donor", y="value", hue="metric", order=donor_order, ax=ax)
        ax.set_xlabel("Donor")
        ax.set_ylabel("Fraction")
        ax.set_title("LLW recovery by donor")
        _save_figure(fig, fig2)

    overlap_plot = overlap_summary.loc[overlap_summary["config_id"].astype(str) == selected_config].copy()
    if overlap_plot.empty:
        _placeholder_figure(fig3, "Raw vs RedCEA Overlap", "No overlap summaries were available for the selected config.")
    else:
        fig, ax = plt.subplots(figsize=(9, 6))
        for _, row in overlap_plot.iterrows():
            x_values = [0, 1]
            y_values = [row["raw_day15_overlap_fraction"], row["redcea_overlap_fraction"]]
            color = "#c44e52" if bool(row["is_twin_pair"]) else "#4c72b0"
            ax.plot(x_values, y_values, marker="o", color=color, alpha=0.8)
            ax.text(1.02, y_values[1], "{0}-{1}".format(row["donor_i"], row["donor_j"]), fontsize=9, va="center")
        ax.set_xticks([0, 1], ["Raw day-15", "RedCEA-enriched"])
        ax.set_ylabel("Overlap fraction")
        ax.set_title("Raw vs RedCEA overlap")
        _save_figure(fig, fig3)

    volcano_plot = cluster_summary.loc[cluster_summary["config_id"].astype(str) == selected_config].copy()
    if volcano_plot.empty:
        _placeholder_figure(fig4, "YFV Volcano Grid", "No cluster summaries were available for the selected config.")
    else:
        volcano_plot["neg_log10_q"] = -np.log10(pd.to_numeric(volcano_plot["qvalue"], errors="coerce").clip(lower=QVALUE_CLIP_MIN, upper=1.0))
        volcano_plot["category"] = "other"
        volcano_plot.loc[volcano_plot["has_llw_match"], "category"] = "llw"
        volcano_plot.loc[volcano_plot["is_enriched"], "category"] = "enriched"
        volcano_plot.loc[volcano_plot["is_enriched"] & volcano_plot["has_llw_match"], "category"] = "enriched_llw"

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
        axes_flat = axes.ravel()
        palette = {
            "other": "#c8d3dc",
            "enriched": "#d97a38",
            "llw": "#4c72b0",
            "enriched_llw": "#7a3b99",
        }
        q_line = -np.log10(float(config.get("enrichment", {}).get("qvalue_threshold", 0.05)))
        for ax, donor in zip(axes_flat, donor_order):
            donor_frame = volcano_plot.loc[volcano_plot["donor"].astype(str) == donor].copy()
            if donor_frame.empty:
                ax.axis("off")
                ax.set_title("{0} (no data)".format(donor))
                continue
            sns.scatterplot(
                data=donor_frame,
                x="log2fc",
                y="neg_log10_q",
                hue="category",
                palette=palette,
                ax=ax,
                s=70,
                edgecolor="black",
                linewidth=0.3,
            )
            ax.axvline(0.0, linestyle="--", color="black", linewidth=1.0)
            ax.axhline(q_line, linestyle="--", color="black", linewidth=1.0)
            ax.set_title(donor)
            ax.set_xlabel("log2FC(post/pre)")
            ax.set_ylabel("-log10(qvalue)")
            ax.legend().remove()
        for ax in axes_flat[len(donor_order):]:
            ax.axis("off")
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=4)
        fig.suptitle("YFV volcano grid")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save_figure(fig, fig4)

    return selected_config


def _table_or_message(path: Path, message: str) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return message
    frame = pd.read_csv(path, sep="\t")
    if frame.empty:
        return message
    return "```text\n{0}\n```".format(frame.to_string(index=False))


def write_yfv_benchmark_summary(config_path: str | Path, outdir: str | Path) -> Path:
    config = load_benchmark_config(config_path)
    paths = ensure_output_dirs(outdir)
    config_summary = pd.read_csv(paths.config_summary_path, sep="\t") if paths.config_summary_path.exists() else pd.DataFrame()
    donor_run_summary = pd.read_csv(paths.donor_run_summary_path, sep="\t") if paths.donor_run_summary_path.exists() else pd.DataFrame()
    best_config_summary = pd.read_csv(paths.best_config_summary_path, sep="\t") if paths.best_config_summary_path.exists() else pd.DataFrame()
    run_metadata = pd.read_csv(paths.run_metadata_path, sep="\t") if paths.run_metadata_path.exists() else pd.DataFrame()
    selected_config = _selected_config(config_summary)

    completed_runs = int(run_metadata["status"].eq("success").sum()) if len(run_metadata) and "status" in run_metadata.columns else 0
    failed_runs = int(run_metadata["status"].eq("error").sum()) if len(run_metadata) and "status" in run_metadata.columns else 0
    warnings: list[str] = []
    if failed_runs:
        warnings.append("Some RedCEA runs failed. Check `logs/clustering_runs.tsv` for tracebacks.")
    if paths.llw_reference_path.exists():
        llw_ref = pd.read_csv(paths.llw_reference_path, sep="\t")
        if llw_ref.empty:
            warnings.append("The configured LLW/VDJdb table produced zero LLW reference rows after filtering.")
    if selected_config is None:
        warnings.append("No successful config summary was available, so figures and donor-level narrative are placeholders.")

    lines = [
        "# YFV RedCEA Benchmark Summary",
        "",
        "## Purpose",
        "",
        "Assess whether donor-matched RedCEA enrichment finds post-vaccination structure, recovers LLW-linked clonotypes as partial validation, and increases cross-donor sharing relative to raw day-15 repertoires.",
        "",
        "## Inputs",
        "",
        "Config path: `{0}`".format(config_path),
        "LLW/VDJdb path: `{0}`".format(dict(config.get("vdjdb", {})).get("path", "")),
        "",
        "Donor pre/post files:",
        "",
    ]
    for donor_id, donor_cfg in dict(config.get("donors", {})).items():
        lines.append("- `{0}`: pre=`{1}`, post=`{2}`".format(donor_id, donor_cfg.get("pre", ""), donor_cfg.get("post", "")))

    lines.extend(
        [
            "",
            "## Pipeline",
            "",
            "1. `python scripts/yfv/1_prepare_yfv_inputs.py --config {0} --outdir {1}`".format(config_path, outdir),
            "2. `python scripts/yfv/2_run_yfv_redcea_grid.py --config {0} --outdir {1}`".format(config_path, outdir),
            "3. `python scripts/yfv/3_annotate_yfv_llw.py --config {0} --outdir {1}`".format(config_path, outdir),
            "4. `python scripts/yfv/4_summarize_yfv_benchmark.py --config {0} --outdir {1}`".format(config_path, outdir),
            "5. `python scripts/yfv/5_compute_yfv_overlap.py --config {0} --outdir {1}`".format(config_path, outdir),
            "6. `python scripts/yfv/6_plot_yfv_benchmark.py --config {0} --outdir {1}`".format(config_path, outdir),
            "7. `python scripts/yfv/7_write_yfv_benchmark_summary.py --config {0} --outdir {1}`".format(config_path, outdir),
            "",
            "## Output Map",
            "",
            "- `runs/`: one RedCEA run directory per donor/config plus assignment parquets.",
            "- `summaries/yfv_donor_run_summary.tsv`: donor-level run summary with enrichment, LLW, and fragmentation diagnostics.",
            "- `summaries/yfv_cluster_summary.tsv`: one row per non-noise candidate cluster, including non-significant clusters.",
            "- `summaries/yfv_llw_summary.tsv`: donor/config LLW recovery summary.",
            "- `summaries/yfv_overlap_summary.tsv`: donor-pair overlap before and after RedCEA enrichment filtering.",
            "- `summaries/yfv_config_summary.tsv`: config-level donor medians plus overlap and fragmentation summaries.",
            "- `summaries/yfv_best_config_summary.tsv`: top configs by each key selection metric.",
            "- `figures/`: manuscript-facing benchmark figures.",
            "- `logs/`: manifests, normalized AIRR inputs, LLW match tables, and run metadata.",
            "",
            "## Figures",
            "",
            "- `figures/yfv_enriched_clusters_by_donor.(png|pdf)`: enriched cluster count per donor for the selected config.",
            "- `figures/yfv_llw_recovery_by_donor.(png|pdf)`: LLW recovery fractions per donor for the selected config.",
            "- `figures/yfv_redcea_vs_raw_overlap.(png|pdf)`: paired raw-vs-filtered overlap comparison across donor pairs.",
            "- `figures/yfv_volcano_grid.(png|pdf)`: 2x3 volcano grid, one donor per panel, with enriched and LLW-highlighted clusters.",
            "",
            "## Run Status",
            "",
            "- Completed runs: `{0}`".format(completed_runs),
            "- Failed runs: `{0}`".format(failed_runs),
            "",
            "## Best Configs",
            "",
            _table_or_message(paths.best_config_summary_path, "No best-config table is available yet."),
            "",
            "## Selected Config",
            "",
            "`{0}`".format(selected_config or "none"),
            "",
            "## Donor-Level Result Summary",
            "",
        ]
    )

    if selected_config is not None and not donor_run_summary.empty:
        selected_rows = donor_run_summary.loc[donor_run_summary["config_id"].astype(str) == selected_config]
        if not selected_rows.empty:
            lines.append("```text")
            lines.append(selected_rows.to_string(index=False))
            lines.append("```")
            lines.append("")
    else:
        lines.append("No donor-level summary is available yet.")
        lines.append("")

    lines.extend(
        [
            "## Warnings",
            "",
        ]
    )
    if warnings:
        for warning in warnings:
            lines.append("- {0}".format(warning))
    else:
        lines.append("- No benchmark warnings were recorded.")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- LLW/VDJdb is partial external validation only.",
            "- Absence of an LLW match is not evidence of a false positive.",
            "- Config selection should balance volcano behavior, retained sample fraction, overlap gain, and fragmentation, not LLW recovery alone.",
            "",
        ]
    )

    paths.benchmark_summary_path.write_text("\n".join(lines), encoding="utf-8")
    return paths.benchmark_summary_path
