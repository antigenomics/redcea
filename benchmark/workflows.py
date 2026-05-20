from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from benchmark.prepare_datasets import align_yfv_embedding_with_airr
from benchmark.evaluation import (
    compute_cross_donor_overlap,
    compute_known_yfv_recovery,
    compute_vdjdb_metrics,
    compute_yfv_cluster_enrichment,
    summarize_yfv_enrichment,
)
from benchmark.plotting import (
    plot_cross_donor_overlap,
    plot_density_by_length,
    plot_final_method_summary,
    plot_vdjdb_benchmark,
    plot_vdjdb_signal_concentration,
    plot_yfv_enrichment_summary,
    plot_yfv_known_recovery_heatmap,
)
from benchmark.run_benchmark import consolidate_run_metadata
from benchmark.runner import extract_embeddings


def ensure_output_dirs() -> None:
    for path in [
        Path("results/clustering_assignments"),
        Path("results/run_metadata"),
        Path("results/enrichment"),
        Path("results/metrics"),
        Path("results/density"),
        Path("figures/clustering_strategy"),
        Path("reports"),
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_assignment_tables(
    assignments_dir: str | Path = "results/clustering_assignments",
    *,
    metadata_parts_dir: str | Path = "results/run_metadata/clustering_run_parts",
    metadata_path: str | Path = "results/run_metadata/clustering_runs.tsv",
    only_success: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()
    run_metadata = consolidate_run_metadata(
        metadata_parts_dir=metadata_parts_dir,
        metadata_path=metadata_path,
    )
    successful_run_ids: set[str]
    if only_success and len(run_metadata):
        successful_run_ids = set(run_metadata.loc[run_metadata["status"] == "success", "run_id"].astype(str))
    else:
        successful_run_ids = set(run_metadata["run_id"].astype(str)) if len(run_metadata) else set()
    frames = []
    for path in sorted(Path(assignments_dir).glob("*.parquet")):
        if only_success and successful_run_ids and path.stem not in successful_run_ids:
            continue
        frames.append(pd.read_parquet(path))
    assignments = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return assignments, run_metadata


def compute_density_by_length(
    processed_dir: str | Path = "data/processed",
    *,
    k: int = 5,
) -> pd.DataFrame:
    processed_dir = Path(processed_dir)
    datasets = {
        "VDJdb / GLC": pd.read_parquet(processed_dir / "vdjdb_glc.parquet"),
        "VDJdb / YLQ": pd.read_parquet(processed_dir / "vdjdb_ylq.parquet"),
    }
    yfv_manifest_path = processed_dir / "yfv_repertoires_manifest.tsv"
    if yfv_manifest_path.exists():
        yfv_manifest = pd.read_csv(yfv_manifest_path, sep="\t")
        for row in yfv_manifest.itertuples(index=False):
            donor_id = str(row.donor_id)
            sample = align_yfv_embedding_with_airr(
                donor_id,
                sample_label="sample",
                embedding_path=Path(row.sample_embedding_path),
                airr_path=Path(row.sample_airr_path),
            )
            background = align_yfv_embedding_with_airr(
                donor_id,
                sample_label="background",
                embedding_path=Path(row.background_embedding_path),
                airr_path=Path(row.background_airr_path),
            )
            datasets[f"YFV / {donor_id} / sample"] = sample
            datasets[f"YFV / {donor_id} / background"] = background

    rows: list[dict[str, object]] = []
    for facet_label, frame in datasets.items():
        if len(frame) < 2:
            continue
        embeddings = extract_embeddings(frame)
        n_neighbors = min(max(2, int(k) + 1), len(frame))
        distances, _ = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(embeddings).kneighbors(embeddings)
        kth_index = min(int(k), distances.shape[1] - 1)
        for cdr3_length, knn_distance in zip(frame["cdr3_length"].astype(int), distances[:, kth_index]):
            rows.append(
                {
                    "facet_label": facet_label,
                    "cdr3_length": int(cdr3_length),
                    "knn_distance": float(knn_distance),
                    "k": int(k),
                }
            )
    return pd.DataFrame(rows)


def run_density_analysis(
    processed_dir: str | Path = "data/processed",
    *,
    density_path: str | Path = "results/density/knn_distance_by_length.tsv",
    figure_stem: str | Path = "figures/clustering_strategy/fig1_density_by_length",
    k: int = 5,
) -> pd.DataFrame:
    ensure_output_dirs()
    density_df = compute_density_by_length(processed_dir, k=k)
    density_df.to_csv(density_path, sep="\t", index=False)
    if len(density_df):
        plot_density_by_length(density_df, figure_stem)
    return density_df


def run_vdjdb_evaluation(
    *,
    assignments_dir: str | Path = "results/clustering_assignments",
    metadata_parts_dir: str | Path = "results/run_metadata/clustering_run_parts",
    metadata_path: str | Path = "results/run_metadata/clustering_runs.tsv",
    metrics_path: str | Path = "results/metrics/vdjdb_clustering_metrics.tsv",
    fig2_stem: str | Path = "figures/clustering_strategy/fig2_vdjdb_f1_precision_recall",
    fig3_stem: str | Path = "figures/clustering_strategy/fig3_vdjdb_cluster_concentration",
) -> pd.DataFrame:
    assignments, run_metadata = load_assignment_tables(
        assignments_dir=assignments_dir,
        metadata_parts_dir=metadata_parts_dir,
        metadata_path=metadata_path,
        only_success=True,
    )
    if assignments.empty or "dataset_mode" not in assignments.columns:
        metrics_df = pd.DataFrame()
        metrics_df.to_csv(metrics_path, sep="\t", index=False)
        return metrics_df
    vdjdb_assignments = assignments.loc[assignments["dataset_mode"] == "vdjdb"].copy()
    metrics_df = compute_vdjdb_metrics(vdjdb_assignments, run_metadata)
    metrics_df.to_csv(metrics_path, sep="\t", index=False)
    if len(metrics_df):
        plot_vdjdb_benchmark(metrics_df, fig2_stem)
        plot_vdjdb_signal_concentration(metrics_df, fig3_stem)
    return metrics_df


def run_yfv_enrichment_evaluation(
    *,
    assignments_dir: str | Path = "results/clustering_assignments",
    metadata_parts_dir: str | Path = "results/run_metadata/clustering_run_parts",
    metadata_path: str | Path = "results/run_metadata/clustering_runs.tsv",
    enrichment_path: str | Path = "results/enrichment/yfv_cluster_enrichment.tsv",
    metrics_path: str | Path = "results/metrics/yfv_enrichment_metrics.tsv",
    fig4_stem: str | Path = "figures/clustering_strategy/fig4_yfv_enrichment_output",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments, _run_metadata = load_assignment_tables(
        assignments_dir=assignments_dir,
        metadata_parts_dir=metadata_parts_dir,
        metadata_path=metadata_path,
        only_success=True,
    )
    if assignments.empty or "dataset_mode" not in assignments.columns:
        enrichment_df = pd.DataFrame()
        summary_df = pd.DataFrame()
        enrichment_df.to_csv(enrichment_path, sep="\t", index=False)
        summary_df.to_csv(metrics_path, sep="\t", index=False)
        return enrichment_df, summary_df
    yfv_assignments = assignments.loc[assignments["dataset_mode"] == "yfv"].copy()
    enrichment_df = compute_yfv_cluster_enrichment(yfv_assignments)
    enrichment_df.to_csv(enrichment_path, sep="\t", index=False)
    summary_df = summarize_yfv_enrichment(enrichment_df)
    summary_df.to_csv(metrics_path, sep="\t", index=False)
    if len(summary_df):
        plot_yfv_enrichment_summary(summary_df, fig4_stem)
    return enrichment_df, summary_df


def run_yfv_known_clonotype_evaluation(
    *,
    assignments_dir: str | Path = "results/clustering_assignments",
    metadata_parts_dir: str | Path = "results/run_metadata/clustering_run_parts",
    metadata_path: str | Path = "results/run_metadata/clustering_runs.tsv",
    known_yfv_path: str | Path = "data/processed/known_yfv_vdjdb_clonotypes.tsv",
    enrichment_path: str | Path = "results/enrichment/yfv_cluster_enrichment.tsv",
    output_path: str | Path = "results/metrics/yfv_known_clonotype_recovery.tsv",
    fig5_stem: str | Path = "figures/clustering_strategy/fig5_yfv_known_clonotype_recovery",
) -> pd.DataFrame:
    assignments, _run_metadata = load_assignment_tables(
        assignments_dir=assignments_dir,
        metadata_parts_dir=metadata_parts_dir,
        metadata_path=metadata_path,
        only_success=True,
    )
    if assignments.empty or "dataset_mode" not in assignments.columns:
        recovery_df = pd.DataFrame()
        recovery_df.to_csv(output_path, sep="\t", index=False)
        return recovery_df
    enrichment_df = pd.read_csv(enrichment_path, sep="\t")
    known_yfv = pd.read_csv(known_yfv_path, sep="\t")
    yfv_assignments = assignments.loc[assignments["dataset_mode"] == "yfv"].copy()
    recovery_df = compute_known_yfv_recovery(yfv_assignments, enrichment_df, known_yfv)
    if len(recovery_df):
        recovery_df = recovery_df.copy()
        recovery_df["_candidate_numeric"] = recovery_df["is_candidate_clustered"].astype(float)
        recovery_df["_significant_numeric"] = recovery_df["is_in_significant_cluster"].astype(float)
        grouped = recovery_df.groupby(["run_id", "donor_id", "method"])
        recovery_df["candidate_level_recovery_rate"] = grouped["_candidate_numeric"].transform("mean").astype(float)
        recovery_df["enrichment_level_recovery_rate"] = grouped["_significant_numeric"].transform("mean").astype(float)
        recovery_df = recovery_df.drop(columns=["_candidate_numeric", "_significant_numeric"])
    else:
        recovery_df["candidate_level_recovery_rate"] = []
        recovery_df["enrichment_level_recovery_rate"] = []
    recovery_df.to_csv(output_path, sep="\t", index=False)
    if len(recovery_df):
        plot_yfv_known_recovery_heatmap(recovery_df, fig5_stem)
    return recovery_df


def run_cross_donor_overlap_evaluation(
    *,
    assignments_dir: str | Path = "results/clustering_assignments",
    metadata_parts_dir: str | Path = "results/run_metadata/clustering_run_parts",
    metadata_path: str | Path = "results/run_metadata/clustering_runs.tsv",
    enrichment_path: str | Path = "results/enrichment/yfv_cluster_enrichment.tsv",
    output_path: str | Path = "results/metrics/yfv_cross_donor_overlap.tsv",
    fig6_stem: str | Path = "figures/clustering_strategy/fig6_yfv_cross_donor_overlap",
) -> pd.DataFrame:
    assignments, _run_metadata = load_assignment_tables(
        assignments_dir=assignments_dir,
        metadata_parts_dir=metadata_parts_dir,
        metadata_path=metadata_path,
        only_success=True,
    )
    if assignments.empty or "dataset_mode" not in assignments.columns:
        overlap_df = pd.DataFrame()
        overlap_df.to_csv(output_path, sep="\t", index=False)
        return overlap_df
    enrichment_df = pd.read_csv(enrichment_path, sep="\t")
    yfv_assignments = assignments.loc[assignments["dataset_mode"] == "yfv"].copy()
    overlap_df = compute_cross_donor_overlap(yfv_assignments, enrichment_df)
    overlap_df.to_csv(output_path, sep="\t", index=False)
    if len(overlap_df):
        plot_cross_donor_overlap(overlap_df, fig6_stem)
    return overlap_df


def build_final_method_comparison(
    *,
    vdjdb_metrics_path: str | Path = "results/metrics/vdjdb_clustering_metrics.tsv",
    yfv_metrics_path: str | Path = "results/metrics/yfv_enrichment_metrics.tsv",
    yfv_recovery_path: str | Path = "results/metrics/yfv_known_clonotype_recovery.tsv",
    overlap_path: str | Path = "results/metrics/yfv_cross_donor_overlap.tsv",
) -> pd.DataFrame:
    def _safe_read(path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path, sep="\t")

    vdjdb_metrics = _safe_read(vdjdb_metrics_path)
    yfv_metrics = _safe_read(yfv_metrics_path)
    yfv_recovery = _safe_read(yfv_recovery_path)
    overlap_df = _safe_read(overlap_path)

    if vdjdb_metrics.empty and yfv_metrics.empty and yfv_recovery.empty and overlap_df.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "parameter_json",
                "vdjdb_mean_f1",
                "vdjdb_mean_precision",
                "vdjdb_mean_recall",
                "vdjdb_mean_purity",
                "yfv_mean_significant_clusters",
                "yfv_mean_retained_fraction",
                "yfv_mean_background_contamination",
                "yfv_mean_weighted_enrichment_score",
                "yfv_mean_candidate_recovery_rate",
                "yfv_mean_enrichment_recovery_rate",
                "yfv_mean_overlap_fold_change",
            ]
        )

    if len(vdjdb_metrics):
        vdjdb_summary = (
            vdjdb_metrics.groupby(["method", "parameter_json"])
            .agg(
                vdjdb_mean_f1=("f1", "mean"),
                vdjdb_mean_precision=("precision", "mean"),
                vdjdb_mean_recall=("recall", "mean"),
                vdjdb_mean_purity=("weighted_cluster_purity", "mean"),
            )
            .reset_index()
        )
    else:
        vdjdb_summary = pd.DataFrame(
            columns=["method", "parameter_json", "vdjdb_mean_f1", "vdjdb_mean_precision", "vdjdb_mean_recall", "vdjdb_mean_purity"]
        )
    if len(yfv_metrics):
        yfv_summary = (
            yfv_metrics.groupby(["method", "parameter_json"])
            .agg(
                yfv_mean_significant_clusters=("number_of_significant_enriched_clusters", "mean"),
                yfv_mean_retained_fraction=("retained_fraction", "mean"),
                yfv_mean_background_contamination=("background_contamination", "mean"),
                yfv_mean_weighted_enrichment_score=("weighted_enrichment_score", "mean"),
            )
            .reset_index()
        )
    else:
        yfv_summary = pd.DataFrame(
            columns=[
                "method",
                "parameter_json",
                "yfv_mean_significant_clusters",
                "yfv_mean_retained_fraction",
                "yfv_mean_background_contamination",
                "yfv_mean_weighted_enrichment_score",
            ]
        )
    if len(yfv_recovery):
        yfv_recovery = yfv_recovery.copy()
        yfv_recovery["candidate_level_recovery_rate"] = pd.to_numeric(
            yfv_recovery["candidate_level_recovery_rate"], errors="coerce"
        )
        yfv_recovery["enrichment_level_recovery_rate"] = pd.to_numeric(
            yfv_recovery["enrichment_level_recovery_rate"], errors="coerce"
        )
        recovery_summary = (
            yfv_recovery.groupby(["method", "parameter_json"])
            .agg(
                yfv_mean_candidate_recovery_rate=("candidate_level_recovery_rate", "mean"),
                yfv_mean_enrichment_recovery_rate=("enrichment_level_recovery_rate", "mean"),
            )
            .reset_index()
        )
    else:
        recovery_summary = pd.DataFrame(columns=["method", "parameter_json", "yfv_mean_candidate_recovery_rate", "yfv_mean_enrichment_recovery_rate"])
    if len(overlap_df):
        overlap_summary = (
            overlap_df.groupby(["method", "parameter_json"])
            .agg(
                yfv_mean_overlap_fold_change=("overlap_fold_change", "mean"),
            )
            .reset_index()
        )
    else:
        overlap_summary = pd.DataFrame(columns=["method", "parameter_json", "yfv_mean_overlap_fold_change"])

    final = vdjdb_summary.merge(yfv_summary, on=["method", "parameter_json"], how="outer")
    final = final.merge(recovery_summary, on=["method", "parameter_json"], how="left")
    final = final.merge(overlap_summary, on=["method", "parameter_json"], how="left")
    sort_columns = [column for column in ["vdjdb_mean_f1", "yfv_mean_enrichment_recovery_rate", "yfv_mean_weighted_enrichment_score"] if column in final.columns]
    if sort_columns:
        final = final.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    return final.reset_index(drop=True)


def write_summary_report(
    comparison_df: pd.DataFrame,
    *,
    output_path: str | Path = "reports/clustering_strategy_summary.md",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Clustering Strategy Summary",
        "",
    ]
    if comparison_df.empty:
        lines.extend(["No successful benchmark runs were available.", ""])
    else:
        best_vdjdb = comparison_df.sort_values("vdjdb_mean_f1", ascending=False).iloc[0]
        best_yfv = comparison_df.sort_values(
            ["yfv_mean_enrichment_recovery_rate", "yfv_mean_weighted_enrichment_score"],
            ascending=[False, False],
        ).iloc[0]
        lines.extend(
            [
                f"- Best VDJdb method: `{best_vdjdb['method']}`",
                f"- Best YFV method: `{best_yfv['method']}`",
                "",
                "## Combined Table",
                "",
                "```text",
                comparison_df.to_string(index=False),
                "```",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_summary_and_method_selection(
    *,
    vdjdb_metrics_path: str | Path = "results/metrics/vdjdb_clustering_metrics.tsv",
    yfv_metrics_path: str | Path = "results/metrics/yfv_enrichment_metrics.tsv",
    yfv_recovery_path: str | Path = "results/metrics/yfv_known_clonotype_recovery.tsv",
    overlap_path: str | Path = "results/metrics/yfv_cross_donor_overlap.tsv",
    output_table_path: str | Path = "results/metrics/final_method_comparison.tsv",
    figure_stem: str | Path = "figures/clustering_strategy/final_method_summary",
    report_path: str | Path = "reports/clustering_strategy_summary.md",
) -> pd.DataFrame:
    ensure_output_dirs()
    comparison_df = build_final_method_comparison(
        vdjdb_metrics_path=vdjdb_metrics_path,
        yfv_metrics_path=yfv_metrics_path,
        yfv_recovery_path=yfv_recovery_path,
        overlap_path=overlap_path,
    )
    comparison_df.to_csv(output_table_path, sep="\t", index=False)
    if len(comparison_df):
        plot_final_method_summary(comparison_df, figure_stem)
    write_summary_report(comparison_df, output_path=report_path)
    return comparison_df
