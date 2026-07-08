from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sns, "set_theme"):
    sns.set_theme(style="whitegrid", context="talk")
else:
    sns.set_style("whitegrid")


def _save(fig: plt.Figure, output_stem: str | Path) -> None:
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")


def plot_density_by_length(df: pd.DataFrame, output_stem: str | Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="cdr3_length", y="knn_distance", hue="facet_label", ax=ax, fliersize=0.5)
    ax.set_xlabel("CDR3 length")
    ax.set_ylabel("Distance to k-th nearest neighbor")
    ax.set_title("Local density depends on CDR3 length")
    _save(fig, output_stem)
    return fig


def plot_vdjdb_benchmark(metrics_df: pd.DataFrame, output_stem: str | Path) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
    for ax, metric in zip(axes, ["f1", "precision", "recall"]):
        sns.barplot(data=metrics_df, x="method", y=metric, hue="epitope", ax=ax)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(metric.upper())
    _save(fig, output_stem)
    return fig


def plot_vdjdb_signal_concentration(metrics_df: pd.DataFrame, output_stem: str | Path) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
    for ax, metric in zip(axes, ["weighted_cluster_purity", "positive_cluster_concentration_top5", "positive_fragmentation"]):
        sns.barplot(data=metrics_df, x="method", y=metric, hue="epitope", ax=ax)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(metric)
    _save(fig, output_stem)
    return fig


def plot_yfv_enrichment_summary(metrics_df: pd.DataFrame, output_stem: str | Path) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.ravel()
    metrics = ["number_of_significant_enriched_clusters", "retained_fraction", "weighted_enrichment_score", "background_contamination"]
    for ax, metric in zip(axes, metrics):
        sns.boxplot(data=metrics_df, x="method", y=metric, ax=ax, color="#d9e8f5")
        sns.stripplot(data=metrics_df, x="method", y=metric, hue="donor_id", ax=ax, dodge=False, size=6)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(metric)
    _save(fig, output_stem)
    return fig


def plot_yfv_known_recovery_heatmap(recovery_df: pd.DataFrame, output_stem: str | Path) -> plt.Figure:
    plot_df = recovery_df.copy()
    plot_df["recovery_code"] = 0
    plot_df.loc[plot_df["is_candidate_clustered"], "recovery_code"] = 1
    plot_df.loc[plot_df["is_in_significant_cluster"], "recovery_code"] = 2
    plot_df["column_label"] = plot_df["donor_id"].astype(str) + " | " + plot_df["known_yfv_clonotype_id"].astype(str)
    heatmap_df = plot_df.pivot_table(index="method", columns="column_label", values="recovery_code", aggfunc="max", fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", cbar_kws={"label": "Recovery code"}, ax=ax)
    _save(fig, output_stem)
    return fig


def plot_cross_donor_overlap(overlap_df: pd.DataFrame, output_stem: str | Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=overlap_df, x="method", y="overlap_fold_change", ax=ax, color="#f3d6b3")
    sns.stripplot(data=overlap_df, x="method", y="overlap_fold_change", ax=ax, color="#8c4c10", size=6)
    ax.tick_params(axis="x", rotation=30)
    _save(fig, output_stem)
    return fig


def plot_final_method_summary(comparison_df: pd.DataFrame, output_stem: str | Path) -> plt.Figure:
    plot_df = comparison_df.copy()
    required = ["method", "vdjdb_mean_f1", "yfv_mean_enrichment_recovery_rate", "yfv_mean_weighted_enrichment_score"]
    for column in required:
        if column not in plot_df.columns:
            raise KeyError("Missing required comparison column: {0}".format(column))
    plot_df["label"] = plot_df["method"].astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
    metrics = [
        ("vdjdb_mean_f1", "Mean VDJdb F1"),
        ("yfv_mean_enrichment_recovery_rate", "Mean YFV Enrichment Recovery"),
        ("yfv_mean_weighted_enrichment_score", "Mean YFV Weighted Enrichment"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        order = (
            plot_df.groupby("label", as_index=False)[metric]
            .mean()
            .sort_values(metric, ascending=False)["label"]
            .tolist()
        )
        sns.barplot(data=plot_df, x="label", y=metric, order=order, ax=ax, color="#c9d9c1")
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel("Method")
        ax.set_title(title)
    _save(fig, output_stem)
    return fig
