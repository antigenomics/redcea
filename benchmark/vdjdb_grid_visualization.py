from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

from .notebook_displays import display_frame, display_image, display_markdown
from .plotting import _save

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sns, "set_theme"):
    sns.set_theme(style="whitegrid", context="talk")
else:
    sns.set_style("whitegrid")


METHOD_ORDER = [
    "hierarchical_leiden",
    "leiden",
    "leiden_dbscan",
    "dbscan",
    "vdbscan_leiden",
    "vdbscan",
]


def load_vdjdb_grid_metrics(grid_csv_path: str | Path) -> pd.DataFrame:
    grid_csv_path = Path(grid_csv_path)
    metrics_df = pd.read_csv(grid_csv_path)
    metrics_df = metrics_df.drop(columns=[column for column in metrics_df.columns if column.startswith("Unnamed:")], errors="ignore")
    params_df = metrics_df["parameter_json"].apply(json.loads).apply(pd.Series)
    plot_df = pd.concat([metrics_df, params_df], axis=1)
    plot_df["method"] = pd.Categorical(plot_df["method"], categories=METHOD_ORDER, ordered=True)
    plot_df["epitope"] = pd.Categorical(
        plot_df["epitope"],
        categories=sorted(plot_df["epitope"].dropna().unique().tolist()),
        ordered=True,
    )
    for column in ["cluster_min_samples", "k_neighbors", "eps_k_neighbors", "leiden_resolution", "leiden_sub_resolution"]:
        if column in plot_df.columns:
            plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    return plot_df.sort_values(["epitope", "method", "f1"], ascending=[True, True, False]).reset_index(drop=True)


def summarize_vdjdb_grid_metrics(plot_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        plot_df.groupby(["method", "epitope"], observed=True)["f1"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )
    collapse_rate = (
        plot_df.assign(collapsed=plot_df["f1"] < 0.1)
        .groupby(["method", "epitope"], observed=True)["collapsed"]
        .mean()
        .reset_index(name="collapse_rate")
    )
    summary = summary.merge(collapse_rate, on=["method", "epitope"], how="left")
    summary["range"] = summary["max"] - summary["min"]
    return summary.sort_values(["epitope", "median", "max"], ascending=[True, False, False]).reset_index(drop=True)


def best_vdjdb_grid_configs(plot_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "epitope",
        "method",
        "f1",
        "precision",
        "recall",
        "weighted_cluster_purity",
        "positive_fragmentation_norm",
        "noise_fraction",
        "runtime_seconds",
        "parameter_json",
    ]
    best = (
        plot_df.sort_values(["epitope", "method", "f1"], ascending=[True, True, False])
        .groupby(["epitope", "method"], observed=True, as_index=False)
        .first()
    )
    return best.loc[:, [column for column in columns if column in best.columns]]


def best_method_runs(plot_df: pd.DataFrame, method_name: str, top_n: int = 3) -> pd.DataFrame:
    subset = plot_df.loc[plot_df["method"].astype(str) == method_name].copy()
    if subset.empty:
        return subset
    columns = [
        "epitope",
        "method",
        "f1",
        "precision",
        "recall",
        "weighted_cluster_purity",
        "positive_fragmentation_norm",
        "cluster_min_samples",
        "k_neighbors",
        "eps_k_neighbors",
        "eps_estimation_based_on",
        "leiden_resolution",
        "noise_fraction",
        "runtime_seconds",
        "parameter_json",
    ]
    top = (
        subset.sort_values(["epitope", "f1"], ascending=[True, False])
        .groupby("epitope", observed=True, as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return top.loc[:, [column for column in columns if column in top.columns]]


def plot_vdjdb_grid_f1_landscape(plot_df: pd.DataFrame, output_stem: str | Path) -> plt.Figure:
    epitopes = plot_df["epitope"].cat.categories.tolist()
    methods = [method for method in METHOD_ORDER if method in set(plot_df["method"].dropna().astype(str))]
    fig, axes = plt.subplots(len(epitopes), len(methods), figsize=(4.0 * len(methods), 3.8 * len(epitopes)), squeeze=False)

    for row_index, epitope in enumerate(epitopes):
        for col_index, method in enumerate(methods):
            ax = axes[row_index][col_index]
            subset = plot_df.loc[(plot_df["epitope"] == epitope) & (plot_df["method"].astype(str) == method)].copy()
            if subset.empty:
                ax.axis("off")
                continue
            heatmap_df = (
                subset.groupby(["cluster_min_samples", "k_neighbors"], observed=True)["f1"]
                .max()
                .unstack()
                .sort_index()
            )
            sns.heatmap(
                heatmap_df,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                vmin=0.0,
                vmax=1.0,
                linewidths=0.5,
                cbar=col_index == len(methods) - 1,
                ax=ax,
            )
            ax.set_title(method)
            ax.set_xlabel("k_neighbors")
            ax.set_ylabel("cluster_min_samples" if col_index == 0 else "")
            if col_index == 0:
                ax.text(-0.55, 0.5, epitope, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=14)
    fig.suptitle("Best F1 over the common hyperparameter surface", y=1.02)
    fig.tight_layout()
    _save(fig, output_stem)
    return fig


def plot_vdjdb_grid_method_stability(plot_df: pd.DataFrame, output_stem: str | Path) -> plt.Figure:
    fig, axes = plt.subplots(1, len(plot_df["epitope"].cat.categories), figsize=(14, 5), sharey=True)
    axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    order = (
        plot_df.groupby("method", observed=True)["f1"]
        .median()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    for ax, epitope in zip(axes, plot_df["epitope"].cat.categories.tolist()):
        subset = plot_df.loc[plot_df["epitope"] == epitope].copy()
        sns.boxplot(data=subset, x="method", y="f1", order=order, color="#d7e8f7", fliersize=0, ax=ax)
        sns.stripplot(data=subset, x="method", y="f1", order=order, hue="method", palette="tab10", dodge=False, alpha=0.55, size=5, ax=ax)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.set_title(str(epitope))
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("F1 across all grid runs")
    fig.suptitle("Method stability: robust methods stay tight, brittle methods spread out", y=1.02)
    fig.tight_layout()
    _save(fig, output_stem)
    return fig


def plot_vdjdb_grid_knn_sensitivity(plot_df: pd.DataFrame, output_stem: str | Path) -> plt.Figure:
    summary = (
        plot_df.groupby(["epitope", "method", "k_neighbors"], observed=True)["f1"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    epitopes = plot_df["epitope"].cat.categories.tolist()
    fig, axes = plt.subplots(1, len(epitopes), figsize=(14, 5), sharey=True)
    axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    palette = dict(zip(METHOD_ORDER, sns.color_palette("tab10", n_colors=len(METHOD_ORDER))))
    for ax, epitope in zip(axes, epitopes):
        subset = summary.loc[summary["epitope"] == epitope].copy()
        for method in [value for value in METHOD_ORDER if value in subset["method"].astype(str).unique()]:
            method_df = subset.loc[subset["method"].astype(str) == method].sort_values("k_neighbors")
            ax.plot(method_df["k_neighbors"], method_df["mean"], marker="o", linewidth=2, label=method, color=palette[method])
            ax.fill_between(method_df["k_neighbors"], method_df["min"], method_df["max"], alpha=0.12, color=palette[method])
        ax.set_title(str(epitope))
        ax.set_xlabel("k_neighbors")
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("F1")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="method")
    fig.suptitle("Sensitivity to neighborhood size: line is mean, band is min-max", y=1.02)
    fig.tight_layout()
    _save(fig, output_stem)
    return fig


def plot_vdjdb_grid_tradeoff_map(plot_df: pd.DataFrame, output_stem: str | Path) -> plt.Figure:
    fig, axes = plt.subplots(1, len(plot_df["epitope"].cat.categories), figsize=(14, 5), sharex=True, sharey=True)
    axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    for ax, epitope in zip(axes, plot_df["epitope"].cat.categories.tolist()):
        subset = plot_df.loc[plot_df["epitope"] == epitope].copy()
        sns.scatterplot(
            data=subset,
            x="recall",
            y="weighted_cluster_purity",
            hue="method",
            size="f1",
            sizes=(40, 260),
            alpha=0.75,
            palette="tab10",
            ax=ax,
        )
        ax.set_title(str(epitope))
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Weighted cluster purity")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("Trade-off map: large points combine high recall, purity, and F1", y=1.02)
    fig.tight_layout()
    _save(fig, output_stem)
    return fig


def show_vdjdb_grid_search_outputs(grid_csv_path: str | Path, repo_root: str | Path) -> pd.DataFrame:
    repo_root = Path(repo_root)
    figure_dir = repo_root / "figures" / "clustering_strategy"
    plot_df = load_vdjdb_grid_metrics(grid_csv_path)
    summary_df = summarize_vdjdb_grid_metrics(plot_df)
    best_df = best_vdjdb_grid_configs(plot_df)
    vdbscan_leiden_top_df = best_method_runs(plot_df, "vdbscan_leiden", top_n=3)

    robust = summary_df.sort_values(["epitope", "median", "max"], ascending=[True, False, False]).groupby("epitope", observed=True).first()
    peak = summary_df.sort_values(["epitope", "max", "median"], ascending=[True, False, False]).groupby("epitope", observed=True).first()
    brittle = summary_df.sort_values(["epitope", "collapse_rate", "range"], ascending=[True, False, False]).groupby("epitope", observed=True).first()
    vdbscan_leiden_summary = summary_df.loc[summary_df["method"].astype(str) == "vdbscan_leiden"].set_index("epitope")

    takeaways = ["## Grid Search Takeaways"]
    for epitope in plot_df["epitope"].cat.categories.tolist():
        robust_row = robust.loc[epitope]
        peak_row = peak.loc[epitope]
        brittle_row = brittle.loc[epitope]
        takeaways.append(
            "- `{0}`: most robust median F1 is `{1}` ({2:.3f}), peak F1 is `{3}` ({4:.3f}), most brittle is `{5}` (collapse rate {6:.0%})".format(
                epitope,
                robust_row["method"],
                robust_row["median"],
                peak_row["method"],
                peak_row["max"],
                brittle_row["method"],
                brittle_row["collapse_rate"],
            )
        )
        if epitope in vdbscan_leiden_summary.index:
            vdbscan_row = vdbscan_leiden_summary.loc[epitope]
            top_vdbscan_row = vdbscan_leiden_top_df.loc[vdbscan_leiden_top_df["epitope"] == epitope].sort_values("f1", ascending=False).iloc[0]
            takeaways.append(
                "  `vdbscan_leiden` is especially interesting here: peak F1 reaches `{0:.3f}` with recall `{1:.3f}` and purity `{2:.3f}`; the strongest region is `k={3}`, `eps_k={4}`, `min_samples={5}`, `leiden_res={6}`, `eps source={7}`. The catch is brittleness: median F1 is only `{8:.3f}` and collapse rate is `{9:.0%}`.".format(
                    top_vdbscan_row["f1"],
                    top_vdbscan_row["recall"],
                    top_vdbscan_row["weighted_cluster_purity"],
                    int(top_vdbscan_row["k_neighbors"]),
                    int(top_vdbscan_row["eps_k_neighbors"]),
                    int(top_vdbscan_row["cluster_min_samples"]),
                    top_vdbscan_row["leiden_resolution"],
                    top_vdbscan_row["eps_estimation_based_on"],
                    vdbscan_row["median"],
                    vdbscan_row["collapse_rate"],
                )
            )
    display_markdown("\n".join(takeaways))
    display_markdown(
        "## Interpretation\n"
        "- `hierarchical_leiden` is the safest default because its distributions are tight and its best runs are close to its median.\n"
        "- `vdbscan_leiden` deserves focused follow-up because it combines a graph partition with density-aware filtering and reaches near-best peaks, especially on `GLCTLVAML`.\n"
        "- The key message from the new plots is not only who wins, but whether success comes from a broad stable plateau or from a narrow parameter pocket."
    )
    display_markdown("## Stability Summary")
    display_frame(summary_df.round(3), rows=len(summary_df))
    display_markdown("## Best Config Per Method And Epitope")
    display_frame(best_df.round(3), rows=len(best_df))
    if not vdbscan_leiden_top_df.empty:
        display_markdown("## vdbscan_leiden Peak Configs")
        display_frame(vdbscan_leiden_top_df.round(3), rows=len(vdbscan_leiden_top_df))

    plot_vdjdb_grid_f1_landscape(plot_df, figure_dir / "vdjdb_grid_f1_landscape")
    plot_vdjdb_grid_method_stability(plot_df, figure_dir / "vdjdb_grid_method_stability")
    plot_vdjdb_grid_knn_sensitivity(plot_df, figure_dir / "vdjdb_grid_knn_sensitivity")
    plot_vdjdb_grid_tradeoff_map(plot_df, figure_dir / "vdjdb_grid_tradeoff_map")

    display_markdown("## Figures")
    display_image(figure_dir / "vdjdb_grid_f1_landscape.png")
    display_image(figure_dir / "vdjdb_grid_method_stability.png")
    display_image(figure_dir / "vdjdb_grid_knn_sensitivity.png")
    display_image(figure_dir / "vdjdb_grid_tradeoff_map.png")
    return plot_df
