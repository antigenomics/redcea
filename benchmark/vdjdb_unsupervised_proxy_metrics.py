from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from .notebook_displays import display_frame, display_image, display_markdown
from .plotting import _save
from .vdjdb_grid_visualization import load_vdjdb_grid_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sns, "set_theme"):
    sns.set_theme(style="whitegrid", context="talk")
else:
    sns.set_style("whitegrid")


def _rank_correlation_table(df: pd.DataFrame, metrics: list[str], *, group_label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        subset = df.loc[:, ["f1", metric]].dropna()
        if len(subset) < 3 or subset[metric].nunique() < 2:
            continue
        rows.append(
            {
                "group": group_label,
                "metric": metric,
                "n_runs": int(len(subset)),
                "pearson_r": float(subset["f1"].corr(subset[metric], method="pearson")),
                "spearman_rho": float(subset["f1"].corr(subset[metric], method="spearman")),
            }
        )
    return pd.DataFrame(rows)


def compute_global_proxy_metrics(plot_df: pd.DataFrame) -> pd.DataFrame:
    out = plot_df.copy()
    out["coverage"] = 1.0 - out["noise_fraction"]
    out["log_n_clusters"] = np.log10(out["n_clusters"].clip(lower=1))
    out["clusters_per_1000_non_noise_points"] = out["n_clusters"] / ((1.0 - out["noise_fraction"]).clip(lower=1e-6) * 1000.0)
    out["throughput_points_per_second"] = 1.0 / out["runtime_seconds"].clip(lower=1e-6)
    return out


def compute_local_assignment_proxy_metrics(assignments_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(Path(assignments_dir).glob("vdjdb_*.parquet")):
        assignments = pd.read_parquet(path)
        sample = assignments.loc[assignments["sample_label"] == "sample"].copy()
        if sample.empty:
            continue
        non_noise = sample.loc[~sample["is_noise"]].copy()
        cluster_sizes = non_noise.groupby("cluster_id").size().sort_values(ascending=False)
        clustered_points = int(len(non_noise))
        cluster_count = int(len(cluster_sizes))
        proportions = (cluster_sizes / max(1, clustered_points)).to_numpy(dtype=float) if cluster_count else np.array([], dtype=float)
        entropy = float(-(proportions * np.log(proportions + 1e-12)).sum()) if cluster_count else np.nan
        rows.append(
            {
                "run_id": path.stem,
                "sample_coverage": float((~sample["is_noise"]).mean()),
                "sample_n_clusters": cluster_count,
                "sample_avg_cluster_size": float(cluster_sizes.mean()) if cluster_count else np.nan,
                "sample_largest_cluster_fraction": float(cluster_sizes.iloc[0] / max(1, clustered_points)) if cluster_count else np.nan,
                "sample_singleton_cluster_fraction": float((cluster_sizes == 1).mean()) if cluster_count else np.nan,
                "sample_singleton_point_fraction": float(cluster_sizes.loc[cluster_sizes == 1].sum() / max(1, clustered_points)) if cluster_count else np.nan,
                "sample_cluster_entropy_norm": float(entropy / np.log(cluster_count)) if cluster_count > 1 else np.nan,
                "sample_effective_cluster_count": float(1.0 / np.square(proportions).sum()) if cluster_count else np.nan,
                "sample_cluster_size_cv": float(cluster_sizes.std() / cluster_sizes.mean()) if cluster_count > 1 and float(cluster_sizes.mean()) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_local_knn_proxy_metrics(assignments_dir: str | Path, redcea_runs_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(Path(assignments_dir).glob("vdjdb_*.parquet")):
        run_id = path.stem
        run_dir = Path(redcea_runs_dir) / run_id
        knn_index_files = sorted(run_dir.glob("knn_sample_sample__k*__l2.indices.npy"))
        if not knn_index_files:
            continue

        assignments = pd.read_parquet(path)
        sample = assignments.loc[assignments["sample_label"] == "sample"].copy().reset_index(drop=True)
        labels = sample["cluster_id"].to_numpy()
        knn_indices = np.load(knn_index_files[0])
        knn_distances = np.load(Path(str(knn_index_files[0]).replace(".indices.npy", ".distances.npy")))
        if knn_indices.shape[0] != len(sample):
            continue
        if knn_indices.shape[1] > 0 and np.all(knn_indices[:, 0] == np.arange(len(sample))):
            knn_indices = knn_indices[:, 1:]
            knn_distances = knn_distances[:, 1:]
        neighbor_labels = labels[knn_indices]
        valid_edges = (labels[:, None] >= 0) & (neighbor_labels >= 0)
        same_cluster = (neighbor_labels == labels[:, None]) & valid_edges
        if not valid_edges.any():
            continue
        mean_all_distance = float(knn_distances.mean())
        mean_same_distance = float(knn_distances[same_cluster].mean()) if same_cluster.any() else np.nan
        rows.append(
            {
                "run_id": run_id,
                "knn_same_cluster_fraction": float(same_cluster.sum() / max(1, valid_edges.sum())),
                "knn_nn_same_cluster_rate": float(same_cluster[:, 0].mean()) if same_cluster.shape[1] else np.nan,
                "knn_mean_same_neighbors": float(same_cluster.sum(axis=1).mean()),
                "knn_same_distance_ratio": float(mean_same_distance / mean_all_distance) if np.isfinite(mean_same_distance) and mean_all_distance else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_proxy_correlation_tables(
    grid_csv_path: str | Path,
    *,
    assignments_dir: str | Path = "results/clustering_assignments",
    redcea_runs_dir: str | Path = "results/redcea_runs",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plot_df = compute_global_proxy_metrics(load_vdjdb_grid_metrics(grid_csv_path))

    global_metrics = [
        "coverage",
        "noise_fraction",
        "n_clusters",
        "log_n_clusters",
        "clusters_per_1000_non_noise_points",
        "runtime_seconds",
        "throughput_points_per_second",
    ]
    global_tables = [
        _rank_correlation_table(plot_df, global_metrics, group_label="all_methods"),
        _rank_correlation_table(
            plot_df.loc[plot_df["method"].astype(str) == "vdbscan_leiden"].copy(),
            global_metrics,
            group_label="vdbscan_leiden_only",
        ),
    ]
    for epitope, subset in plot_df.groupby("epitope", observed=True):
        global_tables.append(_rank_correlation_table(subset, global_metrics, group_label="epitope:{0}".format(epitope)))
    global_corr = (
        pd.concat(global_tables, ignore_index=True)
        .sort_values(["group", "spearman_rho"], ascending=[True, False])
        .reset_index(drop=True)
    )

    local_assignment = compute_local_assignment_proxy_metrics(assignments_dir)
    local_knn = compute_local_knn_proxy_metrics(assignments_dir, redcea_runs_dir)
    local_proxy = local_assignment.merge(local_knn, on="run_id", how="outer")
    local_proxy = local_proxy.merge(plot_df.loc[:, ["run_id", "method", "epitope", "f1"]].drop_duplicates(), on="run_id", how="left")
    local_metrics = [column for column in local_proxy.columns if column.startswith("sample_") or column.startswith("knn_")]
    local_corr = _rank_correlation_table(local_proxy, local_metrics, group_label="local_runs_only").sort_values(
        "spearman_rho",
        ascending=False,
    )
    return plot_df, global_corr.reset_index(drop=True), local_corr.reset_index(drop=True)


def plot_proxy_correlation_heatmaps(
    global_corr: pd.DataFrame,
    local_corr: pd.DataFrame,
    output_stem: str | Path,
) -> plt.Figure:
    pivot_global = global_corr.pivot(index="metric", columns="group", values="spearman_rho")
    pivot_local = local_corr.pivot(index="metric", columns="group", values="spearman_rho") if len(local_corr) else pd.DataFrame()

    ncols = 2 if len(pivot_local) else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7.0 * ncols, max(4.5, 0.45 * max(len(pivot_global), len(pivot_local) or 1))), squeeze=False)
    sns.heatmap(
        pivot_global.sort_values(by=list(pivot_global.columns), ascending=False),
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=1.0,
        linewidths=0.5,
        ax=axes[0][0],
    )
    axes[0][0].set_title("Full VDJdb grid")
    axes[0][0].set_xlabel("")
    axes[0][0].set_ylabel("")
    if len(pivot_local):
        sns.heatmap(
            pivot_local.sort_values(by=list(pivot_local.columns), ascending=False),
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=-1.0,
            vmax=1.0,
            linewidths=0.5,
            ax=axes[0][1],
        )
        axes[0][1].set_title("Local runs with graph artifacts")
        axes[0][1].set_xlabel("")
        axes[0][1].set_ylabel("")
    fig.suptitle("Label-free proxy metrics vs target F1", y=1.02)
    fig.tight_layout()
    _save(fig, output_stem)
    return fig


def show_vdjdb_unsupervised_proxy_outputs(
    grid_csv_path: str | Path,
    repo_root: str | Path,
    *,
    assignments_dir: str | Path = "results/clustering_assignments",
    redcea_runs_dir: str | Path = "results/redcea_runs",
) -> pd.DataFrame:
    repo_root = Path(repo_root)
    figure_dir = repo_root / "figures" / "clustering_strategy"
    plot_df, global_corr, local_corr = build_proxy_correlation_tables(
        grid_csv_path,
        assignments_dir=assignments_dir,
        redcea_runs_dir=redcea_runs_dir,
    )

    best_global = global_corr.loc[global_corr["group"] == "all_methods"].sort_values("spearman_rho", ascending=False)
    best_local = local_corr.sort_values("spearman_rho", ascending=False)
    takeaways = [
        "## Candidate Label-Free Metrics",
        "- `coverage = 1 - noise_fraction` is the strongest full-grid proxy already available in the VDJdb metrics export.",
        "- `n_clusters` by itself is unstable across epitopes and methods, so it is a weak standalone selection criterion.",
        "- On the locally available runs, two stronger structural candidates appear: `sample_singleton_point_fraction` and `knn_nn_same_cluster_rate`.",
    ]
    if len(best_global):
        row = best_global.iloc[0]
        takeaways.append(
            "- Across the full VDJdb grid, the top proxy is `{0}` with Spearman correlation `{1:.3f}` to `F1`.".format(
                row["metric"],
                row["spearman_rho"],
            )
        )
    if len(best_local):
        row = best_local.iloc[0]
        takeaways.append(
            "- On the subset of runs where local assignment or kNN artifacts are available, the best extra proxy is `{0}` with Spearman `{1:.3f}`. This needs validation on the full run archive before we treat it as the default unlabeled score.".format(
                row["metric"],
                row["spearman_rho"],
            )
        )
    display_markdown("\n".join(takeaways))

    display_markdown("## Global Proxy Correlations")
    display_frame(global_corr.round(3), rows=len(global_corr))
    if len(local_corr):
        display_markdown("## Local Structural Proxy Correlations")
        display_frame(local_corr.round(3), rows=len(local_corr))

    plot_proxy_correlation_heatmaps(global_corr, local_corr, figure_dir / "vdjdb_unsupervised_proxy_correlations")
    display_markdown("## Proxy Metric Figures")
    display_image(figure_dir / "vdjdb_unsupervised_proxy_correlations.png")
    return plot_df
