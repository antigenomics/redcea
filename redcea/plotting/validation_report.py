from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from redcea.plotting.figure_io import plt_to_html


def build_validation_report_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        help="tcrempnet run directory, e.g. tcrempnet_YLQPRTFLL_trb_leiden_100k_res100",
    )
    parser.add_argument(
        "--epitope",
        required=True,
        help="Epitope amino-acid sequence, e.g. YLQPRTFLL",
    )
    parser.add_argument(
        "--base-dir",
        default="/projects/immunestatus/vdjdb",
        help="Base directory containing run outputs.",
    )
    parser.add_argument(
        "--tcremp-dir",
        default="/projects/immunestatus/vdjdb/tcremp",
        help="Directory with base TCRemP outputs.",
    )
    parser.add_argument(
        "--background-path",
        default="/projects/immunestatus/vdjdb/tcremp/trb_background_embeddings_10000.parquet",
        help="Path to background embeddings parquet.",
    )
    parser.add_argument(
        "--matchmakers-path",
        default="/home/evlasova/tcrempnet/data/01_05_2025_TCRvdb.csv",
        help="Path to validation CSV.",
    )
    parser.add_argument(
        "--output-html",
        default=None,
        help="Optional output HTML path. Defaults to <run_dir>.html in current directory.",
    )
    return parser


def build_validation_report(args) -> tuple[go.Figure, go.Figure, object]:
    epitope = args.epitope.strip()
    base_dir = Path(args.base_dir) / args.run_dir
    prefix = f"trb_vdjdb_{epitope}"

    clonotypes = pd.read_csv(base_dir / f"{prefix}_enriched_clonotypes_tcremp.tsv", sep="\t")
    clonotypes_sample = clonotypes[clonotypes.source == "sample"].copy()
    clonotypes_sample["clone_id"] = clonotypes_sample["clone_id"].apply(lambda x: int(x.split("_")[1]))

    tcremp_dir = Path(args.tcremp_dir)
    initial_emb = pd.read_parquet(tcremp_dir / f"{prefix}_embeddings.parquet")

    scaler = StandardScaler()
    pca = PCA(n_components=50)
    umap = UMAP(min_dist=0.5)

    initial_emb_pca = pca.fit_transform(scaler.fit_transform(initial_emb))
    initial_emb_umap = umap.fit_transform(initial_emb_pca)

    info = pd.read_csv(tcremp_dir / f"{prefix}_tcremp_representations.tsv", sep="\t")
    info["x"] = initial_emb_umap[:, 0]
    info["y"] = initial_emb_umap[:, 1]
    info = info.merge(clonotypes_sample[["clone_id", "cluster_id"]], how="left")

    bg = pd.read_parquet(args.background_path)
    bg_pca = pca.fit_transform(scaler.fit_transform(bg.copy()))
    bg_umap = umap.fit_transform(bg_pca)

    initial_emb_pca = pca.transform(scaler.transform(initial_emb))
    initial_emb_umap = umap.transform(initial_emb_pca)
    info["x"] = initial_emb_umap[:, 0]
    info["y"] = initial_emb_umap[:, 1]
    info = info.merge(clonotypes_sample[["clone_id", "cluster_id"]], how="left")

    matchmakers = pd.read_csv(args.matchmakers_path).drop(columns=["Unnamed: 0"]).dropna(subset="padj")
    matchmakers["valid"] = matchmakers.padj < 1e-5
    matchmakers = matchmakers[matchmakers["epitope_aa"] == epitope]

    ylq_valid = (
        matchmakers[["cdr3_beta_aa", "TRBV", "TRBJ", "valid"]]
        .groupby(["cdr3_beta_aa", "TRBV", "TRBJ"], as_index=False)
        .any()
    )

    df = info.copy()
    df["TRBV"] = df["v_beta"].str.replace(r"\*.*$", "", regex=True)
    df["TRBJ"] = df["j_beta"].str.replace(r"\*.*$", "", regex=True)
    df["cdr3_beta_aa"] = df["cdr3aa_beta"]
    df = df.merge(ylq_valid, on=["cdr3_beta_aa", "TRBV", "TRBJ"], how="left")
    df["is_clustered"] = df.cluster_id.notna()
    df["cluster"] = df["cluster_id"].fillna("unclustered")

    fig_scatter = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        color_discrete_map={"unclustered": "lightgrey"},
        hover_data=["cdr3aa_beta", "v_beta", "j_beta", "valid"],
    )
    fig_scatter.update_layout(
        title=f"TCR clustering ({epitope})",
        width=1000,
        height=700,
        template="plotly_white",
    )

    bg_df = pd.DataFrame(bg_umap, columns=["x", "y"])
    fig_bg = go.Figure()
    fig_bg.add_trace(
        go.Histogram2dContour(
            x=bg_df["x"],
            y=bg_df["y"],
            ncontours=20,
            contours=dict(coloring="fill", showlines=False),
            colorscale="Greys",
            showscale=False,
            hoverinfo="skip",
            opacity=0.35,
        )
    )
    for trace in fig_scatter.data:
        fig_bg.add_trace(trace)
    fig_bg.update_layout(
        title=f"TCR clustering with background density ({epitope})",
        width=1000,
        height=700,
        template="plotly_white",
    )

    df_cm = df[df.valid.notna()]
    y_true = df_cm.valid.astype(bool)
    y_pred = df_cm.is_clustered.astype(bool)

    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    fig_cm, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred True", "Pred False"])
    ax.set_yticklabels(["GT True", "GT False"])
    ax.set_title(f"{epitope}\nPrecision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    return fig_scatter, fig_bg, fig_cm


def write_validation_report(output_html: str | Path, fig_scatter: go.Figure, fig_bg: go.Figure, fig_cm) -> Path:
    output_path = Path(output_html)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(fig_scatter.to_html(full_html=False, include_plotlyjs="cdn"))
        handle.write(fig_bg.to_html(full_html=False, include_plotlyjs=False))
        handle.write("<h2>Confusion matrix</h2>")
        handle.write(plt_to_html(fig_cm))
    return output_path


def main(argv=None) -> Path:
    parser = build_validation_report_parser()
    args = parser.parse_args(argv)

    base_dir = Path(args.base_dir) / args.run_dir
    output_html = args.output_html or f"{args.run_dir.replace('/', '_')}.html"

    logging.info("Running report for: %s", base_dir)
    logging.info("Epitope: %s", args.epitope.strip())
    logging.info("Output: %s", output_html)

    fig_scatter, fig_bg, fig_cm = build_validation_report(args)
    output_path = write_validation_report(output_html, fig_scatter, fig_bg, fig_cm)
    logging.info("Done.")
    return output_path


__all__ = [
    "build_validation_report",
    "build_validation_report_parser",
    "main",
    "write_validation_report",
]
