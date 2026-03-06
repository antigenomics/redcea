#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP


# -------------------------
# helpers
# -------------------------
def _norm_chain(chain: str) -> str:
    chain = chain.strip().upper()
    if chain not in {"TRA", "TRB"}:
        raise ValueError("chain must be TRA or TRB")
    return chain


def _pick_cols(chain: str):
    # columns expected in tcremp_representations.tsv
    if chain == "TRB":
        return dict(
            cdr3aa="cdr3aa_beta",
            v="v_beta",
            j="j_beta",
            # hover extras if exist
            extra_hover=["cdr3aa_beta", "v_beta", "j_beta", "clone_id", "cluster_id"],
        )
    else:
        return dict(
            cdr3aa="cdr3aa_alpha",
            v="v_alpha",
            j="j_alpha",
            extra_hover=["cdr3aa_alpha", "v_alpha", "j_alpha", "clone_id", "cluster_id"],
        )


def _default_bg_path(chain: str) -> Path:
    # поменяй при необходимости на свои реальные файлы
    if chain == "TRB":
        return Path("/projects/immunestatus/vdjdb/tcremp/trb_background_embeddings_10000.parquet")
    else:
        return Path("/projects/immunestatus/vdjdb/tcremp/tra_background_embeddings_10000.parquet")


# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        help="tcrempnet run directory (relative name under /projects/immunestatus/vdjdb), "
             "e.g. tcrempnet_YLQPRTFLL_trb_leiden_...",
    )
    parser.add_argument("--epitope", required=True, help="Epitope AA sequence, e.g. YLQPRTFLL")
    parser.add_argument("--chain", required=True, help="TRA or TRB")

    parser.add_argument(
        "--bg-parquet",
        default=None,
        help="Path to background embeddings parquet. If not set, uses a chain-specific default.",
    )
    parser.add_argument(
        "--bg-sample-size",
        type=int,
        default=0,
        help="If >0, randomly sample that many background points before building density.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for background sampling / UMAP.",
    )

    # UMAP/PCA knobs (optional but handy)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.5)

    parser.add_argument(
        "--out-html",
        default=None,
        help="Output html file. Default: <run_dir>_bg.html (slashes replaced).",
    )

    args = parser.parse_args()

    epitope = args.epitope.strip()
    chain = _norm_chain(args.chain)

    base_dir = Path("/projects/immunestatus/vdjdb") / args.run_dir
    if not base_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {base_dir}")

    out_html = args.out_html or f"{args.run_dir.replace('/', '_')}_bg.html"

    prefix = f"{chain.lower()}_vdjdb_{epitope}"
    reps_path = Path("/projects/immunestatus/vdjdb/tcremp") / f"{prefix}_tcremp_representations.tsv"
    emb_path = Path("/projects/immunestatus/vdjdb/tcremp") / f"{prefix}_embeddings.parquet"
    clust_path = base_dir / f"{prefix}_enriched_clonotypes_tcremp.tsv"

    if not reps_path.exists():
        raise FileNotFoundError(f"Representations not found: {reps_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")
    if not clust_path.exists():
        raise FileNotFoundError(f"Enriched clonotypes not found: {clust_path}")

    bg_path = Path(args.bg_parquet) if args.bg_parquet else _default_bg_path(chain)
    if not bg_path.exists():
        raise FileNotFoundError(
            f"Background parquet not found: {bg_path}\n"
            f"Pass --bg-parquet explicitly or create the default file."
        )

    cols = _pick_cols(chain)

    # -------------------------
    # load
    # -------------------------
    info = pd.read_csv(reps_path, sep="\t")

    clonotypes = pd.read_csv(clust_path, sep="\t")
    clonotypes = clonotypes[clonotypes["source"] == "sample"].copy()

    # clone_id: "sample_123" -> 123
    clonotypes["clone_id"] = clonotypes["clone_id"].astype(str).apply(lambda x: int(x.split("_")[1]))

    # cluster mapping
    info = info.merge(clonotypes[["clone_id", "cluster_id"]], on="clone_id", how="left")

    initial_emb = pd.read_parquet(emb_path)
    bg = pd.read_parquet(bg_path)

    if args.bg_sample_size and args.bg_sample_size > 0 and args.bg_sample_size < len(bg):
        bg = bg.sample(n=args.bg_sample_size, random_state=args.seed).reset_index(drop=True)

    # -------------------------
    # fit projection on background, transform points
    # -------------------------
    scaler = StandardScaler()
    pca = PCA(n_components=args.pca_dim, random_state=args.seed)
    umap = UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        # random_state=args.seed,
        # transform=True by default in many versions; leave as-is, umap.transform should work
    )

    bg_pca = pca.fit_transform(scaler.fit_transform(bg))
    bg_umap = umap.fit_transform(bg_pca)

    emb_pca = pca.transform(scaler.transform(initial_emb))
    emb_umap = umap.transform(emb_pca)

    info["x"] = emb_umap[:, 0]
    info["y"] = emb_umap[:, 1]

    # -------------------------
    # plot: clusters over background density
    # -------------------------
    plot_df = info.copy()
    plot_df["cluster"] = plot_df["cluster_id"].fillna("unclustered")

    fig_points = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="cluster",
        color_discrete_map={"unclustered": "lightgrey"},
        hover_data=[c for c in cols["extra_hover"] if c in plot_df.columns],
    )

    bg_df = pd.DataFrame(bg_umap, columns=["x", "y"])

    fig = go.Figure()
    fig.add_trace(
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

    for tr in fig_points.data:
        fig.add_trace(tr)

    fig.update_layout(
        title=f"{chain} clustering with background density ({epitope})",
        width=1000,
        height=700,
        template="plotly_white",
        legend_title_text="cluster",
    )

    # -------------------------
    # save
    # -------------------------
    with open(out_html, "w") as f:
        f.write(fig.to_html(full_html=True, include_plotlyjs="cdn"))

    print(f"Run dir: {base_dir}")
    print(f"Epitope: {epitope} | Chain: {chain}")
    print(f"Background: {bg_path} (n={len(bg)})")
    print(f"Saved: {out_html}")


if __name__ == "__main__":
    main()