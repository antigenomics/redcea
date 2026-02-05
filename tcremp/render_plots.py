#!/usr/bin/env python

import argparse
from pathlib import Path
import io
import base64

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP


# =========================
# utils
# =========================
def plt_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img}"/>'


# =========================
# args
# =========================
parser = argparse.ArgumentParser()
parser.add_argument(
    "run_dir",
    help="tcrempnet run directory, e.g. tcrempnet_YLQPRTFLL_trb_leiden_100k_res100",
)
args = parser.parse_args()

base_dir = f'/projects/immunestatus/vdjdb/{args.run_dir}'
out_html = f"{args.run_dir}.html"

print(f"Running report for: {base_dir}")
print(f"Output: {out_html}")


# =========================
# LOAD DATA (copied from notebook)
# =========================
prefix = "trb_vdjdb_YLQPRTFLL"

embeding_sample = pd.read_parquet(
    f"{base_dir}/{prefix}_enriched_embeddings_tcremp.parquet"
)

clonotypes = pd.read_csv(
    f"{base_dir}/{prefix}_enriched_clonotypes_tcremp.tsv",
    sep="\t"
)

clonotypes_sample = clonotypes[clonotypes.source == "sample"].copy()
clonotypes_sample["clone_id"] = clonotypes_sample["clone_id"].apply(
    lambda x: int(x.split("_")[1])
)

initial_emb = pd.read_parquet(
    "/projects/immunestatus/vdjdb/tcremp/trb_vdjdb_YLQPRTFLL_embeddings.parquet"
)

scaler = StandardScaler()
pca = PCA(n_components=50)
umap = UMAP(min_dist=0.5)

initial_emb_pca = pca.fit_transform(
    scaler.fit_transform(initial_emb)
)
initial_emb_umap = umap.fit_transform(initial_emb_pca)

info = pd.read_csv(
    "/projects/immunestatus/vdjdb/tcremp/trb_vdjdb_YLQPRTFLL_tcremp_representations.tsv",
    sep="\t"
)

info["x"] = initial_emb_umap[:, 0]
info["y"] = initial_emb_umap[:, 1]

info = info.merge(
    clonotypes_sample[["clone_id", "cluster_id"]],
    how="left"
)

bg = pd.read_parquet(
    "/projects/immunestatus/vdjdb/tcremp/trb_background_embeddings_10000.parquet"
)

sampled_bg = bg.copy() # ! change if you want

bg_pca = pca.fit_transform(
    scaler.fit_transform(sampled_bg)
)
bg_umap = umap.fit_transform(bg_pca)

initial_emb_pca = pca.transform(
    scaler.transform(initial_emb)
)
initial_emb_umap = umap.transform(initial_emb_pca)

info["x"] = initial_emb_umap[:, 0]
info["y"] = initial_emb_umap[:, 1]

info = info.merge(
    clonotypes_sample[["clone_id", "cluster_id"]],
    how="left"
)

matchmakers = (
    pd.read_csv("/home/evlasova/tcrempnet/data/01_05_2025_TCRvdb.csv")
    .drop(columns=["Unnamed: 0"])
    .dropna(subset="padj")
)

matchmakers["valid"] = matchmakers.padj < 1e-5
matchmakers = matchmakers[matchmakers["epitope_aa"] == "YLQPRTFLL"]

ylq_valid = (
    matchmakers[["cdr3_beta_aa", "TRBV", "TRBJ", "valid"]]
    .groupby(["cdr3_beta_aa", "TRBV", "TRBJ"], as_index=False)
    .any()
)


# =========================
# MERGE VALIDATION
# =========================
df = info.copy()

df["TRBV"] = df["v_beta"].str.replace(r"\*.*$", "", regex=True)
df["TRBJ"] = df["j_beta"].str.replace(r"\*.*$", "", regex=True)
df["cdr3_beta_aa"] = df["cdr3aa_beta"]

df = df.merge(
    ylq_valid,
    on=["cdr3_beta_aa", "TRBV", "TRBJ"],
    how="left",
)

df["is_clustered"] = df.cluster_id.notna()


# =========================
# PLOT 1 — Plotly scatter
# =========================
df["cluster"] = df["cluster_id"].fillna("unclustered")

fig_scatter = px.scatter(
    df,
    x="x",
    y="y",
    color="cluster",
    color_discrete_map={"unclustered": "lightgrey"},
    hover_data=["cdr3aa_beta", "v_beta", "j_beta"],
)

fig_scatter.update_layout(
    title="TCR clustering",
    width=1000,
    height=700,
    template="plotly_white",
)


# =========================
# PLOT 2 — Scatter + background density
# =========================
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

for tr in fig_scatter.data:
    fig_bg.add_trace(tr)

fig_bg.update_layout(
    title="TCR clustering with background density",
    width=1000,
    height=700,
    template="plotly_white",
)


# =========================
# PLOT 3 — Confusion matrix (mpl)
# =========================
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

ax.set_title(
    f"Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}"
)

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()


# =========================
# SAVE SINGLE HTML
# =========================
with open(out_html, "w") as f:
    f.write(fig_scatter.to_html(full_html=False, include_plotlyjs="cdn"))
    f.write(fig_bg.to_html(full_html=False, include_plotlyjs=False))
    f.write("<h2>Confusion matrix</h2>")
    f.write(plt_to_html(fig_cm))

print("Done.")
