from __future__ import annotations

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_volcano(
    df: pd.DataFrame,
    pval_threshold: float = 0.05,
    fold_threshold: float = 0,
    sample_name: str = "",
    ax=None,
    layers=None,
    layer_exclusive: bool = True,
    combo_style=None,
    point_size=None,
):
    """Plot cluster enrichment as a volcano plot with optional overlay layers."""
    df = df.copy()
    eps = 1e-10
    df["log10_pval"] = -np.log10(df["enrichment_pvalue_zbinom"] + eps)
    df["significant"] = (df["enrichment_fdr_zbinom"] < pval_threshold) & (df["log_fold_change"] > fold_threshold)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    main_color = "#FFCB9A"
    base_color = "#D2E8E3"

    if combo_style is None:
        combo_style = {
            "marker": "D",
            "color": "purple",
            "edgecolor": "black",
            "linewidth": 0.4,
            "alpha": 1.0,
            "label": "combo",
        }

    layer_cols = [layer["col"] for layer in (layers or []) if layer.get("col") in df.columns]
    layer_any = np.zeros(len(df), dtype=bool)
    for col in layer_cols:
        layer_any |= df[col].fillna(False).values

    base_mask = ~layer_any if layer_exclusive else np.ones(len(df), dtype=bool)
    base_df = df.loc[base_mask]
    if len(base_df):
        sns.scatterplot(
            data=base_df,
            x="log_fold_change",
            y="log10_pval",
            hue="significant",
            palette={True: main_color, False: base_color},
            edgecolor="black",
            linewidth=0.3,
            alpha=0.85,
            ax=ax,
            **({"s": point_size} if point_size else {}),
        )

    if len(layer_cols) >= 2:
        combo_mask = df[layer_cols].sum(axis=1) > 1
        combo_df = df.loc[combo_mask]
        if len(combo_df):
            sns.scatterplot(
                data=combo_df,
                x="log_fold_change",
                y="log10_pval",
                marker=combo_style["marker"],
                color=combo_style["color"],
                edgecolor=combo_style["edgecolor"],
                linewidth=combo_style["linewidth"],
                alpha=combo_style["alpha"],
                ax=ax,
                label=combo_style["label"],
                **({"s": point_size} if point_size else {}),
            )
        df = df.loc[~combo_mask]

    if layers:
        for layer in layers:
            col = layer["col"]
            mask = df[col].fillna(False).astype(bool).values
            if not mask.any():
                continue
            sns.scatterplot(
                data=df.loc[mask],
                x="log_fold_change",
                y="log10_pval",
                marker=layer.get("marker", "o"),
                color=layer.get("color", "red"),
                edgecolor=layer.get("edgecolor", "black"),
                linewidth=layer.get("linewidth", 0.4),
                alpha=layer.get("alpha", 1.0),
                ax=ax,
                label=layer.get("label", col),
                **({"s": point_size} if point_size else {}),
            )

    sig_mask = df["enrichment_fdr_zbinom"] < pval_threshold
    bonferroni_thresh = float(df.loc[sig_mask, "enrichment_pvalue_zbinom"].max()) if sig_mask.any() else 1.0
    log10_bonferroni = -np.log10(bonferroni_thresh + eps)

    ax.axvline(fold_threshold, ls="--", color="black")
    ax.axhline(-np.log10(pval_threshold + eps), ls="--", color="black")
    ax.axhline(log10_bonferroni, ls=":", color="blue")

    num_significant = int(df["significant"].sum())
    ax.set_xlabel("log2(Fold Enrichment)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(f"{sample_name}: # clusters (fdr < {pval_threshold}): {num_significant}")
    ax.legend()

    return ax


def plot_logo(clonotypes):
    """Render a sequence logo for clonotype CDR3 amino-acid sequences."""
    mat_df = logomaker.alignment_to_matrix(clonotypes)
    logomaker.Logo(mat_df, color_scheme="skylign_protein", ax=None)


__all__ = [
    "plot_logo",
    "plot_volcano",
]
