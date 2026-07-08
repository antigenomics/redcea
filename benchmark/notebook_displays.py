from __future__ import annotations

from pathlib import Path

import pandas as pd


def _try_import_display():
    try:
        from IPython.display import Image, Markdown, display  # type: ignore
    except Exception:
        return None, None, None
    return display, Markdown, Image


def display_markdown(text: str) -> None:
    display, Markdown, _Image = _try_import_display()
    if display is not None and Markdown is not None:
        display(Markdown(text))
    else:
        print(text)


def display_image(path: str | Path, *, alt: str | None = None) -> None:
    path = Path(path)
    display, _Markdown, Image = _try_import_display()
    if display is not None and Image is not None and path.exists():
        display(Image(filename=str(path)))
    else:
        print(alt or path)


def display_frame(frame: pd.DataFrame, *, rows: int = 10) -> pd.DataFrame:
    subset = frame.head(rows)
    display, _Markdown, _Image = _try_import_display()
    if display is not None:
        display(subset)
    else:
        print(subset.to_string(index=False))
    return subset


def show_vdjdb_notebook_outputs(metrics_df: pd.DataFrame, repo_root: str | Path) -> None:
    repo_root = Path(repo_root)
    summary = (
        metrics_df.groupby("method", as_index=False)[["f1", "precision", "recall", "weighted_cluster_purity"]]
        .mean()
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )
    best = summary.iloc[0]
    display_markdown(
        "## Key Takeaways\n"
        f"- Best mean VDJdb F1: `{best['method']}` ({best['f1']:.3f})\n"
        f"- Best mean recall is also led by `{best['method']}` ({best['recall']:.3f})"
    )
    display_markdown("## Mean Metrics By Method")
    display_frame(summary, rows=len(summary))
    display_markdown("## Per-Run Metrics")
    display_frame(metrics_df.sort_values(["epitope", "f1"], ascending=[True, False]), rows=len(metrics_df))
    display_markdown("## Figures")
    display_image(repo_root / "figures" / "clustering_strategy" / "fig2_vdjdb_f1_precision_recall.png")
    display_image(repo_root / "figures" / "clustering_strategy" / "fig3_vdjdb_cluster_concentration.png")


def show_yfv_enrichment_outputs(summary_df: pd.DataFrame, repo_root: str | Path) -> None:
    repo_root = Path(repo_root)
    ranked = summary_df.sort_values("weighted_enrichment_score", ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    method_summary = (
        summary_df.groupby("method", as_index=False)[
            ["number_of_significant_enriched_clusters", "retained_fraction", "background_contamination", "weighted_enrichment_score"]
        ]
        .mean()
        .sort_values("weighted_enrichment_score", ascending=False)
        .reset_index(drop=True)
    )
    display_markdown(
        "## Key Takeaways\n"
        f"- Strongest YFV enrichment score in this table: `{best['run_id']}` ({best['weighted_enrichment_score']:.3f})\n"
        f"- Mean YFV enrichment leader across methods: `{method_summary.iloc[0]['method']}`"
    )
    display_markdown("## Mean Metrics By Method")
    display_frame(method_summary, rows=len(method_summary))
    display_markdown("## Per-Run YFV Enrichment Metrics")
    display_frame(ranked, rows=len(ranked))
    display_markdown("## Figure")
    display_image(repo_root / "figures" / "clustering_strategy" / "fig4_yfv_enrichment_output.png")


def show_yfv_known_recovery_outputs(recovery_df: pd.DataFrame, repo_root: str | Path) -> None:
    repo_root = Path(repo_root)
    grouped = (
        recovery_df.groupby(["donor_id", "method"], as_index=False)[["is_candidate_clustered", "is_in_significant_cluster"]]
        .mean()
        .sort_values(["donor_id", "is_in_significant_cluster", "is_candidate_clustered"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    method_summary = (
        recovery_df.groupby("method", as_index=False)[["is_candidate_clustered", "is_in_significant_cluster"]]
        .mean()
        .sort_values("is_in_significant_cluster", ascending=False)
        .reset_index(drop=True)
    )
    display_markdown(
        "## Key Takeaways\n"
        f"- Best mean recovery into significant clusters: `{method_summary.iloc[0]['method']}` ({method_summary.iloc[0]['is_in_significant_cluster']:.3f})\n"
        f"- Candidate-level recovery is highest for `{method_summary.sort_values('is_candidate_clustered', ascending=False).iloc[0]['method']}`"
    )
    display_markdown("## Mean Recovery By Method")
    display_frame(method_summary, rows=len(method_summary))
    display_markdown("## Recovery By Donor And Method")
    display_frame(grouped, rows=len(grouped))
    display_markdown("## Figure")
    display_image(repo_root / "figures" / "clustering_strategy" / "fig5_yfv_known_clonotype_recovery.png")


def show_cross_donor_overlap_outputs(overlap_df: pd.DataFrame, repo_root: str | Path) -> None:
    repo_root = Path(repo_root)
    ranked = overlap_df.sort_values("overlap_fold_change", ascending=False).reset_index(drop=True)
    method_summary = (
        overlap_df.groupby("method", as_index=False)[["overlap_fold_change", "enriched_overlap_fraction", "raw_overlap_fraction"]]
        .mean()
        .sort_values("overlap_fold_change", ascending=False)
        .reset_index(drop=True)
    )
    display_markdown(
        "## Key Takeaways\n"
        f"- Strongest mean cross-donor enrichment-over-background overlap: `{method_summary.iloc[0]['method']}` ({method_summary.iloc[0]['overlap_fold_change']:.3f})\n"
        f"- Highest single donor-pair overlap fold change: `{ranked.iloc[0]['run_id_a']}` vs `{ranked.iloc[0]['run_id_b']}`"
    )
    display_markdown("## Mean Overlap Metrics By Method")
    display_frame(method_summary, rows=len(method_summary))
    display_markdown("## Donor-Pair Overlap Table")
    display_frame(ranked, rows=len(ranked))
    display_markdown("## Figure")
    display_image(repo_root / "figures" / "clustering_strategy" / "fig6_yfv_cross_donor_overlap.png")


def show_final_summary_outputs(comparison_df: pd.DataFrame, repo_root: str | Path) -> None:
    repo_root = Path(repo_root)
    ranked = comparison_df.reset_index(drop=True)
    best_vdjdb = ranked.sort_values("vdjdb_mean_f1", ascending=False).iloc[0]
    best_yfv = ranked.sort_values(
        ["yfv_mean_enrichment_recovery_rate", "yfv_mean_weighted_enrichment_score"],
        ascending=[False, False],
    ).iloc[0]
    display_markdown(
        "## Final Takeaways\n"
        f"- Best VDJdb method: `{best_vdjdb['method']}` (mean F1 {best_vdjdb['vdjdb_mean_f1']:.3f})\n"
        f"- Best YFV method by current benchmark rule: `{best_yfv['method']}`\n"
        f"- Strongest cross-donor overlap fold change overall: `{ranked.sort_values('yfv_mean_overlap_fold_change', ascending=False).iloc[0]['method']}`"
    )
    display_markdown("## Combined Comparison Table")
    display_frame(ranked, rows=len(ranked))
    report_path = repo_root / "reports" / "clustering_strategy_summary.md"
    if report_path.exists():
        display_markdown("## Written Summary Report")
        display_markdown(report_path.read_text(encoding="utf-8"))
    display_markdown("## Figure")
    display_image(repo_root / "figures" / "clustering_strategy" / "final_method_summary.png")
