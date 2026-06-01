from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from redcea.auxiliary_cluster_metrics import append_auxiliary_cluster_metrics


def _find_run_prefix(run_dir: Path) -> str | None:
    match = next(run_dir.glob("*_summary_tcrempnet.tsv"), None)
    if match is not None:
        return match.name[: -len("_summary_tcrempnet.tsv")]
    return None


def _load_first_npy(run_dir: Path, pattern: str) -> np.ndarray | None:
    match = next(run_dir.glob(pattern), None)
    if match is None:
        return None
    return np.load(match)


def main() -> None:
    runs_root = Path("results/redcea_runs")
    updated = 0
    skipped = 0

    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        prefix = _find_run_prefix(run_dir)
        if prefix is None:
            skipped += 1
            print(f"SKIP {run_dir.name}: summary file not found")
            continue

        summary_path = run_dir / f"{prefix}_summary_tcrempnet.tsv"
        clusters_path = run_dir / f"{prefix}_tcremp_clusters.tsv"
        if not clusters_path.exists():
            skipped += 1
            print(f"SKIP {run_dir.name}: clusters file not found")
            continue

        summary_df = pd.read_csv(summary_path, sep="\t")
        if "enrichment_fdr_zbinom" not in summary_df.columns:
            skipped += 1
            print(f"SKIP {run_dir.name}: not a zbinom summary")
            continue

        clusters_df = pd.read_csv(clusters_path, sep="\t")
        total_sample = int(clusters_df["clone_id"].astype(str).str.startswith("s_").sum())
        total_background = int(clusters_df["clone_id"].astype(str).str.startswith("b_").sum())
        sample_knn_indices = _load_first_npy(run_dir, "knn_sample_sample__*.indices.npy")
        sample_knn_distances = _load_first_npy(run_dir, "knn_sample_sample__*.distances.npy")

        extended = append_auxiliary_cluster_metrics(
            summary_df,
            clusters_df,
            total_sample=total_sample,
            total_background=total_background,
            sample_knn_indices=sample_knn_indices,
            sample_knn_distances=sample_knn_distances,
        )
        extended.to_csv(summary_path, sep="\t", index=False)
        updated += 1
        print(f"OK   {run_dir.name}: wrote {summary_path.name}")

    print(f"DONE updated={updated} skipped={skipped}")


if __name__ == "__main__":
    main()
