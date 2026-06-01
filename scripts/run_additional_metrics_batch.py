from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from redcea.auxiliary_cluster_metrics import (
    AUXILIARY_CLUSTER_METRIC_COLUMNS,
    append_auxiliary_cluster_metrics,
)


SUMMARY_SUFFIX = "_summary_tcrempnet.tsv"
CLUSTERS_SUFFIX = "_tcremp_clusters.tsv"


def setup_logging(log_path: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill additional RedCEA metrics for all run directories and build unified metric tables."
    )
    parser.add_argument("--runs-root", default="results/redcea_runs")
    parser.add_argument("--metadata-path", default="results/run_metadata/clustering_runs.tsv")
    parser.add_argument("--cluster-output", default="results/metrics/redcea_all_cluster_metrics.tsv")
    parser.add_argument("--run-output", default="results/metrics/redcea_all_run_metrics.tsv")
    parser.add_argument("--log-path", default="results/metrics/run_additional_metrics_batch.log")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument(
        "--rewrite-existing",
        action="store_true",
        help="Recompute and rewrite summary files even if additional metric columns are already present.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed worker result instead of continuing.",
    )
    return parser.parse_args()


def find_run_prefix(run_dir: Path) -> str | None:
    matches = sorted(run_dir.glob(f"*{SUMMARY_SUFFIX}"))
    if not matches:
        return None
    return matches[0].name[: -len(SUMMARY_SUFFIX)]


def load_first_npy(run_dir: Path, pattern: str) -> np.ndarray | None:
    match = next(run_dir.glob(pattern), None)
    if match is None:
        return None
    return np.load(match)


def infer_source_counts(cluster_df: pd.DataFrame) -> tuple[int, int]:
    if "source" in cluster_df.columns:
        source = cluster_df["source"].fillna("").astype(str)
        total_sample = int(source.eq("sample").sum())
        total_background = int(source.eq("background").sum())
        return total_sample, total_background

    clone_ids = cluster_df["clone_id"].fillna("").astype(str)
    total_sample = int(clone_ids.str.startswith("s_").sum())
    total_background = int(clone_ids.str.startswith("b_").sum())
    return total_sample, total_background


def needs_metric_backfill(summary_df: pd.DataFrame) -> bool:
    return any(column not in summary_df.columns for column in AUXILIARY_CLUSTER_METRIC_COLUMNS)


def atomic_write_tsv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temp_path, sep="\t", index=False)
    os.replace(temp_path, destination)


def process_single_run(run_dir_str: str, rewrite_existing: bool) -> dict[str, object]:
    run_dir = Path(run_dir_str)
    started = time.perf_counter()
    result: dict[str, object] = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "status": "unknown",
        "message": "",
        "runtime_seconds": np.nan,
        "cluster_rows": 0,
        "summary_rows": 0,
        "summary_path": "",
        "metrics_added": False,
        "traceback": "",
    }
    try:
        prefix = find_run_prefix(run_dir)
        if prefix is None:
            result["status"] = "skipped"
            result["message"] = "summary file not found"
            return result

        summary_path = run_dir / f"{prefix}{SUMMARY_SUFFIX}"
        clusters_path = run_dir / f"{prefix}{CLUSTERS_SUFFIX}"
        result["summary_path"] = str(summary_path)
        if not clusters_path.exists():
            result["status"] = "skipped"
            result["message"] = "clusters file not found"
            return result

        summary_df = pd.read_csv(summary_path, sep="\t")
        result["summary_rows"] = int(len(summary_df))
        if "enrichment_fdr_zbinom" not in summary_df.columns:
            result["status"] = "skipped"
            result["message"] = "not a zbinom summary"
            return result

        if (not rewrite_existing) and (not needs_metric_backfill(summary_df)):
            result["status"] = "ok"
            result["message"] = "metrics already present"
            result["metrics_added"] = False
            return result

        cluster_df = pd.read_csv(clusters_path, sep="\t")
        result["cluster_rows"] = int(len(cluster_df))
        total_sample, total_background = infer_source_counts(cluster_df)
        sample_knn_indices = load_first_npy(run_dir, "knn_sample_sample__*.indices.npy")
        sample_knn_distances = load_first_npy(run_dir, "knn_sample_sample__*.distances.npy")

        extended_df = append_auxiliary_cluster_metrics(
            summary_df=summary_df,
            cluster_df=cluster_df,
            total_sample=total_sample,
            total_background=total_background,
            sample_knn_indices=sample_knn_indices,
            sample_knn_distances=sample_knn_distances,
        )
        atomic_write_tsv(extended_df, summary_path)
        result["status"] = "ok"
        result["message"] = "summary updated"
        result["metrics_added"] = True
        result["summary_rows"] = int(len(extended_df))
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["message"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        return result
    finally:
        result["runtime_seconds"] = float(time.perf_counter() - started)


def parse_parameter_json(value: object) -> dict[str, object]:
    if pd.isna(value):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_run_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        return pd.DataFrame(columns=["run_id"])
    return pd.read_csv(metadata_path, sep="\t")


def collect_cluster_level_table(
    *,
    runs_root: Path,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    metadata_lookup = metadata_df.drop_duplicates(subset=["run_id"], keep="last") if not metadata_df.empty else metadata_df
    frames: list[pd.DataFrame] = []

    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        prefix = find_run_prefix(run_dir)
        if prefix is None:
            continue
        summary_path = run_dir / f"{prefix}{SUMMARY_SUFFIX}"
        if not summary_path.exists():
            continue
        summary_df = pd.read_csv(summary_path, sep="\t")
        summary_df.insert(0, "run_id", run_dir.name)
        summary_df.insert(1, "run_dir", str(run_dir))
        frames.append(summary_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if metadata_lookup.empty:
        return combined

    metadata_cols = [column for column in metadata_lookup.columns if column != "run_id"]
    combined = combined.merge(metadata_lookup[["run_id"] + metadata_cols], on="run_id", how="left")
    parameter_df = pd.json_normalize(combined["parameter_json"].apply(parse_parameter_json)).add_prefix("param_")
    if not parameter_df.empty:
        combined = pd.concat([combined, parameter_df], axis=1)
    return combined


def numeric_summary(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "max": np.nan,
            "q25": np.nan,
            "q75": np.nan,
        }
    return {
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "max": float(numeric.max()),
        "q25": float(numeric.quantile(0.25)),
        "q75": float(numeric.quantile(0.75)),
    }


def build_run_level_table(cluster_df: pd.DataFrame) -> pd.DataFrame:
    if cluster_df.empty:
        return pd.DataFrame()

    metadata_like_columns = [
        "run_id",
        "run_dir",
        "dataset",
        "dataset_mode",
        "method",
        "parameter_json",
        "n_points",
        "n_clusters",
        "n_noise",
        "noise_fraction",
        "runtime_seconds",
        "status",
        "error_message",
        "error_traceback",
        "redcea_output_dir",
    ]
    metadata_like_columns.extend([column for column in cluster_df.columns if column.startswith("param_")])
    metadata_like_columns = [column for column in metadata_like_columns if column in cluster_df.columns]

    metric_columns = [
        column
        for column in cluster_df.columns
        if column not in metadata_like_columns
        and column not in {"cluster_id"}
        and pd.api.types.is_numeric_dtype(cluster_df[column])
    ]

    rows: list[dict[str, object]] = []
    for run_id, frame in cluster_df.groupby("run_id", sort=True):
        row: dict[str, object] = {"run_id": run_id}
        for column in metadata_like_columns:
            row[column] = frame.iloc[0][column]
        row["n_clusters_in_summary"] = int(len(frame))
        if "is_good_candidate" in frame.columns:
            row["good_candidate_count"] = int(frame["is_good_candidate"].fillna(False).astype(bool).sum())
            row["good_candidate_fraction"] = float(frame["is_good_candidate"].fillna(False).astype(float).mean())
        if "is_good_candidate_relaxed" in frame.columns:
            row["good_candidate_relaxed_count"] = int(frame["is_good_candidate_relaxed"].fillna(False).astype(bool).sum())
            row["good_candidate_relaxed_fraction"] = float(
                frame["is_good_candidate_relaxed"].fillna(False).astype(float).mean()
            )
        for column in metric_columns:
            stats = numeric_summary(frame[column])
            for suffix, value in stats.items():
                row[f"{column}__{suffix}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def progress_message(
    *,
    done: int,
    total: int,
    ok: int,
    skipped: int,
    failed: int,
    started_at: float,
    run_id: str,
    status: str,
    message: str,
) -> str:
    elapsed = time.perf_counter() - started_at
    return (
        "Progress {done}/{total} | ok={ok} skipped={skipped} failed={failed} | "
        "elapsed={elapsed:.1f}s | {run_id}: {status} ({message})"
    ).format(
        done=done,
        total=total,
        ok=ok,
        skipped=skipped,
        failed=failed,
        elapsed=elapsed,
        run_id=run_id,
        status=status,
        message=message,
    )


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root)
    metadata_path = Path(args.metadata_path)
    cluster_output = Path(args.cluster_output)
    run_output = Path(args.run_output)
    log_path = Path(args.log_path) if args.log_path else None

    setup_logging(log_path)
    logging.info("Starting additional metrics batch run")
    logging.info("runs_root=%s workers=%d rewrite_existing=%s", runs_root, args.workers, args.rewrite_existing)

    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    total = len(run_dirs)
    if total == 0:
        logging.warning("No run directories found under %s", runs_root)
        return 0

    started_at = time.perf_counter()
    results: list[dict[str, object]] = []
    ok = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_run, str(run_dir), args.rewrite_existing): run_dir.name
            for run_dir in run_dirs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            run_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "run_id": run_id,
                    "run_dir": str(runs_root / run_id),
                    "status": "failed",
                    "message": f"WorkerCrash: {exc}",
                    "runtime_seconds": np.nan,
                    "cluster_rows": 0,
                    "summary_rows": 0,
                    "summary_path": "",
                    "metrics_added": False,
                    "traceback": traceback.format_exc(),
                }
            results.append(result)

            status = str(result["status"])
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1

            logging.info(
                progress_message(
                    done=index,
                    total=total,
                    ok=ok,
                    skipped=skipped,
                    failed=failed,
                    started_at=started_at,
                    run_id=str(result["run_id"]),
                    status=status,
                    message=str(result["message"]),
                )
            )
            if result.get("traceback"):
                logging.error("Traceback for %s\n%s", result["run_id"], result["traceback"])
            if args.fail_fast and status == "failed":
                logging.error("Fail-fast enabled, stopping after first failure")
                break

    result_df = pd.DataFrame(results).sort_values(["status", "run_id"]).reset_index(drop=True)
    result_manifest_path = run_output.parent / "run_additional_metrics_batch_manifest.tsv"
    atomic_write_tsv(result_df, result_manifest_path)
    logging.info("Wrote worker manifest: %s", result_manifest_path)

    metadata_df = load_run_metadata(metadata_path)
    cluster_level_df = collect_cluster_level_table(runs_root=runs_root, metadata_df=metadata_df)
    run_level_df = build_run_level_table(cluster_level_df)
    atomic_write_tsv(cluster_level_df, cluster_output)
    atomic_write_tsv(run_level_df, run_output)
    logging.info("Wrote cluster-level table: %s rows=%d", cluster_output, len(cluster_level_df))
    logging.info("Wrote run-level table: %s rows=%d", run_output, len(run_level_df))

    total_elapsed = time.perf_counter() - started_at
    logging.info(
        "Finished additional metrics batch run | total=%d ok=%d skipped=%d failed=%d elapsed=%.1fs",
        total,
        ok,
        skipped,
        failed,
        total_elapsed,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
