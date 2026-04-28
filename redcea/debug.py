from __future__ import annotations

import json
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def summarize_distribution(values: np.ndarray | list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "min": float("nan"),
            "q01": float("nan"),
            "q05": float("nan"),
            "q25": float("nan"),
            "q50": float("nan"),
            "q75": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
            "max": float("nan"),
        }

    return {
        "min": float(np.min(array)),
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q50": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def prepare_debug_dir(output_path: str | Path, custom_debug_dir: str | None = None) -> Path:
    debug_dir = Path(custom_debug_dir) if custom_debug_dir else Path(output_path) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def save_numpy(path: str | Path, array: np.ndarray) -> None:
    np.save(Path(path), np.asarray(array))


def save_tsv(path: str | Path, frame: pd.DataFrame) -> None:
    frame.to_csv(Path(path), sep="\t", index=False)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _pick_first_column(frame: pd.DataFrame, candidates: list[str], default: str = "") -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    return pd.Series([default] * len(frame), index=frame.index, dtype="object")


def build_input_order_frame(
    joint_representations: pd.DataFrame,
    *,
    sample_size: int,
) -> pd.DataFrame:
    frame = joint_representations.copy().reset_index(drop=True)
    source = np.where(np.arange(len(frame)) < sample_size, "sample", "background")

    locus_series = _pick_first_column(frame, ["locus"])
    if locus_series.eq("").all():
        if "cdr3aa_beta" in frame.columns:
            locus_series = pd.Series(["TRB"] * len(frame), index=frame.index, dtype="object")
        elif "cdr3aa_alpha" in frame.columns:
            locus_series = pd.Series(["TRA"] * len(frame), index=frame.index, dtype="object")

    return pd.DataFrame(
        {
            "clone_id": frame["clone_id"],
            "junction_aa": _pick_first_column(frame, ["junction_aa", "cdr3aa", "cdr3aa_beta", "cdr3aa_alpha"]),
            "v_call": _pick_first_column(frame, ["v_call", "v", "v_beta", "v_alpha", "TRBV", "TRAV"]),
            "j_call": _pick_first_column(frame, ["j_call", "j", "j_beta", "j_alpha", "TRBJ", "TRAJ"]),
            "locus": locus_series,
            "source": source,
            "original_row_index": np.arange(len(frame), dtype=np.int64),
        }
    )


def build_cluster_membership_frame(cluster_df: pd.DataFrame) -> pd.DataFrame:
    frame = cluster_df.copy()
    if "source" not in frame.columns:
        frame["source"] = frame["clone_id"].astype(str).str.startswith("s_").map(
            {True: "sample", False: "background"}
        )
    grouped = (
        frame.loc[frame["cluster_id"] != -1, ["cluster_id", "clone_id", "source"]]
        .groupby("cluster_id", sort=True)
        .agg(
            cluster_size=("clone_id", "size"),
            number_of_sample_clonotypes=("source", lambda s: int((s == "sample").sum())),
            number_of_background_clonotypes=("source", lambda s: int((s == "background").sum())),
            clone_ids_hash=("clone_id", lambda s: sha1(",".join(map(str, sorted(s))).encode("utf-8")).hexdigest()),
        )
        .reset_index()
    )
    return grouped
