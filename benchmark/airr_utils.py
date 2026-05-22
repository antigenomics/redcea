from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def normalize_segment(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
        .str.replace(r"\*.*$", "", regex=True)
        .str.replace("/", "_", regex=False)
    )


def required_present(frame: pd.DataFrame, candidates: list[str], label: str) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    raise KeyError(
        "Could not find a {0} column. Tried: {1}. Available columns: {2}".format(
            label,
            ", ".join(candidates),
            ", ".join(map(str, frame.columns)),
        )
    )


def ensure_nonempty_series(series: pd.Series, *, label: str, source_label: str) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip()
    empty_mask = normalized.eq("")
    if empty_mask.any():
        raise ValueError(
            "Parsed {0} contains empty values for {1}: empty_rows={2}, total_rows={3}".format(
                label,
                source_label,
                int(empty_mask.sum()),
                len(series),
            )
        )
    return series


def infer_chain(frame: pd.DataFrame, default: str = "TRB") -> pd.Series:
    if "chain" in frame.columns:
        return frame["chain"].fillna(default).astype(str)
    if "locus" in frame.columns:
        locus = frame["locus"].fillna(default).astype(str)
        return locus.replace({"beta": "TRB", "alpha": "TRA"})
    return pd.Series([default] * len(frame), index=frame.index, dtype="object")


def read_airr_like_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".gz":
        return pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    return pd.read_csv(path, sep="\t", low_memory=False)


def standardize_metadata_frame(frame: pd.DataFrame, *, chain_default: str = "TRB") -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["cdr3"] = ensure_nonempty_series(
        required_present(
            frame,
            ["cdr3", "cdr3aa", "junction_aa", "cdr3aa_beta", "cdr3aa_alpha", "cdr3_beta_aa", "cdr3_alpha_aa"],
            "cdr3",
        ),
        label="metadata frame",
        source_label="cdr3",
    )
    out["v_gene"] = ensure_nonempty_series(
        normalize_segment(
            required_present(
                frame,
                ["v_gene", "v_call", "v.segm", "v_beta", "v_alpha", "TRBV", "TRAV", "TRBV_IMGT", "v"],
                "v_gene",
            )
        ),
        label="metadata frame",
        source_label="v_gene",
    )
    out["j_gene"] = ensure_nonempty_series(
        normalize_segment(
            required_present(
                frame,
                ["j_gene", "j_call", "j.segm", "j_beta", "j_alpha", "TRBJ", "TRAJ", "TRBJ_IMGT", "j"],
                "j_gene",
            )
        ),
        label="metadata frame",
        source_label="j_gene",
    )
    out["chain"] = infer_chain(frame, default=chain_default)
    return out


def clone_id_candidates(frame: pd.DataFrame, *, sample_label: str | None = None) -> pd.DataFrame:
    candidates = pd.DataFrame(index=frame.index)
    raw_ids = None
    if "clone_id" in frame.columns:
        raw_ids = frame["clone_id"].astype(str)
        candidates["clone_id"] = raw_ids
    elif "clonotype_id" in frame.columns:
        raw_ids = frame["clonotype_id"].astype(str)
        candidates["clone_id"] = raw_ids

    one_based = pd.Series(np.arange(1, len(frame) + 1), index=frame.index).astype(str)
    zero_based = pd.Series(np.arange(len(frame)), index=frame.index).astype(str)
    candidates["row_1_based"] = one_based
    candidates["row_0_based"] = zero_based
    if sample_label is not None:
        prefix = "s_" if sample_label == "sample" else "b_"
        candidates[f"{prefix}row_1_based"] = prefix + one_based
        candidates[f"{prefix}row_0_based"] = prefix + zero_based
        if raw_ids is not None:
            candidates[f"{prefix}clone_id"] = prefix + raw_ids.str.replace(r"^[sb]_", "", regex=True)
    return candidates


def best_clone_id_key(left: pd.DataFrame, right: pd.DataFrame) -> tuple[tuple[str, str] | None, int]:
    best_column = None
    best_overlap = 0
    for left_column in left.columns:
        left_values = set(left[left_column].dropna().astype(str))
        if not left_values:
            continue
        for right_column in right.columns:
            overlap = len(left_values & set(right[right_column].dropna().astype(str)))
            if overlap > best_overlap:
                best_column = (left_column, right_column)
                best_overlap = overlap
    return best_column, best_overlap
