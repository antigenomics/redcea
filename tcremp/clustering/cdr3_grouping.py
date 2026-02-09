import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def compute_cdr3_len(series: pd.Series) -> np.ndarray:
    """
    Strict:
    - if integer dtype -> already lengths
    - if string/object -> use .str.len() but error if non-string values exist
    - NaN forbidden
    """
    if series.isna().any():
        raise ValueError(f"cdr3 column contains NaN (count={int(series.isna().sum())})")

    if pd.api.types.is_integer_dtype(series):
        arr = series.to_numpy()
        if (arr < 0).any():
            raise ValueError("cdr3_len contains negative values")
        return arr.astype(np.int32, copy=False)

    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        is_str = series.map(lambda x: isinstance(x, str))
        if not bool(is_str.all()):
            bad = series[~is_str].head(5).tolist()
            raise TypeError(f"cdr3 column contains non-strings (examples={bad})")
        return series.str.len().to_numpy(dtype=np.int32)

    raise TypeError(f"Unsupported dtype for cdr3 column: {series.dtype}")


def build_len_to_group_id(sample_cdr3_len: np.ndarray, min_frac: float = 0.05) -> Dict[int, int]:
    """
    Bucket lengths cumulatively so each bucket contains >= min_frac of SAMPLE.
    Tail (if < threshold) is merged into the previous bucket.
    """
    if not (0 < min_frac <= 1.0):
        raise ValueError("min_frac must be in (0, 1]")

    sample_cdr3_len = np.asarray(sample_cdr3_len, dtype=np.int32)
    N = int(sample_cdr3_len.shape[0])
    if N == 0:
        raise ValueError("sample is empty")

    threshold = int(np.ceil(min_frac * N))
    vc = pd.Series(sample_cdr3_len).value_counts().sort_index()  # len -> count

    mapping: Dict[int, int] = {}
    gid = 0
    acc = 0
    bucket_lens: List[int] = []

    for L, cnt in vc.items():
        bucket_lens.append(L)
        acc += cnt
        if acc >= threshold:
            for l in bucket_lens:
                mapping[l] = gid
            gid += 1
            acc = 0
            bucket_lens = []

    if bucket_lens:
        if gid == 0:
            for l in bucket_lens:
                mapping[l] = 0
        else:
            for l in bucket_lens:
                mapping[l] = gid - 1

    # Summary
    num_groups = len(set(mapping.values()))
    group_sizes = {}
    len_per_group = {}
    for length, gid in mapping.items():
        if gid not in group_sizes:
            group_sizes[gid] = 0
            len_per_group[gid] = []
        group_sizes[gid] += vc.get(length, 0)
        len_per_group[gid].append(length)
    
    summary = f"\nCDR3 Length Grouping Summary:\n"
    summary += f"  Total groups: {num_groups}\n"
    summary += f"  Total unique lengths: {len(mapping)}\n"
    summary += f"  Min/max group size: {min(group_sizes.values())}/{max(group_sizes.values())} elements\n"
    for gid in sorted(group_sizes.keys()):
        sizes = len_per_group[gid]
        summary += f"  Group {gid}: {len(sizes)} lengths {{{min(sizes)}-{max(sizes)}}}, {group_sizes[gid]} elements\n"
    logging.info(summary)

    return mapping


def map_len_to_group_id(
    cdr3_len: np.ndarray,
    len_to_gid: Dict[int, int],
    unknown_len_policy: str = "nearest",  # nearest + warning by default
) -> np.ndarray:
    """
    Map cdr3_len -> group_id using len_to_gid (learned on SAMPLE).

    unknown_len_policy:
      - "nearest": for missing lengths, assign nearest known length's group_id + WARNING
      - "error":   raise if any missing
    """
    if unknown_len_policy not in {"nearest", "error"}:
        raise ValueError("unknown_len_policy must be 'nearest' or 'error'")

    cdr3_len = np.asarray(cdr3_len, dtype=np.int32)
    out = np.empty_like(cdr3_len, dtype=np.int32)

    missing = sorted({int(L) for L in np.unique(cdr3_len) if int(L) not in len_to_gid})
    if missing and unknown_len_policy == "error":
        raise ValueError(
            f"Found CDR3 lengths not present in SAMPLE mapping: {missing}. "
            "Set unknown_len_policy='nearest' to fallback explicitly."
        )

    if not missing:
        for i, L in enumerate(cdr3_len):
            out[i] = len_to_gid[int(L)]
        return out

    # nearest fallback with explicit warning
    known_lens = np.array(sorted(len_to_gid.keys()), dtype=np.int32)
    known_gids = np.array([len_to_gid[int(L)] for L in known_lens], dtype=np.int32)

    pairs: List[Tuple[int, int, int]] = []
    for L in missing:
        pos = int(np.searchsorted(known_lens, L))
        if pos == 0:
            nearest = int(known_lens[0])
        elif pos == len(known_lens):
            nearest = int(known_lens[-1])
        else:
            left = int(known_lens[pos - 1])
            right = int(known_lens[pos])
            nearest = left if (L - left) <= (right - L) else right
        pairs.append((L, nearest, int(len_to_gid[nearest])))

    logging.warning(
        "CDR3 lengths missing in SAMPLE mapping; using nearest known length:\n%s",
        "\n".join([f"  len={L} -> nearest_len={nn} -> group_id={gid}" for (L, nn, gid) in pairs])
    )

    for i, L in enumerate(cdr3_len):
        L = int(L)
        if L in len_to_gid:
            out[i] = len_to_gid[L]
        else:
            pos = int(np.searchsorted(known_lens, L))
            if pos == 0:
                out[i] = known_gids[0]
            elif pos == len(known_lens):
                out[i] = known_gids[-1]
            else:
                left = int(known_lens[pos - 1])
                right = int(known_lens[pos])
                nn = left if (L - left) <= (right - L) else right
                out[i] = len_to_gid[nn]

    return out
