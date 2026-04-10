from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def compute_topsis_score(df: pd.DataFrame, metric_types: dict, weight_dict: dict | None = None) -> pd.DataFrame:
    """Rank rows by TOPSIS score given metric directions and optional weights."""
    df = df.copy()
    metrics = list(metric_types.keys())

    norm_df = df[metrics].copy()
    scaler = MinMaxScaler()
    norm_df[metrics] = scaler.fit_transform(norm_df[metrics])

    for col, kind in metric_types.items():
        if kind == "cost":
            norm_df[col] = 1 - norm_df[col]

    if weight_dict is None:
        weights = np.ones(len(metrics)) / len(metrics)
    else:
        weights = np.array([weight_dict[m] for m in metrics])
        weights = weights / weights.sum()
    weighted = norm_df * weights

    ideal = weighted.max()
    anti_ideal = weighted.min()

    d_pos = np.linalg.norm(weighted - ideal, axis=1)
    d_neg = np.linalg.norm(weighted - anti_ideal, axis=1)

    score = d_neg / (d_pos + d_neg)
    df["topsis_score"] = score
    return df.sort_values("topsis_score", ascending=False).reset_index(drop=True)


__all__ = [
    "compute_topsis_score",
]
