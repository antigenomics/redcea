from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd


def _pgen_worker(seq: str) -> float:
    try:
        from mir.basic.pgen import OlgaModel

        olga = OlgaModel()
        return float(olga.compute_pgen_cdr3aa(seq))
    except Exception:
        return np.nan


def compute_pgen_pool(
    df: pd.DataFrame,
    seq_col: str = "cdr3aa_beta",
    out_col: str = "pgen",
    processes: int = 32,
    chunksize: int = 500,
    progress: bool = True,
) -> pd.Series:
    """Compute pgen values for amino-acid CDR3 sequences in parallel."""
    seqs = df[seq_col].astype(str).tolist()

    iterator = seqs
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(seqs, desc="pgen", mininterval=0.2)
        except Exception:
            pass

    with Pool(processes=processes) as pool:
        vals = list(pool.imap(_pgen_worker, iterator, chunksize=chunksize))

    return pd.Series(vals, index=df.index, name=out_col)


__all__ = [
    "compute_pgen_pool",
]
