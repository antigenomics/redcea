#!/usr/bin/env python

import argparse
import re
from pathlib import Path

import pandas as pd


# -------------------------
# utils
# -------------------------
def parse_params_from_relpath(run_dir: Path, root: Path):
    rel = run_dir.relative_to(root)
    parts = rel.parts

    params = {"kn": None, "ms": None, "eps_mode": None, "eps": None, "relpath": str(rel)}

    for p in parts:
        m = re.match(r"kn_(\d+)", p)
        if m:
            params["kn"] = int(m.group(1))

        m = re.match(r"ms_(\d+)", p)
        if m:
            params["ms"] = int(m.group(1))

        m = re.match(r"eps_([0-9\.]+)", p)
        if m:
            params["eps"] = float(m.group(1))

        m = re.match(r"eps_([a-zA-Z]+)", p)
        if m:
            params["eps_mode"] = m.group(1)

    return params


def trim_allele(x):
    if pd.isna(x):
        return x
    return str(x).split("*")[0]


def _pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_rep_table(df: pd.DataFrame, chain: str) -> pd.DataFrame:
    """
    Rep tables: expect *_alpha or *_beta fields.
    Returns df with columns: clone_id(int-ish), cdr3, v, j, key
    """
    out = df.copy()

    if "source" in out.columns:
        out = out[out["source"] != "background"].copy()

    if chain == "TRA":
        cdr3_col = _pick_col(out, ["cdr3aa_alpha", "cdr3_alpha", "cdr3"])
        v_col = _pick_col(out, ["v_alpha", "v"])
        j_col = _pick_col(out, ["j_alpha", "j"])
    else:
        cdr3_col = _pick_col(out, ["cdr3aa_beta", "cdr3_beta", "cdr3"])
        v_col = _pick_col(out, ["v_beta", "v"])
        j_col = _pick_col(out, ["j_beta", "j"])

    missing = [x for x in [cdr3_col, v_col, j_col] if x is None]
    if missing:
        raise ValueError(f"Cannot find required rep columns for chain={chain}. "
                         f"Have: {list(out.columns)}")

    out["cdr3"] = out[cdr3_col]
    out["v"] = out[v_col].apply(trim_allele)
    out["j"] = out[j_col].apply(trim_allele)

    # clone_id can be int or "clone_123"
    if "clone_id" in out.columns:
        def _to_int_like(x):
            s = str(x)
            if "_" in s:
                try:
                    return int(s.split("_")[-1])
                except Exception:
                    return pd.NA
            try:
                return int(s)
            except Exception:
                return pd.NA

        out["clone_id_int"] = out["clone_id"].apply(_to_int_like)
    else:
        out["clone_id_int"] = pd.NA

    out["key"] = list(zip(out["cdr3"], out["v"], out["j"]))
    return out[["clone_id_int", "cdr3", "v", "j", "key"]].rename(columns={"clone_id_int": "clone_id"})


def load_clone_table(rep_path: Path, chain: str) -> pd.DataFrame:
    df = pd.read_csv(rep_path, sep="\t")
    return _normalize_rep_table(df, chain)


def load_tp_set(path: str, chain: str):
    """
    TP files: your vdjdb_public_*_minStudies2.tsv
    """
    df = pd.read_csv(path, sep="\t")

    if chain == "TRA":
        cdr3_col = _pick_col(df, ["cdr3_alpha", "cdr3aa_alpha", "cdr3"])
        v_col = _pick_col(df, ["v_alpha", "v"])
        j_col = _pick_col(df, ["j_alpha", "j"])
    else:
        cdr3_col = _pick_col(df, ["cdr3_beta", "cdr3aa_beta", "cdr3"])
        v_col = _pick_col(df, ["v_beta", "v"])
        j_col = _pick_col(df, ["j_beta", "j"])

    missing = [x for x in [cdr3_col, v_col, j_col] if x is None]
    if missing:
        raise ValueError(f"Cannot find required TP columns for chain={chain}. "
                         f"Have: {list(df.columns)}")

    df["v_trim"] = df[v_col].apply(trim_allele)
    df["j_trim"] = df[j_col].apply(trim_allele)

    return set(zip(df[cdr3_col], df["v_trim"], df["j_trim"]))


def load_tn_set(rep_df: pd.DataFrame, tn_thr: int):
    """
    True-negative universe comes ONLY from rep table by clone_id threshold.
    """
    r = rep_df.copy()
    r = r[r["clone_id"].notna()]
    r = r[r["clone_id"] >= tn_thr]
    return set(r["key"])


def load_clustered_keys(cln_path: Path, chain: str):
    """
    Predicted positives: clustered sample clonotypes in enriched_clonotypes file.
    Prefer key-based matching, not clone_id-based.
    """
    df = pd.read_csv(cln_path, sep="\t")

    if "source" in df.columns:
        df = df[df["source"] == "sample"].copy()

    if "cluster_id" in df.columns:
        df = df[df["cluster_id"].notna()].copy()
    else:
        # if cluster_id is absent, treat all rows as positives (rare, but safe fallback)
        df = df.copy()

    if chain == "TRA":
        cdr3_col = _pick_col(df, ["cdr3aa_alpha", "cdr3_alpha", "cdr3"])
        v_col = _pick_col(df, ["v_alpha", "v"])
        j_col = _pick_col(df, ["j_alpha", "j"])
    else:
        cdr3_col = _pick_col(df, ["cdr3aa_beta", "cdr3_beta", "cdr3"])
        v_col = _pick_col(df, ["v_beta", "v"])
        j_col = _pick_col(df, ["j_beta", "j"])

    missing = [x for x in [cdr3_col, v_col, j_col] if x is None]
    if missing:
        raise ValueError(f"Cannot find required clonotypes columns for chain={chain} in {cln_path}. "
                         f"Have: {list(df.columns)}")

    df["v_trim"] = df[v_col].apply(trim_allele)
    df["j_trim"] = df[j_col].apply(trim_allele)
    keys = set(zip(df[cdr3_col], df["v_trim"], df["j_trim"]))
    return keys


def compute_metrics(tp: int, fn: int, fp: int, tn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root_tra", required=True)
    ap.add_argument("--root_trb", required=True)

    ap.add_argument("--prefix_tra", required=True)
    ap.add_argument("--prefix_trb", required=True)

    ap.add_argument("--rep_tra", required=True)
    ap.add_argument("--rep_trb", required=True)

    ap.add_argument("--tp_tra", required=True)
    ap.add_argument("--tp_trb", required=True)

    ap.add_argument("--tn_thr", type=int, default=200000)

    ap.add_argument("--out", default="metrics.tsv")

    args = ap.parse_args()

    rep_tra = load_clone_table(Path(args.rep_tra), "TRA")
    rep_trb = load_clone_table(Path(args.rep_trb), "TRB")

    tp_set_a = load_tp_set(args.tp_tra, "TRA")
    tp_set_b = load_tp_set(args.tp_trb, "TRB")

    # --- universes by KEY ---
    tp_universe_a = set(rep_tra[rep_tra["key"].isin(tp_set_a)]["key"])
    tp_universe_b = set(rep_trb[rep_trb["key"].isin(tp_set_b)]["key"])

    tn_universe_a = load_tn_set(rep_tra, args.tn_thr)
    tn_universe_b = load_tn_set(rep_trb, args.tn_thr)

    rows = []

    root_tra = Path(args.root_tra)
    root_trb = Path(args.root_trb)

    for tra_dir in sorted([p for p in root_tra.rglob("*") if p.is_dir()]):
        cln_tra = tra_dir / f"{args.prefix_tra}_enriched_clonotypes_tcremp.tsv"
        if not cln_tra.exists():
            continue

        rel = tra_dir.relative_to(root_tra)

        trb_dir = root_trb / rel
        cln_trb = trb_dir / f"{args.prefix_trb}_enriched_clonotypes_tcremp.tsv"
        if not cln_trb.exists():
            continue

        params = parse_params_from_relpath(tra_dir, root_tra)

        pred_keys_a = load_clustered_keys(cln_tra, "TRA")
        pred_keys_b = load_clustered_keys(cln_trb, "TRB")

        # --- compute confusion counts via set ops ---
        tp_a = len(tp_universe_a & pred_keys_a)
        fn_a = len(tp_universe_a - pred_keys_a)
        fp_a = len(tn_universe_a & pred_keys_a)
        tn_a = len(tn_universe_a - pred_keys_a)

        tp_b = len(tp_universe_b & pred_keys_b)
        fn_b = len(tp_universe_b - pred_keys_b)
        fp_b = len(tn_universe_b & pred_keys_b)
        tn_b = len(tn_universe_b - pred_keys_b)

        m_a = compute_metrics(tp_a, fn_a, fp_a, tn_a)
        m_b = compute_metrics(tp_b, fn_b, fp_b, tn_b)

        rows.append({
            "relpath": str(rel),
            **params,
            **{f"alpha_{k}": v for k, v in m_a.items()},
            **{f"beta_{k}": v for k, v in m_b.items()},
            # полезно для отладки:
            "alpha_tp_universe": len(tp_universe_a),
            "alpha_tn_universe": len(tn_universe_a),
            "beta_tp_universe": len(tp_universe_b),
            "beta_tn_universe": len(tn_universe_b),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, sep="\t", index=False)

    print(f"\nSaved: {args.out} ({len(df)} runs)")

    if len(df) == 0:
        return

    # сортируем по лучшему F1: сначала beta, если есть, иначе alpha
    sort_col = "beta_f1" if "beta_f1" in df.columns else ("alpha_f1" if "alpha_f1" in df.columns else None)
    if sort_col is None:
        return

    print("\n==============================")
    print("Top-10 runs by", sort_col)
    print("==============================")

    cols = [
        "relpath", "kn", "ms", "eps_mode", "eps",
        "alpha_precision", "alpha_recall", "alpha_f1", "alpha_FP", "alpha_TP", "alpha_FN", "alpha_TN",
        "beta_precision", "beta_recall", "beta_f1", "beta_FP", "beta_TP", "beta_FN", "beta_TN",
    ]
    cols = [c for c in cols if c in df.columns]

    print(df.sort_values(sort_col, ascending=False)[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()