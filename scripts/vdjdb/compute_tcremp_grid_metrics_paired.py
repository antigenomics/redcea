import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


# -------------------------
# parsing params from path
# -------------------------
def parse_params_from_relpath(run_dir: Path, root: Path) -> dict:
    rel = run_dir.relative_to(root)
    parts = [p for p in rel.parts if p and p != "."]

    params = {
        "kn": None,
        "ms": None,
        "eps_mode": None,
        "eps": None,
        "relpath": str(rel),
        "tags": [],
    }

    def set_if_none(key, val):
        if params.get(key) is None:
            params[key] = val

    for part in parts:
        m = re.fullmatch(r"kn_(\d+)", part)
        if m:
            set_if_none("kn", int(m.group(1)))
            continue

        tokens = part.split("__") if "__" in part else [part]

        for tok in tokens:
            m = re.fullmatch(r"ms_(\d+)", tok)
            if m:
                set_if_none("ms", int(m.group(1)))
                continue

            m = re.fullmatch(r"eps_([A-Za-z]+)", tok)
            if m:
                set_if_none("eps_mode", m.group(1))
                continue

            m = re.fullmatch(r"eps_([0-9]*\.?[0-9]+)", tok)
            if m:
                set_if_none("eps", float(m.group(1)))
                continue

            m = re.fullmatch(r"([A-Za-z]+)_([0-9]*\.?[0-9]+)", tok)
            if m:
                k, v = m.group(1), m.group(2)
                val = float(v) if "." in v else int(v)
                if k not in params or params[k] is None:
                    params[k] = val
                else:
                    params["tags"].append(tok)
                continue

            m = re.fullmatch(r"([A-Za-z]+)_([A-Za-z][A-Za-z0-9\-]*)", tok)
            if m:
                k, v = m.group(1), m.group(2)
                if k not in params or params[k] is None:
                    params[k] = v
                else:
                    params["tags"].append(tok)
                continue

            params["tags"].append(tok)

    params["tags"] = "__".join([t for t in params["tags"] if t])
    return params


# -------------------------
# helpers
# -------------------------
def trim_allele(series: pd.Series) -> pd.Series:
    s = series.copy()
    s = s.where(s.notna(), other=pd.NA)
    return s.astype("string").str.replace(r"\*.*$", "", regex=True)


def chain_spec(chain: str) -> dict:
    chain = chain.upper()
    if chain not in {"TRA", "TRB"}:
        raise ValueError(f"Unsupported chain: {chain}. Use TRA or TRB.")

    if chain == "TRB":
        return {
            "chain": "TRB",
            "prefix_chain": "trb",
            "info_cdr3_col": "cdr3aa_beta",
            "info_v_col": "v_beta",
            "info_j_col": "j_beta",
            "val_cdr3_col": "cdr3_beta_aa",
            "val_v_col": "TRBV",
            "val_j_col": "TRBJ",
        }

    return {
        "chain": "TRA",
        "prefix_chain": "tra",
        "info_cdr3_col": "cdr3aa_alpha",
        "info_v_col": "v_alpha",
        "info_j_col": "j_alpha",
        "val_cdr3_col": "cdr3_alpha_aa",
        "val_v_col": "TRAV",
        "val_j_col": "TRAJ",
    }


def default_prefix(chain: str, epitope: str) -> str:
    spec = chain_spec(chain)
    return f"{spec['prefix_chain']}_vdjdb_{epitope}"


def default_representations_tsv(chain: str, epitope: str) -> str:
    pref = default_prefix(chain, epitope)
    return f"/projects/immunestatus/vdjdb/tcremp/{pref}_tcremp_representations.tsv"


def find_clonotypes_file(run_dir: Path, prefix: str) -> Path | None:
    cln = run_dir / f"{prefix}_enriched_clonotypes_tcremp.tsv"
    return cln if cln.exists() else None


def load_info(rep_path: Path) -> pd.DataFrame:
    info = pd.read_csv(rep_path, sep="\t")
    if "clone_id" not in info.columns:
        raise KeyError(f"Representations missing 'clone_id': {rep_path}")
    return info


def load_clustered_key_set(rep_path: Path, clonotypes_path: Path, chain: str) -> set[tuple]:
    """
    Return a set of (cdr3, Vtrim, Jtrim) tuples for clustered sample clonotypes.
    """
    spec = chain_spec(chain)
    info = load_info(rep_path)

    clonotypes = pd.read_csv(clonotypes_path, sep="\t")
    clonotypes_sample = clonotypes[clonotypes.source == "sample"].copy()
    if clonotypes_sample.empty:
        return set()

    clonotypes_sample["clone_id"] = clonotypes_sample["clone_id"].apply(
        lambda x: int(str(x).split("_")[1])
    )
    clustered_ids = set(
        clonotypes_sample.loc[clonotypes_sample["cluster_id"].notna(), "clone_id"].astype(int).tolist()
    )

    df = info.loc[info["clone_id"].isin(clustered_ids)].copy()

    need = [spec["info_cdr3_col"], spec["info_v_col"], spec["info_j_col"]]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(
            f"Representations missing columns for {chain}: {missing}. Available: {list(info.columns)}"
        )

    cdr3 = df[spec["info_cdr3_col"]].astype("string")
    v = trim_allele(df[spec["info_v_col"]])
    j = trim_allele(df[spec["info_j_col"]])

    key_set = set(zip(cdr3.fillna(pd.NA), v.fillna(pd.NA), j.fillna(pd.NA)))
    # выкинем ключи с NA (на всякий)
    key_set = {k for k in key_set if pd.notna(k[0]) and pd.notna(k[1]) and pd.notna(k[2])}
    return key_set


def load_tcrvdb_pairs(validator_csv: Path, epitope: str, padj_thr: float) -> pd.DataFrame:
    mm = (
        pd.read_csv(validator_csv)
        .drop(columns=["Unnamed: 0"], errors="ignore")
        .dropna(subset=["padj"])
    )
    mm = mm[mm["epitope_aa"] == epitope].copy()
    mm["valid"] = mm["padj"] < padj_thr

    need = ["cdr3_alpha_aa", "TRAV", "TRAJ", "cdr3_beta_aa", "TRBV", "TRBJ", "valid"]
    missing = [c for c in need if c not in mm.columns]
    if missing:
        raise KeyError(f"Validator CSV missing columns: {missing}. Available: {list(mm.columns)}")

    # нормализуем аллели в самом валидаторе тоже
    mm["TRAV"] = trim_allele(mm["TRAV"])
    mm["TRAJ"] = trim_allele(mm["TRAJ"])
    mm["TRBV"] = trim_allele(mm["TRBV"])
    mm["TRBJ"] = trim_allele(mm["TRBJ"])

    # оставим уникальные пары; valid по паре — any()
    pairs = (
        mm[need]
        .groupby(["cdr3_alpha_aa", "TRAV", "TRAJ", "cdr3_beta_aa", "TRBV", "TRBJ"], as_index=False)
        .any()
    )
    return pairs


def compute_confusion_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    TP, FN = int(cm[0, 0]), int(cm[0, 1])
    FP, TN = int(cm[1, 0]), int(cm[1, 1])
    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_eval": int(len(y_true)),
    }


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Paired-chain metrics without matching clone_id: "
                    "predicted if exists clustered alpha and/or exists clustered beta matching TCRvdb keys."
    )
    ap.add_argument("--root_tra", required=True, help="Grid root dir for TRA runs.")
    ap.add_argument("--root_trb", required=True, help="Grid root dir for TRB runs.")
    ap.add_argument("--epitope", required=True)
    ap.add_argument("--padj_thr", type=float, default=1e-5)
    ap.add_argument("--validator_csv", default="/home/evlasova/tcrempnet/data/01_05_2025_TCRvdb.csv")
    ap.add_argument("--out", default="metrics_grid.tsv")

    ap.add_argument("--prefix_tra", default=None)
    ap.add_argument("--prefix_trb", default=None)
    ap.add_argument("--representations_tra", default=None)
    ap.add_argument("--representations_trb", default=None)
    args = ap.parse_args()

    root_tra = Path(args.root_tra)
    root_trb = Path(args.root_trb)
    if not root_tra.exists():
        raise FileNotFoundError(f"--root_tra does not exist: {root_tra}")
    if not root_trb.exists():
        raise FileNotFoundError(f"--root_trb does not exist: {root_trb}")

    if args.prefix_tra is None:
        args.prefix_tra = default_prefix("TRA", args.epitope)
    if args.prefix_trb is None:
        args.prefix_trb = default_prefix("TRB", args.epitope)

    if args.representations_tra is None:
        args.representations_tra = default_representations_tsv("TRA", args.epitope)
    if args.representations_trb is None:
        args.representations_trb = default_representations_tsv("TRB", args.epitope)

    rep_tra = Path(args.representations_tra)
    rep_trb = Path(args.representations_trb)
    if not rep_tra.exists():
        raise FileNotFoundError(f"TRA representations not found: {rep_tra}")
    if not rep_trb.exists():
        raise FileNotFoundError(f"TRB representations not found: {rep_trb}")

    # --- ground truth universe: TCRvdb paired entries ---
    pairs = load_tcrvdb_pairs(Path(args.validator_csv), args.epitope, args.padj_thr)

    # derive chain-level GT by triple any(valid)
    gt_a = pairs.groupby(["cdr3_alpha_aa", "TRAV", "TRAJ"], as_index=False)["valid"].any().rename(columns={"valid": "gt_alpha"})
    gt_b = pairs.groupby(["cdr3_beta_aa", "TRBV", "TRBJ"], as_index=False)["valid"].any().rename(columns={"valid": "gt_beta"})

    # attach to pairs
    pairs = pairs.merge(gt_a, on=["cdr3_alpha_aa", "TRAV", "TRAJ"], how="left")
    pairs = pairs.merge(gt_b, on=["cdr3_beta_aa", "TRBV", "TRBJ"], how="left")
    pairs["gt_alpha"] = pairs["gt_alpha"].fillna(False).astype(bool)
    pairs["gt_beta"] = pairs["gt_beta"].fillna(False).astype(bool)
    pairs["gt_or"] = pairs["gt_alpha"] | pairs["gt_beta"]
    pairs["gt_and"] = pairs["gt_alpha"] & pairs["gt_beta"]

    # --- iterate paired runs by relpath ---
    rows = []
    tra_dirs = [p for p in root_tra.rglob("*") if p.is_dir()]
    for tra_run_dir in sorted(tra_dirs):
        tra_cln = find_clonotypes_file(tra_run_dir, args.prefix_tra)
        if tra_cln is None:
            continue

        rel = tra_run_dir.relative_to(root_tra)
        trb_run_dir = root_trb / rel
        trb_cln = find_clonotypes_file(trb_run_dir, args.prefix_trb) if trb_run_dir.exists() else None

        parsed = parse_params_from_relpath(tra_run_dir, root_tra)

        if trb_cln is None:
            rows.append({
                "status": "missing_trb",
                "epitope": args.epitope,
                "relpath": str(rel),
                "run_dir_tra": str(tra_run_dir),
                "run_dir_trb": str(trb_run_dir),
                **parsed,
            })
            continue

        # key sets of clustered clonotypes (chain-specific)
        clustered_alpha_keys = load_clustered_key_set(rep_tra, tra_cln, "TRA")
        clustered_beta_keys = load_clustered_key_set(rep_trb, trb_cln, "TRB")

        # predictions per TCRvdb paired entry
        pred = pairs.copy()
        pred_alpha_key = list(zip(pred["cdr3_alpha_aa"], pred["TRAV"], pred["TRAJ"]))
        pred_beta_key = list(zip(pred["cdr3_beta_aa"], pred["TRBV"], pred["TRBJ"]))

        pred["pred_alpha"] = [k in clustered_alpha_keys for k in pred_alpha_key]
        pred["pred_beta"] = [k in clustered_beta_keys for k in pred_beta_key]
        pred["pred_or"] = pred["pred_alpha"] | pred["pred_beta"]
        pred["pred_and"] = pred["pred_alpha"] & pred["pred_beta"]

        combos = [
            ("alpha", "alpha"),
            ("beta", "beta"),
            ("or", "or"),
            ("and", "and"),
            # если нужно смотреть "валидация по любой цепи" vs "предсказали только по α/β"
            ("or", "alpha"),
            ("or", "beta"),
        ]

        for gt_mode, pred_mode in combos:
            y_true = pred[f"gt_{gt_mode}"].astype(bool)
            y_pred = pred[f"pred_{pred_mode}"].astype(bool)
            m = compute_confusion_metrics(y_true, y_pred)

            rows.append({
                "status": "ok",
                "epitope": args.epitope,
                "relpath": str(rel),
                "run_dir_tra": str(tra_run_dir),
                "run_dir_trb": str(trb_run_dir),
                "gt_mode": gt_mode,
                "pred_mode": pred_mode,
                **parsed,
                **m,
                "n_pred_alpha": int(pred["pred_alpha"].sum()),
                "n_pred_beta": int(pred["pred_beta"].sum()),
            })

    out_df = pd.DataFrame(rows)
    if "f1" in out_df.columns:
        out_df = out_df.sort_values(["status", "f1"], ascending=[True, False])

    out_path = Path(args.out)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved: {out_path} ({len(out_df)} rows)")

    if "f1" in out_df.columns:
        print("\nTop-10 (status=ok) by F1:")
        cols = [c for c in ["gt_mode", "pred_mode", "relpath", "kn", "ms", "eps_mode", "eps", "TP", "FP", "TN", "FN", "precision", "recall", "f1"] if c in out_df.columns]
        print(out_df[out_df["status"] == "ok"][cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
