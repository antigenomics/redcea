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
    chain = chain.lower()
    if chain not in {"tra", "trb"}:
        raise ValueError(f"Unsupported --chain: {chain}. Use tra or trb.")

    if chain == "trb":
        return {
            "chain": "TRB",
            "prefix_chain": "trb",
            "info_cdr3_col": "cdr3aa_beta",
            "info_v_col": "v_beta",
            "info_j_col": "j_beta",
            "tp_cdr3_candidates": ["cdr3_beta_aa", "cdr3aa_beta", "cdr3_beta", "cdr3aa"],
            "tp_v_candidates": ["TRBV", "v_beta", "v_gene", "v"],
            "tp_j_candidates": ["TRBJ", "j_beta", "j_gene", "j"],
        }

    return {
        "chain": "TRA",
        "prefix_chain": "tra",
        "info_cdr3_col": "cdr3aa_alpha",
        "info_v_col": "v_alpha",
        "info_j_col": "j_alpha",
        "tp_cdr3_candidates": ["cdr3_alpha_aa", "cdr3aa_alpha", "cdr3_alpha", "cdr3aa"],
        "tp_v_candidates": ["TRAV", "v_alpha", "v_gene", "v"],
        "tp_j_candidates": ["TRAJ", "j_alpha", "j_gene", "j"],
    }


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of candidate columns exist: {candidates}. Available: {list(df.columns)}")


def load_true_positive_set(tp_tsv: Path, epitope: str, chain: str) -> set[tuple[str, str, str]]:
    """
    Returns set of (cdr3aa, V_trimmed, J_trimmed) for true positives for this epitope.
    If tp file has an epitope column, we filter by args.epitope; otherwise use whole file.
    """
    spec = chain_spec(chain)
    tp = pd.read_csv(tp_tsv, sep="\t")

    # optional filter by epitope if such column exists
    for epi_col in ["epitope", "epitope_aa", "Epitope", "epitope_seq"]:
        if epi_col in tp.columns:
            tp = tp[tp[epi_col] == epitope].copy()
            break

    cdr3_col = _pick_col(tp, spec["tp_cdr3_candidates"])
    v_col = _pick_col(tp, spec["tp_v_candidates"])
    j_col = _pick_col(tp, spec["tp_j_candidates"])

    tp[cdr3_col] = tp[cdr3_col].astype("string")
    tp[v_col] = trim_allele(tp[v_col])
    tp[j_col] = trim_allele(tp[j_col])

    # drop NAs
    tp = tp.dropna(subset=[cdr3_col, v_col, j_col])

    return set(zip(tp[cdr3_col].tolist(), tp[v_col].tolist(), tp[j_col].tolist()))


def find_run_files(run_dir: Path, prefix: str, rep_path: Path):
    cln = run_dir / f"{prefix}_enriched_clonotypes_tcremp.tsv"
    if rep_path.exists() and cln.exists():
        return rep_path, cln
    return None


def compute_metrics_for_run(
    rep_path: Path,
    clonotypes_path: Path,
    tp_set: set[tuple[str, str, str]],
    chain: str,
    tn_clone_id_threshold: int = 200000,
):
    spec = chain_spec(chain)

    info = pd.read_csv(rep_path, sep="\t")
    clonotypes = pd.read_csv(clonotypes_path, sep="\t")

    clonotypes_sample = clonotypes[clonotypes.source == "sample"].copy()

    # "clone_123" -> 123
    clonotypes_sample["clone_id"] = clonotypes_sample["clone_id"].apply(
        lambda x: int(str(x).split("_")[1])
    )

    df = info.merge(
        clonotypes_sample[["clone_id", "cluster_id"]],
        on="clone_id",
        how="left",
    )

    # required columns in representations
    need_info = [spec["info_cdr3_col"], spec["info_v_col"], spec["info_j_col"], "clone_id"]
    missing_info = [c for c in need_info if c not in df.columns]
    if missing_info:
        raise KeyError(
            f"Representations TSV missing columns for chain={chain}: {missing_info}. "
            f"Available columns: {list(df.columns)}"
        )

    # prediction = clustered?
    df["is_clustered"] = df["cluster_id"].notna()

    # build matching key for TP
    df["V_trim"] = trim_allele(df[spec["info_v_col"]])
    df["J_trim"] = trim_allele(df[spec["info_j_col"]])
    df["cdr3_key"] = df[spec["info_cdr3_col"]].astype("string")

    df["is_tp"] = list(zip(df["cdr3_key"], df["V_trim"], df["J_trim"]))
    df["is_tp"] = df["is_tp"].isin(tp_set)

    # True labels:
    # TP => True
    # TN by id threshold => False
    # otherwise => NA (ignored)
    df["valid"] = pd.NA
    df.loc[df["clone_id"] > tn_clone_id_threshold, "valid"] = False
    df.loc[df["is_tp"], "valid"] = True

    df_cm = df[df["valid"].notna()].copy()
    if df_cm.empty:
        return None

    y_true = df_cm["valid"].astype(bool)
    y_pred = df_cm["is_clustered"].astype(bool)

    # labels=[True, False] => [[TP, FN],[FP, TN]]
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    TP, FN = int(cm[0, 0]), int(cm[0, 1])
    FP, TN = int(cm[1, 0]), int(cm[1, 1])

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "chain": spec["chain"],
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "precision": precision, "recall": recall, "f1": f1,
        "n_eval": int(len(df_cm)),
        "n_total": int(len(df)),
        "n_clustered": int(df["is_clustered"].sum()),
        "n_tp_labeled": int(df_cm["valid"].sum()),
        "n_tn_labeled": int((~df_cm["valid"].astype(bool)).sum()),
    }


def default_prefix(chain: str, epitope: str) -> str:
    spec = chain_spec(chain)
    return f"{spec['prefix_chain']}_vdjdb_{epitope}"


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compute TP/FP/TN/FN + precision/recall/F1 for all runs under a grid root dir."
    )
    ap.add_argument("--root", required=True, help="Grid root dir with run subfolders.")
    ap.add_argument("--chain", default="trb", choices=["tra", "trb"])
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--epitope", default="YLQPRTFLL")
    ap.add_argument("--out", default="metrics_grid.tsv")
    ap.add_argument(
        "--representations_tsv",
        required=True,
        help="GLOBAL representations TSV (same for all runs).",
    )

    ap.add_argument(
        "--tp_tsv",
        default=None,
        help="Path to vdjdb_public_*_minStudies2.tsv (true positives). "
             "If not provided, uses tcrempnet/data/vdjdb_public_{TRA|TRB}_minStudies2.tsv",
    )
    ap.add_argument(
        "--tn_clone_id_threshold",
        type=int,
        default=200000,
        help="clone_id > threshold => true negative label",
    )

    args = ap.parse_args()

    root = Path(args.root)
    rep_path = Path(args.representations_tsv)

    if args.prefix is None:
        args.prefix = default_prefix(args.chain, args.epitope)

    if args.tp_tsv is None:
        if args.chain == "tra":
            args.tp_tsv = "/home/evlasova/tcrempnet/data/vdjdb_public_TRA_minStudies2.tsv"
        else:
            args.tp_tsv = "/home/evlasova/tcrempnet/data/vdjdb_public_TRB_minStudies2.tsv"

    tp_set = load_true_positive_set(Path(args.tp_tsv), args.epitope, args.chain)

    rows = []

    for run_dir in sorted([p for p in root.rglob("*") if p.is_dir()]):
        found = find_run_files(run_dir, args.prefix, rep_path)
        if not found:
            continue

        rep_path_found, clonotypes_path = found
        parsed = parse_params_from_relpath(run_dir, root)

        m = compute_metrics_for_run(
            rep_path_found,
            clonotypes_path,
            tp_set=tp_set,
            chain=args.chain,
            tn_clone_id_threshold=args.tn_clone_id_threshold,
        )

        if m is None:
            rows.append({
                "run_dir": str(run_dir),
                "status": "no_labels",
                **parsed,
                "chain": args.chain,
            })
            continue

        rows.append({
            "run_dir": str(run_dir),
            "status": "ok",
            **parsed,
            **m,
        })

    out_df = pd.DataFrame(rows)
    if "f1" in out_df.columns:
        out_df = out_df.sort_values(["status", "f1"], ascending=[True, False])

    out_path = Path(args.out)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved: {out_path} ({len(out_df)} runs)")

    if "f1" in out_df.columns:
        print("\nTop-10 by F1:")
        cols = [
            c for c in [
                "chain", "relpath", "kn", "ms", "eps_mode", "eps",
                "TP", "FP", "TN", "FN", "precision", "recall", "f1",
                "n_eval", "n_tp_labeled", "n_tn_labeled"
            ] if c in out_df.columns
        ]
        print(out_df[out_df["status"] == "ok"][cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()