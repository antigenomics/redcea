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
        "eps": None,          # if path contains eps_0.35 etc.
        "relpath": str(rel),
        "tags": [],           # store unparsed tokens for debugging
    }

    def set_if_none(key, val):
        if params.get(key) is None:
            params[key] = val

    for part in parts:
        # kn_12
        m = re.fullmatch(r"kn_(\d+)", part)
        if m:
            set_if_none("kn", int(m.group(1)))
            continue

        # split by "__" to parse compound segments like "ms_5__eps_background"
        tokens = part.split("__") if "__" in part else [part]

        for tok in tokens:
            # ms_5
            m = re.fullmatch(r"ms_(\d+)", tok)
            if m:
                set_if_none("ms", int(m.group(1)))
                continue

            # eps_background / eps_sample / eps_global ...
            m = re.fullmatch(r"eps_([A-Za-z]+)", tok)
            if m:
                set_if_none("eps_mode", m.group(1))
                continue

            # eps_0.35
            m = re.fullmatch(r"eps_([0-9]*\.?[0-9]+)", tok)
            if m:
                set_if_none("eps", float(m.group(1)))
                continue

            # generic key_number (e.g. q_0.05, deg_10, mincs_5)
            m = re.fullmatch(r"([A-Za-z]+)_([0-9]*\.?[0-9]+)", tok)
            if m:
                k, v = m.group(1), m.group(2)
                val = float(v) if "." in v else int(v)
                if k not in params or params[k] is None:
                    params[k] = val
                else:
                    params["tags"].append(tok)
                continue

            # generic key_string (e.g. mode_background, metric_cosine)
            m = re.fullmatch(r"([A-Za-z]+)_([A-Za-z][A-Za-z0-9\-]*)", tok)
            if m:
                k, v = m.group(1), m.group(2)
                if k not in params or params[k] is None:
                    params[k] = v
                else:
                    params["tags"].append(tok)
                continue

            # fallback
            params["tags"].append(tok)

    params["tags"] = "__".join([t for t in params["tags"] if t])
    return params


# -------------------------
# helpers
# -------------------------
def trim_allele(series: pd.Series) -> pd.Series:
    # remove allele suffix like TRBV12-3*01 -> TRBV12-3
    # keep NaN as NaN
    s = series.copy()
    s = s.where(s.notna(), other=pd.NA)
    return s.astype("string").str.replace(r"\*.*$", "", regex=True)


def chain_spec(chain: str) -> dict:
    """
    Central mapping for alpha/beta column names.
    - info TSV: produced by your pipeline (cdr3aa_*, v_*, j_*)
    - validator CSV: TCRvdb columns (cdr3_*_aa, TR?V, TR?J)
    """
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
            "val_cdr3_col": "cdr3_beta_aa",
            "val_v_col": "TRBV",
            "val_j_col": "TRBJ",
        }

    # TRA
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


def load_validator(validator_csv: Path, epitope: str, padj_thr: float, chain: str) -> pd.DataFrame:
    spec = chain_spec(chain)

    mm = (
        pd.read_csv(validator_csv)
        .drop(columns=["Unnamed: 0"], errors="ignore")
        .dropna(subset=["padj"])
    )
    mm = mm[mm["epitope_aa"] == epitope].copy()
    mm["valid"] = mm["padj"] < padj_thr

    # Ensure needed columns exist
    need = [spec["val_cdr3_col"], spec["val_v_col"], spec["val_j_col"], "valid"]
    missing = [c for c in need if c not in mm.columns]
    if missing:
        raise KeyError(
            f"Validator CSV missing columns for chain={chain}: {missing}. "
            f"Available columns: {list(mm.columns)}"
        )

    out = (
        mm[need]
        .groupby([spec["val_cdr3_col"], spec["val_v_col"], spec["val_j_col"]], as_index=False)
        .any()
    )
    return out


def find_run_files(run_dir: Path, prefix: str, rep_path: Path):
    cln = run_dir / f"{prefix}_enriched_clonotypes_tcremp.tsv"
    if rep_path.exists() and cln.exists():
        return rep_path, cln
    return None


def compute_metrics_for_run(rep_path: Path, clonotypes_path: Path, valid_df: pd.DataFrame, chain: str):
    spec = chain_spec(chain)

    info = pd.read_csv(rep_path, sep="\t")
    clonotypes = pd.read_csv(clonotypes_path, sep="\t")

    # keep sample clonotypes only
    clonotypes_sample = clonotypes[clonotypes.source == "sample"].copy()

    # clone_id: "clone_123" -> 123 (as in your render_plots.py)
    clonotypes_sample["clone_id"] = clonotypes_sample["clone_id"].apply(
        lambda x: int(str(x).split("_")[1])
    )

    # merge cluster_id into info
    df = info.merge(
        clonotypes_sample[["clone_id", "cluster_id"]],
        on="clone_id",
        how="left",
    )

    # Ensure needed info columns exist
    need_info = [spec["info_cdr3_col"], spec["info_v_col"], spec["info_j_col"]]
    missing_info = [c for c in need_info if c not in df.columns]
    if missing_info:
        raise KeyError(
            f"Representations TSV missing columns for chain={chain}: {missing_info}. "
            f"Available columns: {list(df.columns)}"
        )

    # prepare validator join keys (normalized alleles)
    df[spec["val_v_col"]] = trim_allele(df[spec["info_v_col"]])
    df[spec["val_j_col"]] = trim_allele(df[spec["info_j_col"]])
    df[spec["val_cdr3_col"]] = df[spec["info_cdr3_col"]]

    df = df.merge(
        valid_df,
        on=[spec["val_cdr3_col"], spec["val_v_col"], spec["val_j_col"]],
        how="left",
    )

    df["is_clustered"] = df["cluster_id"].notna()

    # evaluate only where validation label exists
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
    }


def default_prefix(chain: str, epitope: str) -> str:
    spec = chain_spec(chain)
    return f"{spec['prefix_chain']}_vdjdb_{epitope}"


def default_representations_tsv(chain: str, epitope: str) -> str:
    # keeps your previous convention: /projects/.../tcremp/{prefix}_tcremp_representations.tsv
    pref = default_prefix(chain, epitope)
    return f"/projects/immunestatus/vdjdb/tcremp/{pref}_tcremp_representations.tsv"


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compute TP/FP/TN/FN + precision/recall/F1 for all runs under a grid root dir."
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Grid root dir with run subfolders.",
    )
    ap.add_argument(
        "--chain",
        default="trb",
        choices=["tra", "trb"],
        help="TCR chain to evaluate (tra or trb).",
    )
    ap.add_argument(
        "--prefix",
        default=None,
        help="File prefix used in run outputs (same as in render_plots.py). "
             "If not provided, inferred as '{tra|trb}_vdjdb_{epitope}'.",
    )
    ap.add_argument(
        "--validator_csv",
        default="/home/evlasova/tcrempnet/data/01_05_2025_TCRvdb.csv",
        help="Path to TCRvdb validation CSV.",
    )
    ap.add_argument("--epitope", default="YLQPRTFLL")
    ap.add_argument("--padj_thr", type=float, default=1e-5)
    ap.add_argument(
        "--out",
        default="metrics_grid.tsv",
        help="Output TSV path.",
    )
    ap.add_argument(
        "--representations_tsv",
        default=None,
        help="GLOBAL representations TSV (same for all runs). "
             "If not provided, inferred from chain+epitope under /projects/immunestatus/vdjdb/tcremp/.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"--root does not exist: {root}")

    # Fill defaults that depend on chain/epitope
    if args.prefix is None:
        args.prefix = default_prefix(args.chain, args.epitope)
    if args.representations_tsv is None:
        args.representations_tsv = default_representations_tsv(args.chain, args.epitope)

    valid_df = load_validator(Path(args.validator_csv), args.epitope, args.padj_thr, args.chain)

    rows = []
    rep_path = Path(args.representations_tsv)

    # walk all dirs and pick those that contain both required files
    for run_dir in sorted([p for p in root.rglob("*") if p.is_dir()]):
        print(f"Processing: {run_dir}")
        found = find_run_files(run_dir, args.prefix, rep_path)
        if not found:
            print(f"  Skipping (missing files): {run_dir}")
            continue

        rep_path_found, clonotypes_path = found
        parsed = parse_params_from_relpath(run_dir, root)

        m = compute_metrics_for_run(rep_path_found, clonotypes_path, valid_df, args.chain)
        if m is None:
            rows.append({
                "run_dir": str(run_dir),
                "status": "no_valid_labels",
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

    # sort for convenience
    if "f1" in out_df.columns:
        out_df = out_df.sort_values(["status", "f1"], ascending=[True, False])

    out_path = Path(args.out)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved: {out_path} ({len(out_df)} runs)")

    # quick top-10 preview
    if "f1" in out_df.columns:
        print("\nTop-10 by F1:")
        cols = [
            c for c in [
                "chain", "relpath", "kn", "ms", "eps_mode", "eps",
                "TP", "FP", "TN", "FN", "precision", "recall", "f1"
            ] if c in out_df.columns
        ]
        print(out_df[out_df["status"] == "ok"][cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
