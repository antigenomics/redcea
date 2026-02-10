#!/bin/bash
set -euo pipefail

BASE="/projects/immunestatus/vdjdb"
SCRIPT="compute_tcremp_grid_metrics.py"

# ---- epitope-specific params ----
EPITOPE="GLCTLVAML"
CHAIN="trb"
PREFIX="${CHAIN}_vdjdb_${EPITOPE}"
REPRESENTATIONS="${BASE}/tcremp/${PREFIX}_tcremp_representations.tsv"

# optional (если нужно)
VALIDATOR_CSV="/home/evlasova/tcrempnet/data/01_05_2025_TCRvdb.csv"
PADJ_THR="1e-5"

OUTDIR="${BASE}/metrics_${EPITOPE}"
mkdir -p "${OUTDIR}"

declare -A ALGOS
ALGOS["leiden"]="tcrempnet_${EPITOPE}_${CHAIN}_leiden"
ALGOS["hierarchical_leiden"]="tcrempnet_${EPITOPE}_${CHAIN}_hierarchical_leiden"
ALGOS["leiden_dbscan"]="tcrempnet_${EPITOPE}_${CHAIN}_leiden_dbscan"
ALGOS["vdbscan"]="tcrempnet_${EPITOPE}_${CHAIN}_vdbscan"

for ALGO in "${!ALGOS[@]}"; do
  ROOT="${BASE}/${ALGOS[$ALGO]}"
  OUT="${OUTDIR}/metrics_${ALGO}.tsv"

  echo "======================================"
  echo "Running metrics for ALGO = ${ALGO}"
  echo "EPITOPE = ${EPITOPE}"
  echo "PREFIX  = ${PREFIX}"
  echo "CHAIN   = ${CHAIN}"
  echo "ROOT    = ${ROOT}"
  echo "REPR    = ${REPRESENTATIONS}"
  echo "OUT     = ${OUT}"
  echo "======================================"

  if [[ ! -d "${ROOT}" ]]; then
    echo "⚠️  Skip: ROOT does not exist: ${ROOT}"
    continue
  fi

  python "${SCRIPT}" \
    --root "${ROOT}" \
    --prefix "${PREFIX}" \
    --representations_tsv "${REPRESENTATIONS}" \
    --validator_csv "${VALIDATOR_CSV}" \
    --epitope "${EPITOPE}" \
    --padj_thr "${PADJ_THR}" \
    --out "${OUT}" \
    --chain "${CHAIN}"
done

echo "✅ All metrics computed in: ${OUTDIR}"
