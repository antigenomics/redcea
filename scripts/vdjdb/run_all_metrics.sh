#!/bin/bash
set -euo pipefail

BASE="/projects/immunestatus/vdjdb"
SCRIPT="compute_tcremp_grid_metrics.py"

# ---- epitope-specific params ----
EPITOPE="GLCTLVAML"
PREFIX="trb_vdjdb_${EPITOPE}"
REPRESENTATIONS="${BASE}/tcremp/${PREFIX}_tcremp_representations.tsv"

# optional (если нужно)
VALIDATOR_CSV="/home/evlasova/tcrempnet/data/01_05_2025_TCRvdb.csv"
PADJ_THR="1e-5"

OUTDIR="${BASE}/metrics_${EPITOPE}"
mkdir -p "${OUTDIR}"

declare -A ALGOS
ALGOS["leiden"]="tcrempnet_${EPITOPE}_trb_leiden"
ALGOS["hierarchical_leiden"]="tcrempnet_${EPITOPE}_trb_hierarchical_leiden"
ALGOS["leiden_dbscan"]="tcrempnet_${EPITOPE}_trb_leiden_dbscan"
ALGOS["vdbscan"]="tcrempnet_${EPITOPE}_trb_vdbscan"

for ALGO in "${!ALGOS[@]}"; do
  ROOT="${BASE}/${ALGOS[$ALGO]}"
  OUT="${OUTDIR}/metrics_${ALGO}.tsv"

  echo "======================================"
  echo "Running metrics for ALGO = ${ALGO}"
  echo "EPITOPE = ${EPITOPE}"
  echo "PREFIX  = ${PREFIX}"
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
    --out "${OUT}"
done

echo "✅ All metrics computed in: ${OUTDIR}"
