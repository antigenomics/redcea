#!/bin/bash
set -euo pipefail

BASE="/projects/immunestatus/vdjdb_valid"
SCRIPT="compute_tcremp_grid_metrics_paired.py"   # <-- новый python-скрипт (paired TRA+TRB, без matching id)

# ---- epitope-specific params ----
EPITOPE="GLCTLVAML"  # <-- эпитоп для анализа (можешь менять на другой, если нужно)

# prefixes + global representations per chain
PREFIX_TRA="tra_vdjdb_${EPITOPE}"
PREFIX_TRB="trb_vdjdb_${EPITOPE}"

REP_TRA="${BASE}/tcremp/${PREFIX_TRA}_tcremp_representations.tsv"
REP_TRB="${BASE}/tcremp/${PREFIX_TRB}_tcremp_representations.tsv"

# optional
VALIDATOR_CSV="/home/evlasova/tcrempnet/data/01_05_2025_TCRvdb.csv"
PADJ_THR="1e-5"

OUTDIR="${BASE}/metrics_${EPITOPE}_paired"
mkdir -p "${OUTDIR}"

# NOTE: тут предполагается, что у тебя симметричные root'ы для TRA и TRB
# типа tcrempnet_${EPITOPE}_tra_leiden и tcrempnet_${EPITOPE}_trb_leiden
declare -A ALGOS
ALGOS["leiden"]="leiden"
ALGOS["hierarchical_leiden"]="hierarchical_leiden"
ALGOS["leiden_dbscan"]="leiden_dbscan"
ALGOS["vdbscan"]="vdbscan"

for ALGO in "${!ALGOS[@]}"; do
  ROOT_TRA="${BASE}/tcrempnet_${EPITOPE}_tra_${ALGOS[$ALGO]}"
  ROOT_TRB="${BASE}/tcrempnet_${EPITOPE}_trb_${ALGOS[$ALGO]}"
  OUT="${OUTDIR}/metrics_${ALGO}.tsv"

  echo "======================================"
  echo "Running paired metrics for ALGO = ${ALGO}"
  echo "EPITOPE     = ${EPITOPE}"
  echo "ROOT_TRA    = ${ROOT_TRA}"
  echo "ROOT_TRB    = ${ROOT_TRB}"
  echo "PREFIX_TRA  = ${PREFIX_TRA}"
  echo "PREFIX_TRB  = ${PREFIX_TRB}"
  echo "REPR_TRA    = ${REP_TRA}"
  echo "REPR_TRB    = ${REP_TRB}"
  echo "OUT         = ${OUT}"
  echo "======================================"

  if [[ ! -d "${ROOT_TRA}" ]]; then
    echo "⚠️  Skip: ROOT_TRA does not exist: ${ROOT_TRA}"
    continue
  fi
  if [[ ! -d "${ROOT_TRB}" ]]; then
    echo "⚠️  Skip: ROOT_TRB does not exist: ${ROOT_TRB}"
    continue
  fi

  python "${SCRIPT}" \
    --root_tra "${ROOT_TRA}" \
    --root_trb "${ROOT_TRB}" \
    --prefix_tra "${PREFIX_TRA}" \
    --prefix_trb "${PREFIX_TRB}" \
    --representations_tra "${REP_TRA}" \
    --representations_trb "${REP_TRB}" \
    --validator_csv "${VALIDATOR_CSV}" \
    --epitope "${EPITOPE}" \
    --padj_thr "${PADJ_THR}" \
    --out "${OUT}"
done

echo "✅ All paired metrics computed in: ${OUTDIR}"
