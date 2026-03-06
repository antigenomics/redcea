#!/bin/bash
set -euo pipefail

BASE="/projects/immunestatus/vdjdb_olga"
SCRIPT="compute_tcremp_grid_metrics.py"

# optional
VALIDATOR_CSV="/home/evlasova/tcrempnet/data/01_05_2025_TCRvdb.csv"
PADJ_THR="1e-5"

EPITOPES=(
  "GILGFVFTL"
  "GLCTLVAML"
  "NLVPMVATV"
  "YLQPRTFLL"
)

CHAINS=("tra" "trb")
TNS=("10" "25")

for EPITOPE in "${EPITOPES[@]}"; do
  for CHAIN in "${CHAINS[@]}"; do
    for TN in "${TNS[@]}"; do

      PREFIX="${CHAIN}_vdjdb_${EPITOPE}_tn${TN}"
      REPRESENTATIONS="${BASE}/tcremp/${PREFIX}_tcremp_representations.tsv"

      ROOT="${BASE}/tcrempnet_${EPITOPE}_${CHAIN}_tn${TN}/leiden"
      OUTDIR="${BASE}/metrics/${EPITOPE}/${CHAIN}/tn${TN}"
      OUT="${OUTDIR}/metrics_leiden.tsv"
      mkdir -p "${OUTDIR}"

      echo "======================================"
      echo "ALGO    = leiden"
      echo "EPITOPE = ${EPITOPE}"
      echo "CHAIN   = ${CHAIN}"
      echo "TN      = ${TN}"
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
    --epitope "${EPITOPE}" \
    --out "${OUT}" \
    --chain "${CHAIN}" \
    --tp_tsv "/home/evlasova/tcrempnet/data/vdjdb_public_${CHAIN^^}_minStudies2.tsv"

    done
  done
done

echo "✅ All metrics computed under: ${BASE}/metrics/"