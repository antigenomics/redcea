#!/bin/bash
set -euo pipefail

BASE="/projects/immunestatus/vdjdb_valid"
SCRIPT="compute_tcremp_grid_metrics_paired.py"

TP_TRA="/home/evlasova/tcrempnet/data/vdjdb_public_TRA_minStudies2.tsv"
TP_TRB="/home/evlasova/tcrempnet/data/vdjdb_public_TRB_minStudies2.tsv"

EPITOPES=(
GILGFVFTL
GLCTLVAML
NLVPMVATV
YLQPRTFLL
YVLDHLIVV
)

TN_VERSIONS=(
tn10
tn25
)

echo "======================================"
echo "TCREmp grid metrics"
echo "======================================"
echo "Base directory: $BASE"
echo ""

for EP in "${EPITOPES[@]}"; do
for TN in "${TN_VERSIONS[@]}"; do

  echo ""
  echo "======================================"
  echo "EPITOPE: $EP   |   VERSION: $TN"
  echo "======================================"

  PREFIX_TRA="tra_vdjdb_${EP}_${TN}"
  PREFIX_TRB="trb_vdjdb_${EP}_${TN}"

  ROOT_TRA="${BASE}/tcrempnet_${EP}_tra_${TN}"
  ROOT_TRB="${BASE}/tcrempnet_${EP}_trb_${TN}"

  REP_TRA="${BASE}/tcremp/tra_vdjdb_${EP}_${TN}_tcremp_representations.tsv"
  REP_TRB="${BASE}/tcremp/trb_vdjdb_${EP}_${TN}_tcremp_representations.tsv"

  OUT="${BASE}/metrics_${EP}_${TN}.tsv"

  echo "TRA root: $ROOT_TRA"
  echo "TRB root: $ROOT_TRB"
  echo "Output:   $OUT"
  echo ""

  python "$SCRIPT" \
    --root_tra "$ROOT_TRA" \
    --root_trb "$ROOT_TRB" \
    --prefix_tra "$PREFIX_TRA" \
    --prefix_trb "$PREFIX_TRB" \
    --rep_tra "$REP_TRA" \
    --rep_trb "$REP_TRB" \
    --tp_tra "$TP_TRA" \
    --tp_trb "$TP_TRB" \
    --out "$OUT"

  echo ""
  echo "Finished $EP $TN"
  echo "Results saved to $OUT"
  echo ""

done
done

echo "======================================"
echo "All runs finished"
echo "======================================"