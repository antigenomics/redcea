#!/bin/bash

set -euo pipefail

AIRR_DIR="/projects/immunestatus/rheum/airr_format"
EMB_DIR="/projects/immunestatus/rheum/tcremp"
OUT_DIR="/projects/immunestatus/rheum/redcea"
RUNS_DIR="$OUT_DIR/runs"

BACKGROUND_FILE="$AIRR_DIR/joint_hd_b27pos.tsv"
BACKGROUND_EMB="$EMB_DIR/joint_hd_b27pos_embeddings.parquet"

samples=(
  as_Abd_PB_F
  as_Abr_PB_F
  as_Ash-110_PB_F_p0
  as_Ash-111_PB_F_p0
  as_Bal_PB_F
  as_Bel_PB_F
  as_Bost_PB_F
  as_Chaad_PB_F
  as_Dv_PB_F
  as_Evst_PB_F
  as_Gar_PB_F
  as_GE_PB_F
  as_Gonch_PB_F
  as_i70_PB_F
  as_Kal_PB_F
  as_Kud_PB_F
  as_Luk_PB_F
  as_Mart_PB_F
  as_Mikh_PB_F
  as_Shep_PB_F
  as_Tsib_PB_F
  as_TwHM1_PB_F
  as_Uv_PB_F
  as_Vas_PB_F
  as_Vats_PB_F
  as_Vol_PB_F
  as_Zakh_PB_F
)

JOB_PREFIX="redcea_rheum"
CPUS_PER_TASK=16
MEMORY="32gb"
TIME_LIMIT="08:00:00"
PARTITION="medium"
CONSTRAINT="hpc"
LOG_DIR="$OUT_DIR/logs"

CHAIN="TRB"
NPROC=16

if [[ ! -f "$BACKGROUND_FILE" ]]; then
  echo "Background AIRR file not found: $BACKGROUND_FILE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$LOG_DIR" "$RUNS_DIR" "$EMB_DIR"

echo "Selected ${#samples[@]} samples"
echo "Background: $BACKGROUND_FILE"
echo "Background embedding: $BACKGROUND_EMB"
echo "Output dir: $OUT_DIR"

if [[ ! -f "$BACKGROUND_EMB" ]]; then
  echo "Background embedding not found: $BACKGROUND_EMB" >&2
  echo "Run scripts/rheum/sbatch_joint_hd_background.sh first." >&2
  exit 1
fi

for sample in "${samples[@]}"; do
  sample_file="$AIRR_DIR/${sample}.tsv"
  sample_emb="$EMB_DIR/${sample}_embeddings.parquet"
  run_dir="$RUNS_DIR/$sample"

  if [[ ! -f "$sample_file" ]]; then
    echo "Skipping $sample: sample AIRR file not found: $sample_file" >&2
    continue
  fi

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_PREFIX}_${sample}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/${sample}.%j.log
#SBATCH --constraint=${CONSTRAINT}
#SBATCH --partition=${PARTITION}

set -euo pipefail
mkdir -p $(printf '%q' "$run_dir") $(printf '%q' "$OUT_DIR")
mkdir -p $(printf '%q' "$EMB_DIR")
redcea -is $(printf '%q' "$sample_file") -ib $(printf '%q' "$BACKGROUND_FILE") -o $(printf '%q' "$run_dir") -e $(printf '%q' "$sample") -c $(printf '%q' "$CHAIN") -np $(printf '%q' "$NPROC") -se $(printf '%q' "$sample_emb") -be $(printf '%q' "$BACKGROUND_EMB")
find $(printf '%q' "$run_dir") -maxdepth 1 -type f -name $(printf '%q' "${sample}_*") -exec cp -f {} $(printf '%q' "$OUT_DIR/") \;
EOF
done
