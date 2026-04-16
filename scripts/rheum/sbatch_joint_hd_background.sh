#!/bin/bash

set -euo pipefail

AIRR_DIR="/projects/immunestatus/rheum/airr_format"
OUT_DIR="/projects/immunestatus/rheum/redcea"
LOG_DIR="$OUT_DIR/logs"

BACKGROUND_FILE="$AIRR_DIR/joint_hd.tsv"

CPUS_PER_TASK=16
MEMORY="32gb"
TIME_LIMIT="01:00:00"
PARTITION="medium"
CONSTRAINT="hpc"
NPROC=16

if [[ ! -f "$BACKGROUND_FILE" ]]; then
  echo "Background AIRR file not found: $BACKGROUND_FILE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$LOG_DIR" "$OUT_DIR/shared_embeddings/trb" "$OUT_DIR/shared_embeddings/tra"

trb_job_id="$(
  sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=rheum_joint_hd_trb
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/joint_hd.trb.%j.log
#SBATCH --constraint=${CONSTRAINT}
#SBATCH --partition=${PARTITION}

set -euo pipefail
tcremp-run -i $(printf '%q' "$BACKGROUND_FILE") -c TRB -o $(printf '%q' "$OUT_DIR/shared_embeddings/trb") -np ${NPROC}
EOF
)"

tra_job_id="$(
  sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=rheum_joint_hd_tra
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/joint_hd.tra.%j.log
#SBATCH --constraint=${CONSTRAINT}
#SBATCH --partition=${PARTITION}

set -euo pipefail
tcremp-run -i $(printf '%q' "$BACKGROUND_FILE") -c TRA -o $(printf '%q' "$OUT_DIR/shared_embeddings/tra") -np ${NPROC}
EOF
)"

echo "Submitted TRB background job: $trb_job_id"
echo "Submitted TRA background job: $tra_job_id"
