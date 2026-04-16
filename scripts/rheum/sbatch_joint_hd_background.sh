#!/bin/bash

set -euo pipefail

AIRR_DIR="/projects/immunestatus/rheum/airr_format"
OUT_DIR="/projects/immunestatus/rheum/tcremp"
LOG_DIR="$OUT_DIR/logs"

BACKGROUND_FILE="$AIRR_DIR/joint_hd_b27pos.tsv"

CPUS_PER_TASK=32
MEMORY="64gb"
TIME_LIMIT="08:00:00"
PARTITION="medium"
CONSTRAINT="hpc"
NPROC=32

if [[ ! -f "$BACKGROUND_FILE" ]]; then
  echo "Background AIRR file not found: $BACKGROUND_FILE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"

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
tcremp-run -i $(printf '%q' "$BACKGROUND_FILE") -c TRB -o $(printf '%q' "$OUT_DIR") -np ${NPROC}
EOF
)"

echo "Submitted TRB background job: $trb_job_id"
