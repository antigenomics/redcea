#!/bin/bash

#SBATCH --job-name=olga_trb       # Job name
#SBATCH --cpus-per-task=100     # Run on a single CPU
#SBATCH --mem=128gb                 # Job memory request
#SBATCH --time=24:00:00           # Time limit hrs:min:sec
#SBATCH --output=JobName.%j.log   # Standard output and error log
#SBATCH --constraint=hpc
#SBATCH --partition=long

set -eu

# === Параметры ===
INPUT_GZ="/projects/immunestatus/vdjrearm/sample/TRB_1e7.tsv.gz"
INPUT="${INPUT_GZ%.gz}"
PREFIX="batch_trb"
N=100
OUTDIR="/projects/immunestatus/vdjrearm/sample/pgens_trb_batches"
LOGDIR="/projects/immunestatus/vdjrearm/sample/logs_trb"
MERGED_OUT="/projects/immunestatus/vdjrearm/sample/TRB_1e7_pgens.tsv"

mkdir -p "$OUTDIR" "$LOGDIR"

echo "=== [TRB] 1. Разархивация $INPUT_GZ ===" | tee "$LOGDIR/main.log"
gunzip -c "$INPUT_GZ" > "$INPUT"

echo "=== [TRB] 2. Разделение на $N батчей ===" | tee -a "$LOGDIR/main.log"
split -d -n l/$N "$INPUT" "$OUTDIR/$PREFIX"_
rm "$INPUT"  # удаляем исходный распакованный файл

echo "=== [TRB] 3. Запуск OLGA в $N потоков ===" | tee -a "$LOGDIR/main.log"
for f in "$OUTDIR"/${PREFIX}_*; do
    (
        out="${f}_pgens.tsv"
        log_file="$LOGDIR/$(basename "$f").log"
        echo "[TRB] Processing $f -> $out (log: $log_file)"
        olga-compute_pgen --humanTRB -i "$f" -o "$out" >"$log_file" 2>&1
        rm "$f"  # удаляем батч после завершения
    ) &
    while (( $(jobs | wc -l) >= N )); do sleep 1; done
done
wait

echo "=== [TRB] 4. Объединение результатов ===" | tee -a "$LOGDIR/main.log"
head -n 1 "${OUTDIR}/${PREFIX}_00_pgens.tsv" > "$MERGED_OUT"
tail -n +2 -q "${OUTDIR}"/${PREFIX}_*_pgens.tsv >> "$MERGED_OUT"

echo "=== [TRB] 5. Очистка временных файлов ===" | tee -a "$LOGDIR/main.log"
rm -rf "$OUTDIR"

echo "✅ [TRB] Готово: $MERGED_OUT" | tee -a "$LOGDIR/main.log"
