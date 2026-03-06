#!/bin/bash

# Путь к общим файлам
AIRR_DIR="/projects/immunestatus/vdjdb_valid/airr_format"
EMB_DIR="/projects/immunestatus/vdjdb_valid/tcremp"

# Список сэмплов
samples=(
trb_background trb_vdjdb_uni
)

# Запуск цикла сабмита задач
for sample in "${samples[@]}"; do
  SAMPLE_FILE="$AIRR_DIR/${sample}.tsv"

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tcremp_${sample}
#SBATCH --cpus-per-task=100
#SBATCH --mem=256gb
#SBATCH --time=01:00:00
#SBATCH --output=logs_tcremp/${sample}.%j.log
#SBATCH --constraint=hpc
#SBATCH --partition=medium

tcremp-run -i "$SAMPLE_FILE" -c TRB -o "$EMB_DIR" -np 100 --cluster False
EOF

done
