#!/bin/bash

# Путь к общим файлам
AIRR_DIR="/projects/immunestatus/vdjdb/airr_format"
EMB_DIR="/projects/immunestatus/vdjdb/tcremp"

# Список сэмплов
samples=(
tra_background tra_vdjdb
)

# Запуск цикла сабмита задач
for sample in "${samples[@]}"; do
  SAMPLE_FILE="$AIRR_DIR/${sample}.tsv"

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tcremp_${sample}
#SBATCH --cpus-per-task=100
#SBATCH --mem=256gb
#SBATCH --time=08:00:00
#SBATCH --output=logs_tcremp/${sample}.%j.log
#SBATCH --constraint=hpc
#SBATCH --partition=medium

tcremp-run -i "$SAMPLE_FILE" -c TRA -o "$EMB_DIR" -np 100 --cluster False
EOF

done
