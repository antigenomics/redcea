#!/bin/bash

# Путь к общим файлам
AIRR_DIR="/projects/immunestatus/olga/airr_format"
EMB_DIR="/projects/immunestatus/olga/tcremp1"

# Список сэмплов
samples=(
TRB_1e7
)

# Запуск цикла сабмита задач
for sample in "${samples[@]}"; do
  SAMPLE_FILE="$AIRR_DIR/${sample}.tsv.gz"

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tcremp_${sample}
#SBATCH --cpus-per-task=100
#SBATCH --mem=512gb
#SBATCH --time=24:00:00
#SBATCH --output=logs_tcremp/${sample}.%j.log
#SBATCH --constraint=hpc
#SBATCH --partition=long

tcremp-run -i "$SAMPLE_FILE" -c TRB -o "$EMB_DIR" -np 100
EOF

done
