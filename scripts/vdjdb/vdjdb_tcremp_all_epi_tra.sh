#!/bin/bash

# Путь к общим файлам
AIRR_DIR="/projects/immunestatus/vdjdb_olga/airr_format"
EMB_DIR="/projects/immunestatus/vdjdb_olga/tcremp"

mkdir -p logs_tcremp

files=("$AIRR_DIR"/tra*.tsv)

# Проходим по всем tsv файлам
for SAMPLE_FILE in "${files[@]}"; do

  sample=$(basename "$SAMPLE_FILE" .tsv)

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tcremp_${sample}
#SBATCH --cpus-per-task=32
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --output=logs_tcremp/${sample}.%j.log
#SBATCH --constraint=hpc
#SBATCH --partition=short

tcremp-run -i "$SAMPLE_FILE" -c TRA -o "$EMB_DIR" -np 32 -kn 12
EOF

done