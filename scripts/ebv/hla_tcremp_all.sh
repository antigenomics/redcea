#!/bin/bash

# Путь к общим файлам
AIRR_DIR="/projects/immunestatus/emerson_hla/airr_format"
EMB_DIR="/projects/immunestatus/emerson_hla/tcremp"

# Список сэмплов
samples=(
HIP00640  HIP03004  HIP05578  HIP09159  HIP12143  HIP13722  HIP13871  HIP14077  HIP14911
HIP00825  HIP04576  HIP08230  HIP10376  HIP12538  HIP13794  HIP14018  HIP14209
HIP00826  HIP05409  HIP08345  HIP10564  HIP13178  HIP13859  HIP14074  HIP14230
)

# Запуск цикла сабмита задач
for sample in "${samples[@]}"; do
  SAMPLE_FILE="$AIRR_DIR/${sample}.tsv"

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tcremp_${sample}
#SBATCH --cpus-per-task=100
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --output=logs_tcremp/${sample}.%j.log
#SBATCH --constraint=hpc
#SBATCH --partition=short

tcremp-run -i "$SAMPLE_FILE" -c TRB -o "$EMB_DIR" -np 100
EOF

done


