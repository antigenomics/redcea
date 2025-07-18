#!/bin/sh

#SBATCH --job-name=tcrempnet_S1       # Job name
#SBATCH --cpus-per-task=16        # Run on a single CPU
#SBATCH --mem=256gb                 # Job memory request
#SBATCH --time=02:00:00           # Time limit hrs:min:sec
#SBATCH --output=JobName.%j.log   # Standard output and error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elizaveta.k.vlasova@gmail.com
#SBATCH --constraint=hpc
#SBATCH --partition=short


python tcrEmpNet.py \
  --sample /projects/immunestatus/pogorelyy/airr_format/S1_15_F1.txt \
  --background /projects/immunestatus/pogorelyy/airr_format/S1_0_F1_with_1.txt \
  --output /projects/immunestatus/pogorelyy/tcrempnet/S1_F1 \
  --chain TRB \
  --prefix yfv_S1_F1_with_1 \
  -np 16