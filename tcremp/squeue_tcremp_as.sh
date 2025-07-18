#!/bin/sh

#SBATCH --job-name=tcremp_as_Mikh        # Job name
#SBATCH --cpus-per-task=48        # Run on a single CPU
#SBATCH --mem=4gb                 # Job memory request
#SBATCH --time=02:00:00           # Time limit hrs:min:sec
#SBATCH --output=tcremp_as_Mikh.%j.log   # Standard output and error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elizaveta.k.vlasova@gmail.com
#SBATCH --constraint=hpc
#SBATCH --partition=short


python tcremp_run.py \
  --input /projects/immunestatus/rheum/airr_format/as_Mikh_SFCD8.tsv \
  --output /projects/immunestatus/test \
  --chain TRB \
  -np 48