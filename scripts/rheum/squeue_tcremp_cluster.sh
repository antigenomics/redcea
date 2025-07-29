#!/bin/sh

#SBATCH --job-name=tcremp_as_Mikh        # Job name
#SBATCH --cpus-per-task=16        # Run on a single CPU
#SBATCH --mem=256gb                 # Job memory request
#SBATCH --time=02:00:00           # Time limit hrs:min:sec
#SBATCH --output=tcremp_as_Mikh.%j.log   # Standard output and error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elizaveta.k.vlasova@gmail.com
#SBATCH --constraint=hpc
#SBATCH --partition=short


python tcremp_cluster.py \
  --input /projects/immunestatus/rheum/tcremp/hd_TwHM2_PB_F_embeddings.parquet \
  --output /projects/immunestatus/test/test.csv \
  --kth_neighbor 4