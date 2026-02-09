#!/bin/sh

#SBATCH --job-name=olga_emerson       # Job name
#SBATCH --cpus-per-task=100     # Run on a single CPU
#SBATCH --mem=128gb                 # Job memory request
#SBATCH --time=24:00:00           # Time limit hrs:min:sec
#SBATCH --output=JobName.%j.log   # Standard output and error log
#SBATCH --constraint=hpc
#SBATCH --partition=long


python pgens_compute.py --file --name /projects/immunestatus/olga/tcremp/TRB_1e7_rep.tsv --processes 100