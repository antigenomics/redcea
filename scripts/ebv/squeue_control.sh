#!/bin/sh

#SBATCH --job-name=tcremp_emerson_control_ebv        # Job name
#SBATCH --cpus-per-task=48        # Run on a single CPU
#SBATCH --mem=256gb                 # Job memory request
#SBATCH --time=48:00:00           # Time limit hrs:min:sec
#SBATCH --output=control.%j.log   # Standard output and error log
#SBATCH --constraint=hpc
#SBATCH --partition=long

tcremp-run -i /projects/immunestatus/emerson/airr_format/control_ebv.tsv -c TRB -o /projects/immunestatus/emerson/tcremp -np 48