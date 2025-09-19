#!/bin/bash
#
#SBATCH --job-name=run_bumpy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1          
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --partition=general
#SBATCH --output=bumpy_%j.out      

echo "Job started on $(hostname) at $(date)"

module load anaconda
conda activate bumpy_env    
       
python bumpy.py -f flat.pdb -s sphere -z 10 -g r_sphere:100 -o sphere.pdb

echo "Job finished at $(date)"