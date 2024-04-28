#!/bin/bash

#SBATCH --job-name=v200h3
#SBATCH --mail-user=srayan@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
 
#SBATCH --account=stats_dept1
#SBATCH --partition=standard

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

## 5GB/cpu is the basic share
#SBATCH --mem=96000m

## wall time hours:minutes:seconds
#SBATCH --time=24:00:00

###   Load software modules
### nvidia-cuda-mps-control -d
### module load singularity cuda/11.3.0

eval "$(conda shell.bash hook)"
conda activate 608proj

####  Commands your job should run follow this line

python3 test_vi.py --seed 44 --horizon 200 --episode 5