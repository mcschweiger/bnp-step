#!/bin/bash
#SBATCH --job-name=bifish_job
#SBATCH --mail-type=ALL
# PEDRO: change this to your email address
#SBATCH --mail-user=mcschwei@asu.edu
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
##SBATCH -q grp_spresse
#SBATCH -p public
#SBATCH --time=2-00:00:00
#SBATCH --array=1-12
#SBATCH --ntasks=1
#SBATCH --gres=gpus-per-node:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# === Use manually installed Julia ===
# PEDRO: change this path to point to your julia .bin
export JULIA_BIN=/home/mcschwei/programs/julia-current/bin/julia

echo "JULIA_BIN = $JULIA_BIN"
$JULIA_BIN --version || { echo "Julia not found or failed"; exit 1; }

echo "SLURM job started on $(hostname) at $(date)"
echo "Working directory: $(pwd)"
ls -lh .

# === Change to scripts directory ===
# PEDRO: change this path to your BiFISH/scripts folder
cd /scratch/mcschwei/codes/BNPStep/scripts || { echo "scripts directory not found"; exit 1; }
test -f run_zhuang.jl || { echo "run_zhuang.jl not found"; exit 1; }

# Now just run the script (no need to instantiate or precompile)
$JULIA_BIN --threads auto BNPStep_experiment.jl $SLURM_ARRAY_TASK_ID

