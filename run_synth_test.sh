#!/bin/bash
#SBATCH --job-name=synth-pile-test
#SBATCH --account=project_2017850   
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module purge
module load pytorch   # check available with: module avail pytorch

export HF_login_synth="your_token_here"   # or set via MyCSC secrets

mkdir -p logs

python synthetic_test.py
