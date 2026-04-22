#!/bin/bash
#SBATCH --job-name=synth-${DATASET_SAFE}-${STYLE}
#SBATCH --account=project_2017850
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export HF_TOKEN="your_token_here"

module purge
module load pytorch  # check with: module avail pytorch

cd ~/synthpile

python generate.py \
    --dataset "$DATASET" \
    --style "$STYLE" \
    --texts_per_dataset 1000 \
    --output_dir outputs
