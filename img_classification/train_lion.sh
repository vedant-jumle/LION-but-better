#!/bin/bash
#SBATCH --job-name=lion_train
#SBATCH --partition=gpu-v100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/vvjumle/logs/lion_train_%j.out
#SBATCH --error=/scratch/vvjumle/logs/lion_train_%j.err

set -e

REPO=/scratch/vvjumle/LION-but-better/img_classification
DATA=/scratch/vvjumle/imagenet-folder
OUTPUT=/scratch/vvjumle/checkpoints/lion_tiny_decay

mkdir -p /scratch/vvjumle/logs
mkdir -p $OUTPUT

module load 2025
module load miniconda3
conda activate LION
module load cuda/12.9

cd $REPO

echo "=== GPU info ==="
nvidia-smi

echo "=== Starting LION-D Tiny training ==="
torchrun --nproc_per_node=1 --master_port=29500 main_lion.py \
    --model lion_tiny_patch16_224 \
    --data-path $DATA \
    --batch-size 256 \
    --mask_type Decay \
    --format Attention \
    --epochs 30 \
    --output_dir $OUTPUT \
    --num_workers 8

echo "=== Training complete. Checkpoints at $OUTPUT ==="
