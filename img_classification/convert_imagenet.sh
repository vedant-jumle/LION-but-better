#!/bin/bash
#SBATCH --job-name=imagenet_convert
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/vvjumle/logs/imagenet_convert_%j.out
#SBATCH --error=/scratch/vvjumle/logs/imagenet_convert_%j.err

set -e

module load 2025
module load miniconda3
conda activate LION

SRC=/scratch/vvjumle/imagenet/data
DST=/scratch/vvjumle/imagenet-folder

echo "=== Preparing 500K train subset + full val ==="
python /scratch/vvjumle/LION-but-better-experiments/img_classification/prepare_imagenet_subset.py \
    --src $SRC --dst $DST --train_n 500000

echo "=== Done ==="
