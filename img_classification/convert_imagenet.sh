#!/bin/bash
#SBATCH --job-name=imagenet_convert
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/vvjumle/logs/imagenet_convert_%j.out
#SBATCH --error=/scratch/vvjumle/logs/imagenet_convert_%j.err

set -e

module load 2025
module load miniconda3
conda activate LION

SRC=/scratch/vvjumle/imagenet/data
DST=/scratch/vvjumle/imagenet-folder

echo "=== Converting train split ==="
python /scratch/vvjumle/LION-but-better-experiments/img_classification/convert_imagenet.py \
    --split train --src $SRC --dst $DST

echo "=== Converting validation split ==="
python /scratch/vvjumle/LION-but-better-experiments/img_classification/convert_imagenet.py \
    --split validation --src $SRC --dst $DST

echo "=== Done ==="
