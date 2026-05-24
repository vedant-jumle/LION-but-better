"""
Convert HuggingFace parquet ImageNet to ImageFolder format.
Usage: python convert_imagenet.py --split train --src /scratch/vvjumle/imagenet/data --dst /scratch/vvjumle/imagenet-folder
"""
import argparse
import os
from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='train', choices=['train', 'validation', 'test'])
parser.add_argument('--src', default='/scratch/vvjumle/imagenet/data')
parser.add_argument('--dst', default='/scratch/vvjumle/imagenet-folder')
args = parser.parse_args()

print(f"Loading {args.split} split from {args.src}...")
ds = load_dataset('parquet', data_dir=args.src, split=args.split, num_proc=4)

out_dir = os.path.join(args.dst, args.split)
os.makedirs(out_dir, exist_ok=True)

print(f"Converting {len(ds)} samples to {out_dir}...")
for i, sample in enumerate(tqdm(ds, desc=args.split)):
    cls = str(sample['label'])
    cls_dir = os.path.join(out_dir, cls)
    os.makedirs(cls_dir, exist_ok=True)
    img_path = os.path.join(cls_dir, f'{i}.JPEG')
    if not os.path.exists(img_path):
        sample['image'].save(img_path, 'JPEG')

print(f"Done. Output at {out_dir}")
