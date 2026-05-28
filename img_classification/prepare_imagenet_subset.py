"""
Stream parquet ImageNet directly to ImageFolder format.
Train: stratified 500K subset (~500 images/class).
Val: full 50K.
No arrow cache — reads parquet directly via pyarrow to stay under file quota.

Usage:
  python prepare_imagenet_subset.py --src /scratch/vvjumle/imagenet/data --dst /scratch/vvjumle/imagenet-folder
"""
import argparse
import os
import glob
import random
from collections import defaultdict
from io import BytesIO

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

TRAIN_TARGET = 500_000
VAL_TARGET = None  # all

def process_split(parquet_files, out_dir, target_n, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)

    # first pass: count per class to compute per-class quota
    if target_n is not None:
        print("Counting class distribution...")
        label_counts = defaultdict(int)
        for f in tqdm(parquet_files, desc="counting"):
            tbl = pq.read_table(f, columns=['label'])
            for label in tbl['label'].to_pylist():
                label_counts[label] += 1
        total = sum(label_counts.values())
        n_classes = len(label_counts)
        per_class_quota = {cls: max(1, round(target_n * cnt / total))
                           for cls, cnt in label_counts.items()}
        print(f"Total={total}, classes={n_classes}, target={target_n}")
    else:
        per_class_quota = None

    # second pass: save images
    saved = defaultdict(int)
    img_idx = defaultdict(int)

    for f in tqdm(parquet_files, desc="converting"):
        tbl = pq.read_table(f, columns=['image', 'label'])
        labels = tbl['label'].to_pylist()
        images = tbl['image'].to_pylist()

        for img_data, label in zip(images, labels):
            if label < 0 or label >= 1000:
                continue
            if per_class_quota and saved[label] >= per_class_quota[label]:
                continue

            cls_dir = os.path.join(out_dir, str(label))
            os.makedirs(cls_dir, exist_ok=True)

            img_path = os.path.join(cls_dir, f'{img_idx[label]}.JPEG')
            img_idx[label] += 1

            if os.path.exists(img_path):
                saved[label] += 1
                continue

            try:
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    img = Image.open(BytesIO(img_data['bytes'])).convert('RGB')
                elif isinstance(img_data, bytes):
                    img = Image.open(BytesIO(img_data)).convert('RGB')
                else:
                    img = img_data.convert('RGB')
                img.save(img_path, 'JPEG', quality=95)
                saved[label] += 1
            except Exception as e:
                print(f"Warning: skipped image {img_path}: {e}")

    total_saved = sum(saved.values())
    print(f"Saved {total_saved} images to {out_dir}")
    return total_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='/scratch/vvjumle/imagenet/data')
    parser.add_argument('--dst', default='/scratch/vvjumle/imagenet-folder')
    parser.add_argument('--train_n', type=int, default=TRAIN_TARGET)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.src, '*.parquet')))
    train_files = [f for f in all_files if 'train' in os.path.basename(f)]
    val_files = [f for f in all_files if 'validation' in os.path.basename(f) or 'test' in os.path.basename(f)]

    print(f"Found {len(train_files)} train parquet files, {len(val_files)} val parquet files")

    print("\n=== Processing train ===")
    process_split(train_files, os.path.join(args.dst, 'train'), args.train_n, args.seed)

    print("\n=== Processing val ===")
    process_split(val_files, os.path.join(args.dst, 'val'), None, args.seed)

    print("\nDone.")


if __name__ == '__main__':
    main()
