"""
compute_hashes.py
------------------
Implements Algorithm 1 (Section 3.3.3) over every original and transformed
image in the dataset, and saves the resulting binary hash codes to a CSV
file for later use by evaluate.py.

Usage
-----
    python src/compute_hashes.py \
        --raw_dir data/raw_images \
        --transformed_dir data/transformed \
        --backbone vgg16 \
        --out_csv results/hash_database.csv
"""

import argparse
import os

import pandas as pd
from tqdm import tqdm

from feature_extractor import FeatureExtractor
from hash_utils import binarize, hash_to_string

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(folder: str):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTENSIONS)])


def main():
    parser = argparse.ArgumentParser(description="Compute VGG16 perceptual hashes for all images.")
    parser.add_argument("--raw_dir", default="data/raw_images")
    parser.add_argument("--transformed_dir", default="data/transformed")
    parser.add_argument("--backbone", default="vgg16", choices=["vgg16", "vgg19"])
    parser.add_argument("--out_csv", default="results/hash_database.csv")
    args = parser.parse_args()

    extractor = FeatureExtractor(backbone=args.backbone)

    all_images = []
    for folder in (args.raw_dir, args.transformed_dir):
        for fname in list_images(folder):
            all_images.append((fname, os.path.join(folder, fname)))

    records = []
    for fname, path in tqdm(all_images, desc="Hashing"):
        feature_vector = extractor.extract(path)          # Equation 3.9
        hash_bits = binarize(feature_vector)               # Equations 3.10-3.11
        records.append({"image": fname, "hash": hash_to_string(hash_bits)})

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pd.DataFrame(records).to_csv(args.out_csv, index=False)
    print(f"Saved {len(records)} hashes to {args.out_csv}")


if __name__ == "__main__":
    main()
