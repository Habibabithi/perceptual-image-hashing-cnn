"""
generate_dataset.py
--------------------
Implements Sections 3.4-3.6 of the thesis:

  1. Reads the original images from `data/raw_images/`.
  2. Applies each of the ten transformations in `transformations.TRANSFORMATIONS`
     at each of its ten parameter settings, saving the results to
     `data/transformed/`.
  3. Builds the ground-truth pair list described in Section 3.6:
        G = (original_image, transformed_image, label)
     where label = 1 for a "similar" pair (an image and its own transformed
     variant) and label = 0 for a "dissimilar" pair (two different source
     images). Hamming distances are filled in later by evaluate.py, once
     hashes have been computed by compute_hashes.py.

Usage
-----
    python src/generate_dataset.py \
        --raw_dir data/raw_images \
        --out_dir data/transformed \
        --pairs_csv results/ground_truth_pairs.csv
"""

import argparse
import itertools
import os

import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformations import TRANSFORMATIONS

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(folder: str):
    return sorted(
        [f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTENSIONS)]
    )


def generate_transformed_images(raw_dir: str, out_dir: str) -> pd.DataFrame:
    """Apply every transformation/parameter combination to every raw image.
    Returns a DataFrame recording (original_image, transformed_image,
    transformation, parameter)."""
    os.makedirs(out_dir, exist_ok=True)
    records = []

    image_files = list_images(raw_dir)
    if not image_files:
        raise FileNotFoundError(
            f"No images found in '{raw_dir}'. Place your 50 source images there first."
        )

    for filename in tqdm(image_files, desc="Images"):
        stem, ext = os.path.splitext(filename)
        img_path = os.path.join(raw_dir, filename)
        img = Image.open(img_path).convert("RGB")

        for transform_name, (func, param_values) in TRANSFORMATIONS.items():
            for param in param_values:
                out_name = f"{stem}__{transform_name}__{param}{ext}"
                out_path = os.path.join(out_dir, out_name)
                if not os.path.exists(out_path):
                    transformed = func(img, param)
                    transformed.save(out_path)
                records.append(
                    {
                        "original_image": filename,
                        "transformed_image": out_name,
                        "transformation": transform_name,
                        "parameter": param,
                    }
                )

    return pd.DataFrame(records)


def build_ground_truth_pairs(transform_df: pd.DataFrame, raw_dir: str) -> pd.DataFrame:
    """
    Build the full set of similar and dissimilar pairs described in
    Section 3.6 / Table 3.2:

      - Similar pairs: (original_image, its own transformed_image), label = 1.
      - Dissimilar pairs: (original_image, a transformed_image derived from a
        DIFFERENT source image), label = 0.

    This reproduces the 1 : 49 similar-to-dissimilar ratio used in the
    thesis for a 50-image dataset (each original is compared against its own
    100 transformed variants, and against the ~4,900 transformed variants of
    the other 49 images filtered down as needed for tractable evaluation).
    """
    originals = sorted(transform_df["original_image"].unique())
    rows = []

    for orig in originals:
        own_variants = transform_df.loc[transform_df["original_image"] == orig, "transformed_image"]
        for variant in own_variants:
            rows.append({"image_a": orig, "image_b": variant, "label": 1})

        other_variants = transform_df.loc[transform_df["original_image"] != orig, "transformed_image"]
        for variant in other_variants:
            rows.append({"image_a": orig, "image_b": variant, "label": 0})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate transformed images and ground-truth pairs.")
    parser.add_argument("--raw_dir", default="data/raw_images")
    parser.add_argument("--out_dir", default="data/transformed")
    parser.add_argument("--transform_csv", default="results/transform_log.csv")
    parser.add_argument("--pairs_csv", default="results/ground_truth_pairs.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.transform_csv), exist_ok=True)

    transform_df = generate_transformed_images(args.raw_dir, args.out_dir)
    transform_df.to_csv(args.transform_csv, index=False)
    print(f"Saved transformation log ({len(transform_df)} rows) to {args.transform_csv}")

    pairs_df = build_ground_truth_pairs(transform_df, args.raw_dir)
    pairs_df.to_csv(args.pairs_csv, index=False)
    print(f"Saved ground-truth pairs ({len(pairs_df)} rows) to {args.pairs_csv}")


if __name__ == "__main__":
    main()
