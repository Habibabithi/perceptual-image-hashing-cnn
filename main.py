"""
main.py
-------
Runs the full perceptual-image-hashing pipeline end to end:

    1. generate_dataset.py  -> apply the 10 transformations, build ground truth pairs
    2. compute_hashes.py    -> extract VGG16 features and binarise them into hashes
    3. evaluate.py          -> sweep thresholds, plot the ROC curve, report AUC/EER

Run each stage individually (see src/<script>.py --help) if you only want to
redo one part of the pipeline, e.g. after adding new raw images.

Usage
-----
    python main.py
"""

import subprocess
import sys

STAGES = [
    ["python", "src/generate_dataset.py"],
    ["python", "src/compute_hashes.py"],
    ["python", "src/evaluate.py"],
]


def run():
    for stage in STAGES:
        print(f"\n=== Running: {' '.join(stage)} ===")
        result = subprocess.run(stage)
        if result.returncode != 0:
            print(f"Stage failed: {' '.join(stage)}")
            sys.exit(result.returncode)
    print("\nPipeline complete. See results/roc_curve.png and results/threshold_metrics.csv")


if __name__ == "__main__":
    run()
