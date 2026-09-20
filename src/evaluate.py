"""
evaluate.py
-----------
Implements Algorithm 2 (Section 3.6) and the evaluation metrics of Chapter 4:
TPR, FPR, Precision, F1, the ROC curve, AUC (trapezoidal rule), the Equal
Error Rate (EER), and the Youden's-J-optimal threshold.

Usage
-----
    python src/evaluate.py \
        --pairs_csv results/ground_truth_pairs.csv \
        --hash_csv results/hash_database.csv \
        --roc_png results/roc_curve.png \
        --metrics_csv results/threshold_metrics.csv
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hash_utils import string_to_hash, normalized_hamming_distance


def compute_distances(pairs_df: pd.DataFrame, hash_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the normalised Hamming distance d_H(b_A, b_B) (Equation 3.13)
    to every pair in `pairs_df`, looking up each image's hash in `hash_df`."""
    hash_lookup = dict(zip(hash_df["image"], hash_df["hash"]))

    distances = []
    for _, row in pairs_df.iterrows():
        h_a = hash_lookup.get(row["image_a"])
        h_b = hash_lookup.get(row["image_b"])
        if h_a is None or h_b is None:
            distances.append(np.nan)
            continue
        d = normalized_hamming_distance(string_to_hash(h_a), string_to_hash(h_b))
        distances.append(d)

    pairs_df = pairs_df.copy()
    pairs_df["hamming_distance"] = distances
    return pairs_df.dropna(subset=["hamming_distance"])


def sweep_thresholds(pairs_df: pd.DataFrame, n_thresholds: int = 200) -> pd.DataFrame:
    """
    Algorithm 2: sweep the decision threshold epsilon over [0, 1] and compute
    TP, FP, FN, TN, TPR, FPR, Precision, and F1 at each threshold
    (Equations 3.18-3.19, 4.1-4.2).
    """
    y_true = pairs_df["label"].values
    d = pairs_df["hamming_distance"].values

    thresholds = np.linspace(0, 1, n_thresholds)
    rows = []

    for eps in thresholds:
        y_pred = (d < eps).astype(int)  # Equation 3.17

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (tn + fp) if (tn + fp) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * precision * tpr / (precision + tpr)) if (precision + tpr) else 0.0

        rows.append(
            {"epsilon": eps, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
             "TPR": tpr, "FPR": fpr, "Precision": precision, "F1": f1}
        )

    return pd.DataFrame(rows)


def compute_auc(metrics_df: pd.DataFrame) -> float:
    """
    Trapezoidal-rule AUC (Equation 4.3):

        AUC = sum_j  0.5 * (TPR_j + TPR_{j+1}) * (FPR_{j+1} - FPR_j)

    Implemented manually (rather than via np.trapz/np.trapezoid) so the
    result is identical across NumPy versions.
    """
    sorted_df = metrics_df.sort_values("FPR").reset_index(drop=True)
    fpr = sorted_df["FPR"].values
    tpr = sorted_df["TPR"].values

    auc = 0.0
    for j in range(len(fpr) - 1):
        auc += 0.5 * (tpr[j] + tpr[j + 1]) * (fpr[j + 1] - fpr[j])
    return float(auc)


def compute_eer_and_best_threshold(metrics_df: pd.DataFrame):
    """Equal Error Rate (Equation 4.4) and Youden's-J-optimal threshold
    (Equation 4.5)."""
    j = metrics_df["TPR"] - metrics_df["FPR"]
    best_idx = j.idxmax()
    best_row = metrics_df.loc[best_idx]

    fnr = 1 - metrics_df["TPR"]
    eer_idx = (metrics_df["FPR"] - fnr).abs().idxmin()
    eer = float(metrics_df.loc[eer_idx, "FPR"])

    return eer, float(best_row["epsilon"]), float(best_row["TPR"]), float(best_row["FPR"])


def plot_roc(metrics_df: pd.DataFrame, auc: float, out_path: str):
    plt.figure(figsize=(6, 5))
    plt.plot(metrics_df["FPR"], metrics_df["TPR"], label=f"VGG16 hash (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guessing")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve — VGG16-Based Perceptual Hashing")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved ROC curve to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the perceptual hashing scheme.")
    parser.add_argument("--pairs_csv", default="results/ground_truth_pairs.csv")
    parser.add_argument("--hash_csv", default="results/hash_database.csv")
    parser.add_argument("--roc_png", default="results/roc_curve.png")
    parser.add_argument("--metrics_csv", default="results/threshold_metrics.csv")
    parser.add_argument("--n_thresholds", type=int, default=200)
    args = parser.parse_args()

    pairs_df = pd.read_csv(args.pairs_csv)
    hash_df = pd.read_csv(args.hash_csv)

    pairs_df = compute_distances(pairs_df, hash_df)
    metrics_df = sweep_thresholds(pairs_df, n_thresholds=args.n_thresholds)
    metrics_df.to_csv(args.metrics_csv, index=False)

    auc = compute_auc(metrics_df)
    eer, best_eps, best_tpr, best_fpr = compute_eer_and_best_threshold(metrics_df)

    print(f"AUC                       : {auc:.4f}")
    print(f"Equal Error Rate (EER)    : {eer:.4f}")
    print(f"Youden's-J-optimal eps*   : {best_eps:.4f}  (TPR={best_tpr:.4f}, FPR={best_fpr:.4f})")

    plot_roc(metrics_df, auc, args.roc_png)


if __name__ == "__main__":
    main()
