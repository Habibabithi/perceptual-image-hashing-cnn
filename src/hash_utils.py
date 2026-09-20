"""
hash_utils.py
-------------
Hash generation and similarity measurement, implementing Section 3.3 of the
thesis ("Formal Hash Generation and Similarity Measurement").

Functions
---------
binarize(feature_vector)
    Mean-thresholded binarisation (Equations 3.10-3.11).

hamming_distance(hash_a, hash_b)
    Raw Hamming distance between two binary hash codes (Equation 3.12).

normalized_hamming_distance(hash_a, hash_b)
    Hamming distance normalised to [0, 1] by hash length (Equation 3.13).
"""

import numpy as np


def binarize(feature_vector: np.ndarray) -> np.ndarray:
    """
    Convert a real-valued feature vector h into a binary hash code b,
    using the mean of the vector as an image-adaptive threshold:

        b(i) = 1  if h(i) > mean(h)
        b(i) = 0  if h(i) <= mean(h)

    Parameters
    ----------
    feature_vector : np.ndarray, shape (n,)

    Returns
    -------
    np.ndarray of {0, 1}, shape (n,), dtype=int
    """
    mu = np.mean(feature_vector)
    return (feature_vector > mu).astype(int)


def hamming_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> int:
    """Number of bit positions at which two binary hashes differ."""
    if len(hash_a) != len(hash_b):
        raise ValueError("Hash codes must be of equal length.")
    return int(np.sum(hash_a != hash_b))


def normalized_hamming_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> float:
    """Hamming distance normalised to [0, 1] by the hash length n."""
    n = len(hash_a)
    return hamming_distance(hash_a, hash_b) / n


def hash_to_string(hash_bits: np.ndarray) -> str:
    """Convenience: represent a binary hash as a compact bit-string,
    useful for storing hashes in a CSV file."""
    return "".join(str(b) for b in hash_bits)


def string_to_hash(hash_str: str) -> np.ndarray:
    """Inverse of `hash_to_string`."""
    return np.array([int(c) for c in hash_str], dtype=int)
