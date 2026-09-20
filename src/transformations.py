"""
transformations.py
-------------------
Implements the ten categories of content-preserving image transformations
used in Chapter 3 (Section 3.5, Table 3.1) to emulate common attacks against
perceptual hashing schemes. Each transformation accepts a PIL Image and a
single parameter value, and returns a transformed PIL Image.

The TRANSFORMATIONS dict at the bottom maps each transformation's name to
(function, [list of parameter values]), exactly matching Table 3.1 in the
thesis, and is consumed by generate_dataset.py.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage.util import random_noise
from scipy.ndimage import median_filter as scipy_median_filter, gaussian_filter


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 1. Rotation
# ---------------------------------------------------------------------------
def rotate(img: Image.Image, angle: float) -> Image.Image:
    """Rotate the image by `angle` degrees, keeping the original canvas size."""
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(128, 128, 128))


# ---------------------------------------------------------------------------
# 2. Speckle noise
# ---------------------------------------------------------------------------
def speckle_noise(img: Image.Image, variance: float) -> Image.Image:
    """Add multiplicative speckle noise with the given variance."""
    arr = np.asarray(img).astype(np.float64) / 255.0
    noisy = random_noise(arr, mode="speckle", var=variance, clip=True)
    return Image.fromarray(to_uint8(noisy * 255.0))


# ---------------------------------------------------------------------------
# 3. Gaussian filter (by variance) and 4. Gaussian filter (by filter size)
# ---------------------------------------------------------------------------
def gaussian_filter_by_variance(img: Image.Image, variance: float) -> Image.Image:
    """Blur the image with a Gaussian filter whose sigma is derived from `variance`."""
    sigma = float(np.sqrt(variance) * 10)  # scale so the ten variance settings give a visible range
    arr = np.asarray(img).astype(np.float64)
    blurred = np.stack(
        [gaussian_filter(arr[..., c], sigma=sigma) for c in range(arr.shape[-1])], axis=-1
    )
    return Image.fromarray(to_uint8(blurred))


def gaussian_filter_by_size(img: Image.Image, filter_size: int) -> Image.Image:
    """Blur the image using PIL's Gaussian blur with a radius derived from
    the requested (odd) filter size."""
    radius = max(filter_size / 3.0, 0.1)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


# ---------------------------------------------------------------------------
# 5. Median filter and 6. Circular median filter
# ---------------------------------------------------------------------------
def median_filter(img: Image.Image, filter_size: int) -> Image.Image:
    """Apply a square median filter of the given (odd) window size."""
    size = int(filter_size) if int(filter_size) % 2 == 1 else int(filter_size) + 1
    arr = np.asarray(img)
    filtered = np.stack(
        [scipy_median_filter(arr[..., c], size=size) for c in range(arr.shape[-1])], axis=-1
    )
    return Image.fromarray(to_uint8(filtered))


def circular_median_filter(img: Image.Image, ball_radius: float) -> Image.Image:
    """Apply a median filter using a circular (disk-shaped) footprint of the
    given radius, approximating the 'ball radius' parameter in Table 3.1."""
    r = max(int(round(ball_radius)), 1)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    footprint = (x ** 2 + y ** 2) <= r ** 2
    arr = np.asarray(img)
    filtered = np.stack(
        [scipy_median_filter(arr[..., c], footprint=footprint) for c in range(arr.shape[-1])],
        axis=-1,
    )
    return Image.fromarray(to_uint8(filtered))


# ---------------------------------------------------------------------------
# 7. Gamma correction
# ---------------------------------------------------------------------------
def gamma_correction(img: Image.Image, gamma: float) -> Image.Image:
    """Apply a power-law gamma transform: out = 255 * (in / 255) ** gamma."""
    arr = np.asarray(img).astype(np.float64) / 255.0
    corrected = np.power(arr, gamma) * 255.0
    return Image.fromarray(to_uint8(corrected))


# ---------------------------------------------------------------------------
# 8. Brightness, 9. Colour, 10. Sharpness
# ---------------------------------------------------------------------------
def brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def color(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Color(img).enhance(factor)


def sharpness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Sharpness(img).enhance(factor)


# ---------------------------------------------------------------------------
# Table 3.1: transformation name -> (function, parameter values)
# ---------------------------------------------------------------------------
TRANSFORMATIONS = {
    "rotation": (rotate, [1, 2, 3, 4, 6, 8, 10, 12, 13, 15]),
    "speckle_noise": (speckle_noise, [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]),
    "gaussian_filter_variance": (
        gaussian_filter_by_variance,
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    ),
    "gaussian_filter_size": (gaussian_filter_by_size, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]),
    "median_filter": (median_filter, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]),
    "circular_median_filter": (
        circular_median_filter,
        [1, 1.5, 2, 2.2, 2.5, 2.6, 2.7, 3, 3.5],
    ),
    "gamma": (gamma_correction, [0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45]),
    "brightness": (brightness, [0.7, 0.8, 1, 1.2, 1.5, 2, 2.2, 2.5]),
    "color": (color, [0.2, 0.5, 1, 1.2, 1.3, 1.5, 2, 2.3, 2.5, 3]),
    "sharpness": (sharpness, [0.1, 0.5, 1, 2, 3, 5, 7, 8, 10]),
}
