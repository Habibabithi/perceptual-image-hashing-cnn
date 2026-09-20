"""
feature_extractor.py
---------------------
CNN-based feature extraction for perceptual image hashing.

This module implements the feature-extraction stage of the hashing pipeline
described in Chapter 3 of the thesis:

    1. An input image is resized to 224x224x3 and preprocessed for VGG16.
    2. The image is forward-propagated through the VGG16 convolutional base
       (pretrained on ImageNet, classification head removed).
    3. Global average pooling is applied over the final 7x7x512 feature map,
       producing a single 512-dimensional feature vector per image
       (Equations 3.7 and 3.9 in the thesis).

Using Keras' VGG16(include_top=False, pooling="avg") performs steps 2 and 3
in a single call, since `pooling="avg"` applies global average pooling to
the output of the last convolutional block automatically.

Architectures supported: "vgg16" (default, used for the final scheme),
"vgg19", and "alexnet"-style shallow CNN (bonus comparison network used in
Chapter 3 discussion). VGG19 uses the same interface and also outputs a
512-dim vector via average pooling.
"""

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess
from tensorflow.keras.preprocessing import image as keras_image

IMG_SIZE = (224, 224)


class FeatureExtractor:
    """Wraps a pretrained CNN backbone and exposes a simple `.extract(path)` API."""

    def __init__(self, backbone: str = "vgg16"):
        backbone = backbone.lower()
        self.backbone = backbone

        if backbone == "vgg16":
            self.model = VGG16(weights="imagenet", include_top=False, pooling="avg")
            self.preprocess = vgg16_preprocess
        elif backbone == "vgg19":
            self.model = VGG19(weights="imagenet", include_top=False, pooling="avg")
            self.preprocess = vgg19_preprocess
        else:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. Choose 'vgg16' or 'vgg19'."
            )

    def _load_and_preprocess(self, image_path: str) -> np.ndarray:
        """Load an image from disk, resize to 224x224, and apply the
        network-specific preprocessing (mean-subtraction / channel scaling)."""
        img = keras_image.load_img(image_path, target_size=IMG_SIZE)
        arr = keras_image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = self.preprocess(arr)
        return arr

    def extract(self, image_path: str) -> np.ndarray:
        """Return the 512-dimensional feature vector h for a single image
        (Equation 3.9). This is the pre-binarisation perceptual embedding."""
        arr = self._load_and_preprocess(image_path)
        features = self.model.predict(arr, verbose=0)
        return features.flatten()  # shape: (512,)

    def extract_batch(self, image_paths: list) -> np.ndarray:
        """Vectorised version of `extract` for a list of image paths.
        Returns an array of shape (N, 512)."""
        batch = np.vstack([self._load_and_preprocess(p) for p in image_paths])
        features = self.model.predict(batch, verbose=0)
        return features.reshape(len(image_paths), -1)
