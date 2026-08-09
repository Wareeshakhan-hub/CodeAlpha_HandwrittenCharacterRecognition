"""
utils.py
--------
Shared helper functions for the Handwritten Character Recognition project.

Project : CodeAlpha Machine Learning Internship - Task 3
Author  : Wareesha Khan
Description:
    Keeps the preprocessing used at TRAINING time and at INFERENCE time
    (in the Streamlit app) perfectly identical. This is the single most
    common bug in handwriting-recognition demos - if you train on one
    kind of image and predict on another, accuracy silently collapses.
"""

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# EMNIST "Balanced" label mapping (47 classes)
# Source: official EMNIST balanced-mapping.txt (NIST / Cohen et al., 2017)
# Balanced merges visually-similar upper/lower case letters (e.g. 'c'/'C')
# so classes stay evenly represented and the model isn't punished for a
# confusion humans make too.
# ---------------------------------------------------------------------------
EMNIST_BALANCED_MAPPING = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n',
    44: 'q', 45: 'r', 46: 't',
}

NUM_CLASSES = len(EMNIST_BALANCED_MAPPING)  # 47
IMG_SIZE = 28


def fix_emnist_orientation(images: np.ndarray) -> np.ndarray:
    """
    EMNIST images are stored transposed (rows/cols swapped) relative to the
    orientation a human expects. This swaps them back so a '7' actually
    looks like a '7' when you plot it. Apply this ONCE, right after loading
    the raw arrays, before any other preprocessing.

    images: array of shape (N, 28, 28)
    """
    return np.transpose(images, axes=(0, 2, 1))


def normalize_batch(images: np.ndarray) -> np.ndarray:
    """
    Scales to [0, 1] and reshapes to (N, 28, 28, 1) for the CNN input.
    images: array of shape (N, 28, 28), values 0-255
    """
    images = images.astype('float32') / 255.0
    return images.reshape((-1, IMG_SIZE, IMG_SIZE, 1))


def preprocess_canvas_image(raw_image: np.ndarray) -> np.ndarray:
    """
    Turns a raw drawn/uploaded image (RGBA or RGB or grayscale, any size,
    dark-strokes-on-light-background OR light-strokes-on-dark-background)
    into a single (1, 28, 28, 1) float32 array ready for model.predict().

    Steps (mirrors how MNIST/EMNIST characters are framed):
      1. Convert to single-channel grayscale.
      2. Auto-detect polarity so strokes end up WHITE on BLACK (like EMNIST).
      3. Crop tightly to the bounding box of the stroke (remove empty margin).
      4. Pad to a square and resize to 28x28 - this centering step matters
         a lot for real-world accuracy; skipping it is a common reason
         "it works on the test set but not when I draw it myself" happens.
      5. Normalize to [0, 1].
    """
    img = raw_image

    # Flatten alpha / RGB down to grayscale
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = img.astype('uint8')

    # Make sure strokes are WHITE on a BLACK background (EMNIST convention).
    # If the image is mostly bright (light background, dark pen strokes),
    # invert it.
    if img.mean() > 127:
        img = 255 - img

    # Threshold + find the bounding box of the actual stroke
    _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    if coords is None:
        # Blank canvas - nothing drawn yet
        return normalize_batch(np.zeros((1, IMG_SIZE, IMG_SIZE), dtype='float32'))

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y + h, x:x + w]

    # Pad to a square with a small margin, then resize down to 20x20 and
    # place inside a 28x28 canvas (mirrors the original MNIST/EMNIST
    # construction process, which improves recognition accuracy).
    side = max(w, h)
    margin = int(side * 0.35)
    square = np.zeros((side + 2 * margin, side + 2 * margin), dtype='uint8')
    y_off = margin + (side - h) // 2
    x_off = margin + (side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = cropped

    resized = cv2.resize(square, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    return normalize_batch(resized.reshape(1, IMG_SIZE, IMG_SIZE))


def top_k_predictions(probabilities: np.ndarray, k: int = 5):
    """
    probabilities: 1D array of length NUM_CLASSES (softmax output for one image)
    Returns a list of (character, confidence_percent) sorted descending.
    """
    top_indices = np.argsort(probabilities)[::-1][:k]
    return [(EMNIST_BALANCED_MAPPING[i], float(probabilities[i]) * 100) for i in top_indices]
