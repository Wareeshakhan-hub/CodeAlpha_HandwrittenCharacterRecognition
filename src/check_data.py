"""
check_data.py
--------------
CodeAlpha Machine Learning Internship - Task 3
Author: Wareesha Khan

Run this BEFORE train_model.py. It downloads a tiny slice of EMNIST
(20 images, a few seconds) and saves a labeled grid so you can visually
confirm the characters look right-way-up before committing to a full
~20-30 minute training run.

Run:
    python check_data.py
Then open sample_check.png (created in this same folder).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

from utils import EMNIST_BALANCED_MAPPING, fix_emnist_orientation


def main():
    print("Downloading a small sample of EMNIST 'balanced' to sanity-check ...")
    ds = tfds.load('emnist/balanced', split='train[:20]', as_supervised=True)

    images, labels = [], []
    for img, label in tfds.as_numpy(ds):
        images.append(img.squeeze(-1))
        labels.append(int(label))

    images = fix_emnist_orientation(np.array(images))

    fig, axes = plt.subplots(4, 5, figsize=(10, 8.5))
    for ax, img, label in zip(axes.flat, images, labels):
        ax.imshow(img, cmap='gray')
        ax.set_title(f"'{EMNIST_BALANCED_MAPPING[label]}'", fontsize=16)
        ax.axis('off')

    fig.suptitle(
        "Sanity check - Wareesha Khan\n"
        "If these characters look normal and right-way-up, you're good to run train_model.py.\n"
        "If they look rotated/mirrored, open utils.py and remove the fix_emnist_orientation() call.",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig('sample_check.png', dpi=120)
    print("\nSaved sample_check.png - open it and confirm the characters look correct.")
    print("If they look right-way-up: run `python train_model.py` next.")


if __name__ == '__main__':
    main()
