"""
train_model.py
---------------
CodeAlpha Machine Learning Internship - Task 3: Handwritten Character Recognition
Author: Wareesha Khan

Objective : Identify handwritten characters (digits 0-9 and letters A-Z / a-z).
Approach  : Image processing + deep learning (CNN) on the EMNIST "Balanced" split.
Dataset   : EMNIST (Extended MNIST) - Balanced, 47 classes, ~131,600 images.

Run:
    python train_model.py

Output:
    model/handwritten_char_cnn.keras   <- trained model
    model/label_mapping.json           <- class index -> character
    model/training_history.png         <- accuracy/loss curves
    model/confusion_matrix.png         <- test-set confusion matrix

NOTE ON RUNNING THIS SCRIPT
----------------------------
This script downloads EMNIST Balanced (~200 MB) the first time it runs,
via `tensorflow-datasets` (Google's official dataset servers). This is
used instead of the older `emnist` PyPI package, whose Google Drive link
is frequently rate-limited and can silently hand back a corrupted/HTML
file instead of the real dataset (`zipfile.BadZipFile: File is not a zip
file` is the classic symptom of that).

Tip: before running the full training below, run `python check_data.py`
first - it downloads a tiny slice of the data and saves `sample_check.png`
so you can visually confirm the characters look right-way-up before
committing to a full training run.

If the download still fails on your network:
    1. Easiest fix: run this script on Google Colab (free, full internet
       + GPU): https://colab.research.google.com
    2. Or download the EMNIST Balanced files manually from
       https://www.nist.gov/itl/products-and-services/emnist-dataset
       or the Kaggle mirror https://www.kaggle.com/datasets/crawford/emnist
       and adapt `load_data()` below to read from disk instead.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')  # so this runs headless (no display needed)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import EMNIST_BALANCED_MAPPING, NUM_CLASSES, IMG_SIZE, fix_emnist_orientation, normalize_batch

MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_data():
    """Downloads (if needed) and returns EMNIST Balanced train/test splits,
    correctly oriented and normalized. Uses tensorflow-datasets (downloads
    from Google's dataset servers) rather than the `emnist` PyPI package,
    which relies on a Google Drive link that frequently gets rate-limited."""
    import tensorflow_datasets as tfds

    print("Loading EMNIST 'balanced' training data (first run downloads ~200 MB)...")
    train_ds = tfds.load('emnist/balanced', split='train', batch_size=-1, as_supervised=True)
    test_ds = tfds.load('emnist/balanced', split='test', batch_size=-1, as_supervised=True)

    x_train, y_train = tfds.as_numpy(train_ds)
    x_test, y_test = tfds.as_numpy(test_ds)

    # tfds gives (N, 28, 28, 1) uint8 - drop the channel dim before applying
    # our shared orientation-fix / normalize helpers (which expect (N,28,28)).
    x_train = x_train.squeeze(-1)
    x_test = x_test.squeeze(-1)

    # EMNIST images are stored rotated 90° + mirrored (this is documented by
    # both NIST and tensorflow-datasets itself) - fix_emnist_orientation()
    # undoes that so a '7' actually looks like a '7'.
    x_train = fix_emnist_orientation(x_train)
    x_test = fix_emnist_orientation(x_test)

    x_train = normalize_batch(x_train)
    x_test = normalize_batch(x_test)

    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")
    print(f"Classes: {NUM_CLASSES}")
    return x_train, y_train, x_test, y_test


def build_model():
    """CNN architecture: two conv blocks + dense head. Sized to comfortably
    beat 85%+ on EMNIST Balanced (a genuinely harder dataset than MNIST,
    since some letters are visually ambiguous even for humans, e.g. 'l'/'I'/'1')."""
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ], name='handwritten_char_cnn')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'], label='train')
    axes[0].plot(history.history['val_accuracy'], label='val')
    axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].legend()

    axes[1].plot(history.history['loss'], label='train')
    axes[1].plot(history.history['val_loss'], label='val')
    axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].legend()

    fig.suptitle('Handwritten Character Recognition - Training History (Wareesha Khan)')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'), dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred):
    labels = [EMNIST_BALANCED_MAPPING[i] for i in range(NUM_CLASSES)]
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(labels, fontsize=7, rotation=90)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix - EMNIST Balanced (47 classes)')
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close(fig)


def main():
    x_train, y_train, x_test, y_test = load_data()

    # Light augmentation - handwriting varies in slant/position/size, so a
    # model that's only ever seen perfectly centered characters generalizes
    # poorly to a user's own drawing in the Streamlit app.
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
    )
    datagen.fit(x_train)

    model = build_model()
    model.summary()

    cb = [
        callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'handwritten_char_cnn.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=1,
        ),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    ]

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        validation_data=(x_test, y_test),
        epochs=30,
        callbacks=cb,
    )

    plot_history(history)

    # Final evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal test accuracy: {test_acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print("\nClassification report:\n")
    print(classification_report(
        y_test, y_pred,
        target_names=[EMNIST_BALANCED_MAPPING[i] for i in range(NUM_CLASSES)],
        zero_division=0,
    ))
    plot_confusion_matrix(y_test, y_pred)

    # Save label mapping alongside the model so app.py never has to guess it
    with open(os.path.join(MODEL_DIR, 'label_mapping.json'), 'w') as f:
        json.dump({str(k): v for k, v in EMNIST_BALANCED_MAPPING.items()}, f, indent=2)

    print(f"\nSaved model + mapping + plots to '{MODEL_DIR}/'.")
    print("Next step: streamlit run app.py")


if __name__ == '__main__':
    main()
