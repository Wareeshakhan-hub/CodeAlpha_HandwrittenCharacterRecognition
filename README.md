<div align="center">

# ✍️ Handwritten Character Recognition

### Deep learning system that reads handwritten digits and letters in real time

**CodeAlpha Machine Learning Internship — Task 3**
Author: **Wareesha Khan**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-informational?style=flat-square)
![CodeAlpha](https://img.shields.io/badge/CodeAlpha-Internship-6D28D9?style=flat-square)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Design Notes](#design-notes--how-predictions-stay-accurate)
- [Roadmap](#roadmap)
- [Author](#author)
- [License](#license)

---

## Overview

This repository contains **Task 3** of the [CodeAlpha](https://www.codealpha.tech) Machine Learning
Internship: a Convolutional Neural Network that recognizes handwritten **digits (0–9)** and
**letters (A–Z, a–z)**, trained on the **EMNIST Balanced** dataset (47 classes, ~131,600 images).

The model is served through an interactive Streamlit application — draw a character with your mouse
and get an instant prediction with a confidence breakdown.

| | |
|---|---|
| **Objective** | Identify handwritten characters and digits |
| **Approach** | Image processing + deep learning (CNN) |
| **Dataset** | EMNIST — Balanced split |
| **Extendable to** | Full word / sentence recognition via sequence modeling (CRNN) |

---

## Features

- 🖌️ **Live drawing canvas** — sketch a character and get a prediction in under a second
- 📊 **Confidence breakdown** — top-5 candidate characters with confidence scores, not just a single guess
- 📜 **Prediction history** — every prediction is logged (timestamp, character, confidence, source) to a
  local CSV, viewable, exportable, and clearable from inside the app
- 🎨 **Custom-designed UI** — a purpose-built dark theme, not the default Streamlit look
- 🧪 **Data sanity check** — a lightweight script to verify image orientation before committing to a full
  training run

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | TensorFlow / Keras (CNN) |
| Data | `tensorflow-datasets` (EMNIST Balanced) |
| Web app | Streamlit + `streamlit-drawable-canvas` |
| Image preprocessing | OpenCV, Pillow, NumPy |
| History log | Pandas (CSV) |

---

## Model Architecture

A 3-block CNN ending in a 47-way softmax over the EMNIST Balanced label set:

```
Input (28×28×1)
 → Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout
 → Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout
 → Conv2D(128) → BatchNorm → MaxPool → Dropout
 → Flatten → Dense(256) → BatchNorm → Dropout
 → Dense(47, softmax)
```

Trained with light data augmentation (rotation/shift/zoom to mimic natural handwriting variance), the
`Adam` optimizer, early stopping, and learning-rate reduction on plateau.

**Results:** after running `train_model.py`, replace this line with your own test accuracy, and add the
generated `model/training_history.png` and `model/confusion_matrix.png` here.

---

## Project Structure

```
CodeAlpha_HandwrittenCharacterRecognition/
├── check_data.py        # Quick sanity check: confirms image orientation before full training
├── train_model.py       # Loads EMNIST, builds and trains the CNN, saves the model
├── app.py                # Streamlit app: draw a character, get a live prediction, view history
├── utils.py               # Shared preprocessing (identical at train & inference time) + label map
├── requirements.txt
├── .gitignore
├── model/                 # Created after training: model file, label map, evaluation plots
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Wareeshakhan-hub/CodeAlpha_HandwrittenCharacterRecognition.git
cd CodeAlpha_HandwrittenCharacterRecognition
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Sanity-check the data

```bash
python check_data.py
```

Downloads a small slice of EMNIST and saves `sample_check.png` — open it and confirm the characters
look normal and right-way-up before committing to a full training run.

### 4. Train the model

```bash
python train_model.py
```

Downloads EMNIST Balanced (~200 MB, via `tensorflow-datasets`) on first run and trains the CNN, saving
everything to `model/`. Takes roughly 20–30 minutes on CPU, faster on GPU.

### 5. Launch the app

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`), draw a character, and
click **Predict**.

### Troubleshooting

| Issue | Fix |
|---|---|
| `zipfile.BadZipFile` / download fails | This project uses `tensorflow-datasets`, not the older `emnist` PyPI package (whose Google Drive link is prone to rate-limiting). Re-run `pip install -r requirements.txt` to make sure you're on the current dependency set. |
| Download blocked on your network | Run `check_data.py` / `train_model.py` on [Google Colab](https://colab.research.google.com) instead (free, full internet + GPU), then download the resulting `model/` folder back into this project. |
| Characters look rotated/mirrored in `sample_check.png` | Open `utils.py` and remove the `fix_emnist_orientation()` call in `load_data()`, then re-run `check_data.py`. |

---

## Design Notes — How Predictions Stay Accurate

A common reason handwriting demos work on the test set but fail on a user's own drawing is that the
live input isn't preprocessed the same way the training images were. This project centralizes that
logic in `utils.py` so both `train_model.py` and `app.py` use the **exact same** pipeline:

1. Orientation fix for EMNIST's transposed image format.
2. Grayscale + polarity normalization (strokes white-on-black).
3. Tight crop to the character's bounding box, then centered padding — mirrors how EMNIST/MNIST
   characters were originally framed.
4. Resize to 28×28 and normalize to `[0, 1]`.

---

## Roadmap

- [ ] Sequence modeling (CRNN + CTC loss) for full **word/sentence** recognition
- [ ] Fine-tuning on a personal handwriting sample
- [ ] Export to TensorFlow Lite for a mobile version

---

## Author

**Wareesha Khan**
Machine Learning Intern @ CodeAlpha

Built as part of the **CodeAlpha Machine Learning Internship** · [www.codealpha.tech](https://www.codealpha.tech)

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

<div align="center">
<sub>© 2026 Wareesha Khan — Handwritten Character Recognition — CodeAlpha ML Internship, Task 3</sub>
</div>