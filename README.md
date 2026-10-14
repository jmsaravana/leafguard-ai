# LeafGuard AI 🌿
### Early Plant Disease Detection Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)

A beginner-friendly deep learning mini-project that detects plant diseases from leaf images using a Convolutional Neural Network (CNN) and displays results in a modern Streamlit web app.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🖼️ Image Upload | JPG, JPEG, PNG |
| 🧠 CNN Architecture | 4-block custom CNN (224 × 224 input) |
| 🌱 Supported Classes | 6 (see below) |
| 🎨 UI | Green-themed Streamlit dashboard |
| 🚧 Demo Mode | Clearly labelled simulated predictions when model is absent |
| ⚠️ Disclaimer | Prominent experimental-results warning |

**Supported Classes**
1. Healthy
2. Tomato Early Blight
3. Tomato Late Blight
4. Tomato Leaf Mold
5. Potato Early Blight
6. Potato Late Blight

---

## 📁 Project Structure

```
leafguard AI/
├── app.py               ← Streamlit web application
├── train_model.py       ← CNN training script
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
├── dataset/             ← (you create this) labelled image folders
│   ├── Healthy/
│   ├── Tomato Early Blight/
│   ├── Tomato Late Blight/
│   ├── Tomato Leaf Mold/
│   ├── Potato Early Blight/
│   └── Potato Late Blight/
├── leafguard_model.keras  ← (generated after training)
└── training_curves.png    ← (generated after training)
```

---

## 🚀 Quick Start

### 1 — Prerequisites

- **Python 3.9 or higher** — https://python.org/downloads
- **pip** (usually bundled with Python)

### 2 — Install Dependencies

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

> ⏳ TensorFlow is a large package (~500 MB). Installation may take a few minutes.

### 3 — (Optional) Prepare the Dataset

> Skip this step if you just want to explore the app in **Demo Mode**.

Download a plant disease dataset.  The recommended free dataset is
**PlantVillage** on Kaggle:

```
https://www.kaggle.com/datasets/emmarex/plantdisease
```

Organise the images into the following folder structure inside the project:

```
dataset/
├── Healthy/
│   ├── image001.jpg
│   └── ...
├── Tomato Early Blight/
│   └── ...
├── Tomato Late Blight/
│   └── ...
├── Tomato Leaf Mold/
│   └── ...
├── Potato Early Blight/
│   └── ...
└── Potato Late Blight/
    └── ...
```

> ℹ️ Sub-folder names must **exactly** match the class names above.
> Aim for at least **50 images per class** (more is better).

### 4 — Train the Model

```bash
python train_model.py
```

What happens during training:
- Dataset folders are verified.
- Images are split 80 % training / 20 % validation.
- Data augmentation (flips, rotations, zoom, brightness) is applied.
- The CNN trains for up to 30 epochs with early stopping.
- The best model is saved as `leafguard_model.keras`.
- A training curve plot is saved as `training_curves.png`.

### 5 — Launch the Web App

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`).
Open it in your browser.

---

## 🎮 Demo Mode

If `leafguard_model.keras` is **not** found when you start the app,
it runs in **Demo Mode**:

- You can still upload images and click **Analyse Leaf**.
- The predicted class and confidence are **randomly simulated** — they are NOT real model predictions.
- All demo results are clearly labelled with a red dashed warning banner.
- This mode is provided only to let you explore the UI before training.

---

## 🏗️ CNN Architecture

```
Input (224 × 224 × 3)
│
├── Rescaling [0–255 → 0–1]
├── Data Augmentation (train only)
│
├── Conv2D(32)  → BatchNorm → ReLU → MaxPool
├── Conv2D(64)  → BatchNorm → ReLU → MaxPool
├── Conv2D(128) → BatchNorm → ReLU → MaxPool
├── Conv2D(256) → BatchNorm → ReLU → MaxPool
│
├── GlobalAveragePooling2D
├── Dense(256, ReLU)
├── Dropout(0.4)
└── Dense(6, Softmax)  → class probabilities
```

**Training settings**

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam (lr = 0.001) |
| Loss | Categorical Cross-Entropy |
| Batch size | 32 |
| Max epochs | 30 |
| Early stopping patience | 5 |
| LR reduction patience | 3 |
| Validation split | 20 % |

---

## ⚠️ Important Disclaimer

> Predictions made by LeafGuard AI are **experimental** and intended for
> **educational purposes only**.  They do **not** replace professional
> agricultural advice.  Always consult a qualified agronomist or plant
> pathologist before taking action on your crops.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `MISSING folder` error | Check dataset folder names match class names exactly |
| App shows Demo Mode | Train the model first (`python train_model.py`) |
| Low accuracy after training | Add more images; aim for 200+ per class |
| Slow training | Use a GPU-enabled machine or Google Colab |

---

## 🌐 Running on Google Colab (Free GPU)

1. Upload the project files to your Google Drive.
2. Open a new Colab notebook.
3. Mount Drive and `cd` to the project folder.
4. Run `!pip install -r requirements.txt`.
5. Upload your dataset.
6. Run `!python train_model.py`.
7. Download `leafguard_model.keras` and use it locally.

---

## 📜 License

This project is for educational purposes.  Feel free to modify and extend it.
