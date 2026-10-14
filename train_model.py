# =============================================================================
# LeafGuard AI — CNN Training Script
# =============================================================================
# Usage:
#   python train_model.py
#
# Dataset layout expected (one sub-folder per class):
#
#   dataset/
#   ├── Healthy/
#   │   ├── img001.jpg
#   │   └── ...
#   ├── Tomato Early Blight/
#   │   └── ...
#   ├── Tomato Late Blight/
#   │   └── ...
#   ├── Tomato Leaf Mold/
#   │   └── ...
#   ├── Potato Early Blight/
#   │   └── ...
#   └── Potato Late Blight/
#       └── ...
#
# After training the model is saved as: leafguard_model.keras
# =============================================================================

import os
import sys

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Check for dataset before importing TensorFlow (fast fail) ─────────────────
DATASET_DIR = "dataset"

if not os.path.isdir(DATASET_DIR):
    print("=" * 65)
    print("ERROR: Dataset folder not found!")
    print("=" * 65)
    print(f"\nExpected path: ./{DATASET_DIR}/")
    print("\nPlease create the following folder structure:")
    print("""
  dataset/
  +-- Healthy/
  |   +-- img001.jpg
  |   +-- ...
  +-- Tomato Early Blight/
  |   +-- ...
  +-- Tomato Late Blight/
  |   +-- ...
  +-- Tomato Leaf Mold/
  |   +-- ...
  +-- Potato Early Blight/
  |   +-- ...
  +-- Potato Late Blight/
      +-- ...
""")
    print("A good free dataset: PlantVillage on Kaggle")
    print("  https://www.kaggle.com/datasets/emmarex/plantdisease")
    print("\nEach sub-folder name must exactly match one of the class names")
    print("listed above.  Place at least ~50 images per class.")
    sys.exit(1)

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")

# ── Configuration ──────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)   # Input resolution for the CNN
BATCH_SIZE  = 32           # Number of images per training step
EPOCHS      = 30           # Maximum training epochs (early stopping may halt sooner)
VAL_SPLIT   = 0.2          # 20 % of data reserved for validation
SEED        = 42           # Random seed for reproducibility
MODEL_PATH  = "leafguard_model.keras"

CLASS_NAMES = [
    "Healthy",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Potato Early Blight",
    "Potato Late Blight",
]


# ── Verify class folders ───────────────────────────────────────────────────────
def verify_dataset(dataset_dir: str, class_names: list[str]) -> None:
    """Check that each expected class folder exists and has images."""
    found_classes = sorted(os.listdir(dataset_dir))
    print("\n── Dataset Verification ──────────────────────────────────────")
    errors = []
    for cls in class_names:
        cls_path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_path):
            errors.append(f"  MISSING folder: {cls_path}")
            continue
        images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        count = len(images)
        warning = " ⚠️  (< 50 images — consider adding more)" if count < 50 else ""
        print(f"  ✓  {cls:<30} {count:>5} images{warning}")

    # Warn about extra folders (not fatal)
    for folder in found_classes:
        if folder not in class_names:
            print(f"  ℹ️  Extra folder ignored: {folder}")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(e)
        print("\nPlease fix the above issues and re-run train_model.py")
        sys.exit(1)

    print("─" * 65)


verify_dataset(DATASET_DIR, CLASS_NAMES)


# ── Data loaders ──────────────────────────────────────────────────────────────
print("\n── Loading Dataset ───────────────────────────────────────────")

# Training split
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",           # labels from sub-folder names
    label_mode="categorical",    # one-hot encoded labels
    class_names=CLASS_NAMES,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="training",
)

# Validation split
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="validation",
)

print(f"Class names detected: {train_ds.class_names}")
print(f"Training batches:   {len(train_ds)}")
print(f"Validation batches: {len(val_ds)}")

# ── Performance optimisation ──────────────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE

# Normalise pixel values from [0, 255] to [0, 1] using a Rescaling layer inside
# the model so the saved model handles normalisation automatically.
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ── Data augmentation (applied only during training) ──────────────────────────
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1),
    ],
    name="data_augmentation",
)


# ── CNN model architecture ─────────────────────────────────────────────────────
def build_model(num_classes: int) -> tf.keras.Model:
    """
    Build a custom CNN suitable for plant disease classification.

    Architecture overview:
      Rescaling → Augmentation (train only) →
      3× [Conv2D → BatchNorm → ReLU → MaxPool] →
      GlobalAveragePool → Dense(256) → Dropout → Dense(num_classes, softmax)
    """
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image_input")

    # Normalise: [0, 255] → [0, 1]
    x = layers.Rescaling(1.0 / 255, name="rescaling")(inputs)

    # Data augmentation (active only during model.fit)
    x = data_augmentation(x)

    # ── Block 1 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # ── Block 2 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # ── Block 3 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # ── Block 4 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # ── Head ─────────────────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="LeafGuard_CNN")
    return model


# ── Build & compile ───────────────────────────────────────────────────────────
print("\n── Building Model ────────────────────────────────────────────")
model = build_model(num_classes=len(CLASS_NAMES))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    # Stop training when validation loss stops improving for 5 consecutive epochs
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    # Save the best model checkpoint during training
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    # Reduce learning rate when validation loss plateaus
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
]


# ── Training ───────────────────────────────────────────────────────────────────
print("\n── Training ──────────────────────────────────────────────────")
print(f"Max epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  Validation split: {VAL_SPLIT}")
print("─" * 65)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1,
)


# ── Final evaluation ───────────────────────────────────────────────────────────
print("\n── Final Evaluation on Validation Set ───────────────────────")
val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"  Validation Loss:     {val_loss:.4f}")
print(f"  Validation Accuracy: {val_acc * 100:.2f}%")


# ── Plot training curves ──────────────────────────────────────────────────────
def plot_history(history, save_path: str = "training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LeafGuard AI — Training History", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy", color="#16a34a")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy",   color="#4ade80", linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss", color="#dc2626")
    axes[1].plot(history.history["val_loss"], label="Val Loss",   color="#f87171", linestyle="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nTraining curves saved → {save_path}")


plot_history(history)

print("\n" + "=" * 65)
print(f"✅ Model saved → {MODEL_PATH}")
print("   Launch the app with:  streamlit run app.py")
print("=" * 65)
