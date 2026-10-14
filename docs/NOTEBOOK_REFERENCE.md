# Notebook Reference — Cats and Dogs Classifier

This document is a comprehensive, cell-by-cell reference for `CatAndDogsClassifier.ipynb`, the sole project artifact of the **Image Recognition Project**. All content is derived from a verified cell-by-cell analysis of the notebook.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment & Dependencies](#2-environment--dependencies)
3. [Notebook Architecture](#3-notebook-architecture)
4. [Model Architecture](#4-model-architecture)
5. [Data Pipeline](#5-data-pipeline)
6. [Training Configuration](#6-training-configuration)
7. [Evaluation & Results](#7-evaluation--results)

---

## 1. Project Overview

This project is a **binary image classification** system that distinguishes between images of **cats** and **dogs**.

- **Task:** Binary classification (cat = 0, dog = 1).
- **Context:** CodeAlpha internship project (Image Recognition task).
- **Approach:** **Transfer learning** — a pre-trained VGG16 convolutional neural network (trained on ImageNet) is used as a fixed feature extractor, with new fully-connected classification layers trained on top.
- **Dataset:** Google's `cats_and_dogs_filtered` dataset (a filtered subset of the Dogs vs. Cats Kaggle competition dataset).
- **Runtime environment:** Google Colab (Python 3.10).
- **Repository:** `https://github.com/Abdulrahman-Ayman-Farid/Image_Recognition_Project.git`
- **Notebook:** `CatAndDogsClassifier.ipynb` — the sole project artifact containing all code, outputs, and inline images.
- **Final validation accuracy:** 93.00%.

---

## 2. Environment & Dependencies

The notebook runs in Google Colab (Python 3.10). The following versions were observed in the Colab environment:

| Package      | Version |
|--------------|---------|
| TensorFlow   | 2.15.0  |
| Keras        | 2.15.0  |
| NumPy        | 1.25.2  |
| Matplotlib   | 3.7.1   |
| Pillow       | 9.4.0   |

### Python Imports

The following Python imports are found in the notebook code cells:

| Import | Role |
|--------|------|
| `os` | Operating system interface — used for path construction (`os.path.join`), directory listing (`os.listdir`), and directory creation (`os.makedirs`). |
| `zipfile` | Standard library module for reading and extracting ZIP archives — used to extract the downloaded dataset. |
| `wget` | Third-party Python package for downloading files from URLs — used to download the dataset ZIP from Google Cloud Storage (in the final version of the download cell). |
| `matplotlib.pyplot` | Plotting library — used to create sample image grids and training/evaluation curves (accuracy and loss plots). |
| `matplotlib.image` | Image reading submodule of Matplotlib — used via `mpimg.imread()` to load sample images for visualization. |
| `tensorflow.keras.preprocessing.image.ImageDataGenerator` | Keras data augmentation and batch generation utility — generates augmented image batches from directory structure for training and rescaled batches for validation. |
| `tensorflow.keras.applications.vgg16.VGG16` | Pre-trained VGG16 model from Keras Applications — loaded with ImageNet weights as the frozen convolutional base for transfer learning. |
| `tensorflow.keras.layers` | Keras layers API — used to build the custom classification head: `Flatten`, `Dense`, `Dropout`. |
| `tensorflow.keras.models` | Keras model API — used to construct the final `Model` by connecting the VGG16 base input to the custom head output. |
| `tensorflow.keras.optimizers.RMSprop` | Keras optimizer — used to compile the model with learning_rate=0.0001. |
| `numpy` | Numerical computing library — used for array manipulation during single-image inference (`np.expand_dims` to add batch dimension). |
| `tensorflow.keras.preprocessing.image` (as `image`) | Keras image preprocessing utilities — used via `image.load_img()` and `image.img_to_array()` to load and convert a single image for inference. |
| `tensorflow.keras.applications.vgg16.preprocess_input` | VGG16-specific preprocessing function — applies mean RGB subtraction; used during single-image inference. |

### Shell Commands

In addition to the Python imports above, the notebook uses the following shell commands:

- `!pip install tensorflow matplotlib` — used in Cell 1 to install TensorFlow and Matplotlib.
- `!wget` (shell command) — used in the earlier version of the download cell (Cell 2) to fetch the dataset ZIP.

---

## 3. Notebook Architecture

The notebook contains **15 cells** total: **1 markdown cell** and **14 code cells**.

> **Note on dual passes:** The notebook contains two "passes" of the same workflow. Cells 1–5 are an earlier version (no execution counts), and Cells 6–14 are the re-executed, final authoritative version (with execution counts 7–23 and full outputs).

### Cell-by-Cell Breakdown

| Cell Number | Cell Type | Description | Key Functions/Classes Used |
|-------------|-----------|-------------|----------------------------|
| 0 | Markdown | An HTML anchor linking to Google Colab with a badge image. Provides an "Open in Colab" button when viewed on GitHub. | N/A |
| 1 | Code (exec=None) | Installs TensorFlow and Matplotlib via `!pip install tensorflow matplotlib`. | `pip install` |
| 2 | Code (exec=None) | **EARLIER VERSION** — Downloads the dataset using shell `!wget` from `https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip`, extracts it to `/tmp` using `zipfile.ZipFile`, and defines directory paths for train/validation splits. | `os.path.join`, `zipfile.ZipFile`, `!wget` |
| 3 | Code (exec=None) | **EARLIER VERSION** — Visualizes 8 cat and 8 dog training images in a 4×4 grid using Matplotlib. | `plt.subplot`, `plt.imshow`, `mpimg.imread`, `os.listdir` |
| 4 | Code (exec=None) | **EARLIER VERSION** — Creates `ImageDataGenerator` for training (with augmentation) and validation (rescale only), sets up `flow_from_directory` generators with batch_size=20, binary class mode, target_size=(224,224). | `ImageDataGenerator`, `flow_from_directory` |
| 5 | Code (exec=None) | **EARLIER VERSION** — Defines the transfer learning model: loads VGG16 (input_shape=(224,224,3), include_top=False, weights='imagenet'), freezes all base layers, appends Flatten→Dense(512,ReLU)→Dropout(0.5)→Dense(1,sigmoid), compiles with RMSprop(lr=0.0001), binary_crossentropy, acc metric. | `VGG16`, `layers.Flatten`, `layers.Dense`, `layers.Dropout`, `models.Model`, `RMSprop`, `model.compile` |
| 6 | Code (exec=7) | **FINAL VERSION** — Re-downloads and extracts the dataset using the Python `wget` package instead of shell `!wget`. Includes a directory existence check with `os.makedirs`. | `wget.download`, `zipfile.ZipFile`, `os.path.exists`, `os.makedirs` |
| 7 | Code (exec=8) | **FINAL VERSION** — Re-defines all directory paths and visualizes 16 sample training images (8 cats + 8 dogs) in a 4×4 grid. | `plt.subplot`, `plt.imshow`, `mpimg.imread`, `os.listdir`, `os.path.join` |
| 8 | Code (exec=9) | **FINAL VERSION** — Identical augmentation pipeline to Cell 4. Training generator applies augmentation; validation generator only rescales. Both use batch_size=20, binary class mode, 224×224 target size. Output: Found 2000 training images, 1000 validation images. | `ImageDataGenerator`, `flow_from_directory` |
| 9 | Code (exec=10) | **FINAL VERSION** — Identical model definition to Cell 5. VGG16 weights download triggered (58.9 MB). | `VGG16`, `layers.Flatten`, `layers.Dense`, `layers.Dropout`, `models.Model`, `RMSprop`, `model.compile` |
| 10 | Code (exec=11) | Trains the model for 10 epochs with 100 steps per epoch and 50 validation steps, verbose=2. Training history stored in `vgghist`. | `model.fit` |
| 11 | Code (exec=12) | Evaluates the trained model on the full validation set (50 steps). Reports validation loss=0.1900 and accuracy=93.00%. | `model.evaluate` |
| 12 | Code (exec=13) | Saves the trained model to `cats_and_dogs_vgg16.h5` (HDF5 format). Keras issues a UserWarning recommending the native `.keras` format. | `model.save` |
| 13 | Code (exec=14) | Plots training curves (accuracy and loss for both training and validation) across all 10 epochs using Matplotlib. | `vgghist.history`, `plt.plot`, `plt.legend`, `plt.figure`, `plt.show` |
| 14 | Code (exec=23) | Defines a `load_image()` helper function for single-image inference. Loads, resizes to (224,224), converts to array, expands dims, and applies `preprocess_input`. Demonstrates prediction on `/content/cat3.jpg` — correctly predicts "Cat". | `image.load_img`, `image.img_to_array`, `np.expand_dims`, `preprocess_input`, `model.predict` |

---

## 4. Model Architecture

The model uses **transfer learning**: a pre-trained VGG16 convolutional base (frozen) with a custom classification head trained on the cats-and-dogs dataset.

### Base Model: VGG16 (Pre-trained on ImageNet)

- **Input shape:** (224, 224, 3) — RGB images resized to 224×224 pixels.
- **Weights:** ImageNet.
- **include_top:** `False` — the original VGG16 fully-connected classification layers are removed.
- **All base layers frozen:** `layer.trainable = False` for every layer in `base_model.layers`.
- **Weights file downloaded:** `vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5` (58.9 MB).

### Custom Classification Head (appended on top of VGG16)

| Layer | Type | Output Shape | Activation | Notes |
|-------|------|--------------|------------|-------|
| base_model.output → Flatten | Flatten | (batch, 25088) | — | Flattens VGG16's 7×7×512 output |
| Dense | Fully connected | (batch, 512) | ReLU | 512 neurons |
| Dropout | Regularization | (batch, 512) | — | Rate = 0.5 |
| Dense | Output | (batch, 1) | Sigmoid | Binary classification output |

### Architecture Diagram (ASCII)

```
VGG16 base (frozen, ImageNet weights)
    ↓
Flatten
    ↓
Dense(512, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(1, Sigmoid)
```

> **Note:** Total trainable parameters are only the 3 new top layers (Flatten + Dense(512) + Dense(1)). The ~13.5M VGG16 convolutional parameters are frozen and not updated during training.

---

## 5. Data Pipeline

Training and validation data are loaded using `ImageDataGenerator` from `tensorflow.keras.preprocessing.image`.

- **Training generator:** `train_datagen.flow_from_directory(train_dir, batch_size=20, class_mode='binary', target_size=(224, 224))`
- **Validation generator:** `test_datagen.flow_from_directory(validation_dir, batch_size=20, class_mode='binary', target_size=(224, 224))`
- **Image preprocessing:** pixel rescaling by `1./255` (normalizes to [0,1]).
- **Batch size:** 20.
- **Target image size:** (224, 224).
- **Class mode:** binary (cats=0, dogs=1).

### Data Augmentation (training only)

| Augmentation | Parameter | Value | Description |
|--------------|-----------|-------|-------------|
| Rescaling | rescale | 1./255. | Normalizes pixel values to [0, 1] |
| Rotation | rotation_range | 40 | Random rotation up to ±40 degrees |
| Width shift | width_shift_range | 0.2 | Random horizontal shift up to 20% of width |
| Height shift | height_shift_range | 0.2 | Random vertical shift up to 20% of height |
| Shear | shear_range | 0.2 | Random shear transformation up to 20% |
| Zoom | zoom_range | 0.2 | Random zoom up to 20% |
| Horizontal flip | horizontal_flip | True | Randomly mirrors images horizontally |

> **Note:** Validation set only has `rescale=1.0/255.` (no augmentation) — standard practice.

### Dataset Details

- **Source URL:** `https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip`
- **Size:** 65.43 MB compressed.
- **Provider:** Google ML Education (mledu-datasets).
- **Origin:** Filtered subset of the Kaggle "Dogs vs. Cats" competition dataset.
- **Directory structure after extraction to `/tmp/cats_and_dogs_filtered/`:**

```
/tmp/cats_and_dogs_filtered/
├── train/
│   ├── cats/    (1000 images)
│   └── dogs/    (1000 images)
└── validation/
    ├── cats/    (500 images)
    └── dogs/    (500 images)
```

- **Total:** 2,000 training images, 1,000 validation images, 3,000 total.

---

## 6. Training Configuration

- **Optimizer:** RMSprop (learning_rate=0.0001)
- **Loss function:** binary_crossentropy
- **Metrics:** accuracy (`acc`)
- **Number of epochs:** 10
- **Steps per epoch:** 100 (2,000 images / batch_size 20)
- **Validation steps:** 50 (1,000 images / batch_size 20)
- **Batch size:** 20
- **Verbose:** 2 (one line per epoch)
- **Training history variable:** `vgghist`
- **Callbacks:** NONE — no EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, or TensorBoard callbacks are used.
- **Learning rate scheduling:** NONE — fixed learning rate of 0.0001 throughout training.

### Epoch-by-Epoch Training Progression

| Epoch | Training Loss | Training Acc | Validation Loss | Validation Acc | Time (s) |
|-------|---------------|--------------|-----------------|----------------|----------|
| 1 | 0.6810 | 0.6765 | 0.3371 | 0.8650 | 1677 |
| 2 | 0.4581 | 0.7890 | 0.2783 | 0.8800 | 1668 |
| 3 | 0.4241 | 0.8050 | 0.2374 | 0.9020 | 1661 |
| 4 | 0.3843 | 0.8215 | 0.2263 | 0.9130 | 1657 |
| 5 | 0.3785 | 0.8245 | 0.2076 | 0.9160 | 1663 |
| 6 | 0.3486 | 0.8380 | 0.6387 | 0.7450 | 1664 |
| 7 | 0.3408 | 0.8575 | 0.1990 | 0.9260 | 1633 |
| 8 | 0.3213 | 0.8610 | 0.2056 | 0.9190 | 1663 |
| 9 | 0.3239 | 0.8545 | 0.2133 | 0.9160 | 1665 |
| 10 | 0.3006 | 0.8625 | 0.1900 | 0.9300 | 1668 |

### Key Observations

- **Best validation accuracy:** 93.00% (epoch 10).
- **Best validation loss:** 0.1900 (epoch 10).
- **Training accuracy:** 67.65% → 86.25% (steady improvement).
- **Epoch 6 anomaly:** Validation loss spiked to 0.6387 and accuracy dropped to 74.50%, then recovered.
- **No overfitting evident** — validation accuracy consistently exceeds training accuracy (because training uses augmented/harder images while validation uses clean images).
- **Total training time:** ~16,649 seconds (~4 hours 37 minutes), suggesting CPU or slow GPU execution in Colab.

---

## 7. Evaluation & Results

### Final Evaluation (Cell 11)

```python
model.evaluate(validation_generator, steps=50)
```

- **Validation Loss:** 0.18995824456214905
- **Validation Accuracy:** 0.9300000071525574 (93.00%)
- **Evaluation steps:** 50 (1,000 validation images)
- **Evaluation time:** 529 seconds (~9 minutes)

| Metric | Value |
|--------|-------|
| Validation Accuracy | 93.00% |
| Validation Loss | 0.1900 |
| Evaluation steps | 50 |
| Evaluation time | ~529 seconds |

### Training Curves (Cell 13)

Two plots generated using Matplotlib:

1. **Training vs. Validation Accuracy** — red=training, blue=validation. Shows validation accuracy consistently above training accuracy, converging to ~93%.
2. **Training vs. Validation Loss** — red=training, blue=validation. Shows both losses decreasing, with the epoch 6 spike visible in validation loss.

### Model Saving (Cell 12)

```python
model.save('cats_and_dogs_vgg16.h5')
```

- Saves in HDF5 format.
- Keras issues a UserWarning that `.h5` is legacy and recommends the native `.keras` format.
- The file is saved in the Colab runtime, NOT committed to the repository.
- No model loading code is present in the notebook.

### Single Image Inference (Cell 14)

The `load_image()` helper function performs single-image inference:

```python
def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)
```

- **Loading:** `image.load_img(img_path, target_size=(224,224))`
- **Conversion:** `image.img_to_array()`
- **Batch dimension:** `np.expand_dims(img_array, axis=0)`
- **Preprocessing:** `preprocess_input()`
- **Prediction:** `model.predict(new_image)` — if `prediction[0] > 0.5` → "Dog", else → "Cat".
- **Test image:** `/content/cat3.jpg`
- **Result:** Prediction: Cat (correct)
- **Inference time:** 487ms for a single image.

### Known Issues

- **Preprocessing inconsistency:** During training (Cell 8), images are only rescaled (`1./255`). During inference (Cell 14), `vgg16.preprocess_input` is used, which applies mean RGB subtraction (a different normalization). This could affect prediction quality.
- **No confusion matrix or classification report** is generated in the notebook.
- **No separate test set exists** — only train/validation splits.
- **No callbacks** (EarlyStopping, ModelCheckpoint) are used.
- **Hardcoded Colab paths** (`/content/cat3.jpg`) won't work locally without modification.
- **Legacy `.h5` model format** (Keras recommends `.keras`).
