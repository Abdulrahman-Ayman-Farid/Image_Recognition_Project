# Cats and Dogs Classification using VGG16

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Dataset](#dataset)
7. [Usage](#usage)
8. [Model Architecture](#model-architecture)
9. [Results](#results)
10. [Contributing](#contributing)
11. [License](#license)

## Overview

This project is a **binary image classification** system that distinguishes between images of **cats** and **dogs** using **transfer learning** with a pre-trained VGG16 convolutional neural network. The VGG16 model, originally trained on the ImageNet dataset, is used as a frozen feature extractor. A custom classification head — consisting of fully-connected layers — is trained on top to adapt the model to the cats-vs-dogs task.

The project is implemented as a single Jupyter notebook (`CatAndDogsClassifier.ipynb`) and was developed in Google Colab. It covers the complete machine learning pipeline: dataset download and extraction, data augmentation, model definition, training, evaluation, visualization of training curves, model saving, and single-image inference.

- **Task:** Binary classification (cat = 0, dog = 1)
- **Approach:** Transfer learning with a frozen VGG16 base
- **Dataset:** Google's `cats_and_dogs_filtered` dataset (a filtered subset of the Kaggle "Dogs vs. Cats" competition dataset)
- **Runtime environment:** Google Colab (Python 3.10)
- **Repository:** [https://github.com/Abdulrahman-Ayman-Farid/Image_Recognition_Project.git](https://github.com/Abdulrahman-Ayman-Farid/Image_Recognition_Project.git)

## Features

- **Transfer learning with VGG16** — leverages a pre-trained ImageNet model as a fixed feature extractor, reducing training time and data requirements.
- **Data augmentation pipeline** — applies random rotations, shifts, shears, zooms, and horizontal flips to the training set to improve generalization and reduce overfitting.
- **End-to-end ML workflow** — covers the full pipeline from dataset download and preprocessing through training, evaluation, visualization, model saving, and inference.
- **Training curve visualization** — plots training and validation accuracy and loss across all epochs using Matplotlib.
- **Single-image inference** — includes a helper function (`load_image`) that loads, preprocesses, and predicts on any new image.
- **Model export** — saves the trained model in HDF5 (`.h5`) format for reuse and deployment.
- **Colab-ready** — includes an "Open in Colab" badge and is designed to run end-to-end in Google Colab with GPU acceleration.

## Tech Stack

| Category | Technology | Version (Colab) |
|----------|-----------|-----------------|
| Language | Python | 3.10 |
| Deep Learning Framework | TensorFlow | 2.15.0 |
| Neural Network API | Keras | 2.15.0 |
| Numerical Computing | NumPy | 1.25.2 |
| Data Visualization | Matplotlib | 3.7.1 |
| Image Processing | Pillow (PIL) | 9.4.0 |
| File Download | wget | 3.2 |
| Runtime Environment | Google Colab | — |

## Project Structure

```
Image_Recognition_Project/
├── CatAndDogsClassifier.ipynb   # Main Jupyter notebook with the full ML pipeline
├── README.md                     # Project documentation (this file)
├── requirements.txt              # Python dependencies
└── docs/
    └── NOTEBOOK_REFERENCE.md     # Detailed cell-by-cell notebook reference
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Option 1: Google Colab (Recommended)

The notebook is designed to run in Google Colab, which provides a pre-configured Python environment with GPU support. No local installation is required.

1. Click the **"Open in Colab"** badge at the top of the notebook on GitHub.
2. Run all cells in order (`Runtime → Run all` or `Shift + Enter` on each cell).

The first cell installs the required packages:

```bash
pip install tensorflow matplotlib
```

### Option 2: Local Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Abdulrahman-Ayman-Farid/Image_Recognition_Project.git
cd Image_Recognition_Project
```

2. **Create and activate a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install the dependencies:**

```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook:**

```bash
jupyter notebook CatAndDogsClassifier.ipynb
```

> **Note:** If running locally, replace the `!wget` shell command in the dataset download cell with the Python `wget` package version (already included in the notebook's final cells), or use `urllib.request` as an alternative.

## Dataset

The project uses Google's **`cats_and_dogs_filtered`** dataset, a filtered subset of the Kaggle "Dogs vs. Cats" competition dataset.

- **Source URL:** `https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip`
- **Compressed size:** ~65 MB
- **Provider:** Google ML Education (mledu-datasets)
- **Total images:** 3,000 (2,000 training + 1,000 validation)

### Directory Structure

After extraction to `/tmp/cats_and_dogs_filtered/`:

```
/tmp/cats_and_dogs_filtered/
├── train/
│   ├── cats/    (1000 images)
│   └── dogs/    (1000 images)
└── validation/
    ├── cats/    (500 images)
    └── dogs/    (500 images)
```

### Data Augmentation

The training set is augmented in real-time using `ImageDataGenerator` to improve model generalization:

| Augmentation | Parameter | Value |
|-------------|-----------|-------|
| Rescaling | `rescale` | 1./255 |
| Rotation | `rotation_range` | 40 degrees |
| Width shift | `width_shift_range` | 0.2 |
| Height shift | `height_shift_range` | 0.2 |
| Shear | `shear_range` | 0.2 |
| Zoom | `zoom_range` | 0.2 |
| Horizontal flip | `horizontal_flip` | True |

> The validation set is only rescaled (no augmentation), following standard ML practice.

## Usage

### Training the Model

1. Open `CatAndDogsClassifier.ipynb` in Google Colab or Jupyter Notebook.
2. Run the cells in order:
   - **Cell 1:** Installs dependencies (`tensorflow`, `matplotlib`).
   - **Cells 2–3 (or 6–7):** Downloads and extracts the dataset, then visualizes sample images.
   - **Cell 4 (or 8):** Creates data generators with augmentation for training and rescaling for validation.
   - **Cell 5 (or 9):** Defines the VGG16 transfer learning model and compiles it.
   - **Cell 10:** Trains the model for 10 epochs (100 steps per epoch, 50 validation steps).
   - **Cell 11:** Evaluates the model on the validation set.
   - **Cell 12:** Saves the trained model to `cats_and_dogs_vgg16.h5`.
   - **Cell 13:** Plots training and validation accuracy/loss curves.
   - **Cell 14:** Demonstrates single-image inference.

### Running Inference on a New Image

```python
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Load and predict
img_path = 'path/to/your/image.jpg'
new_image = load_image(img_path)
prediction = model.predict(new_image)
print(f"Prediction: {'Dog' if prediction[0] > 0.5 else 'Cat'}")
```

### Loading the Saved Model

```python
from tensorflow.keras.models import load_model

model = load_model('cats_and_dogs_vgg16.h5')
```

## Model Architecture

The model uses **transfer learning**: a pre-trained VGG16 convolutional base (frozen) with a custom classification head trained on the cats-and-dogs dataset.

### Base Model: VGG16 (Pre-trained on ImageNet)

- **Input shape:** (224, 224, 3) — RGB images resized to 224×224 pixels
- **Weights:** ImageNet
- **include_top:** `False` — original fully-connected layers removed
- **All base layers frozen:** `layer.trainable = False` for every layer

### Custom Classification Head

| Layer | Type | Output Shape | Activation | Notes |
|-------|------|-------------|------------|-------|
| Flatten | Flatten | (batch, 25088) | — | Flattens VGG16's 7×7×512 output |
| Dense | Fully connected | (batch, 512) | ReLU | 512 neurons |
| Dropout | Regularization | (batch, 512) | — | Rate = 0.5 |
| Dense | Output | (batch, 1) | Sigmoid | Binary classification output |

### Architecture Diagram

```
Input -> Conv2D -> MaxPool -> Conv2D -> MaxPool -> Flatten -> Dense -> Dropout -> Dense(Output)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | RMSprop |
| Learning rate | 0.0001 |
| Loss function | binary_crossentropy |
| Metrics | accuracy |
| Epochs | 10 |
| Steps per epoch | 100 |
| Validation steps | 50 |
| Batch size | 20 |

> **Note:** Only the 3 new top layers (Flatten + Dense(512) + Dense(1)) are trainable. The ~13.5M VGG16 convolutional parameters are frozen and not updated during training.

## Results

The model was trained for 10 epochs on the cats and dogs filtered dataset.

| Metric | Value |
|--------|-------|
| Validation Accuracy | 93.00% |
| Validation Loss | 0.1900 |

Training and validation accuracy/loss curves are generated by the notebook (Cell 13) to visualize model performance over epochs. The model correctly predicts unseen images, as demonstrated in the inference cell (Cell 14).

> **Note:** Actual results may vary slightly depending on the runtime environment, random seed, and hardware. The values above were observed during the notebook's execution in Google Colab.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit them with clear, descriptive messages:
   ```bash
   git commit -m "Add: your descriptive commit message"
   ```
4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a Pull Request against the `main` branch.

### Ideas for Contribution

- Fine-tune the top VGG16 layers for improved accuracy
- Experiment with different architectures (ResNet, EfficientNet, MobileNet)
- Add a Flask/FastAPI web interface for interactive predictions
- Implement data augmentation with `tf.data` pipelines
- Add unit tests and CI/CD

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

> **Note:** A `LICENSE` file is not currently included in the repository. Please add one if you intend to distribute this project. The VGG16 model weights are derived from ImageNet and are subject to their respective licensing terms.
