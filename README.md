# 🌿 Cassava Leaf Disease Detection

A deep learning system for detecting cassava leaf diseases from images, built with TensorFlow and deployed via a Streamlit web application. Three transfer learning models were trained and compared — EfficientNetB0, ResNet50, and MobileNetV2.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Streamlit App](#streamlit-app)
- [Acknowledgements](#acknowledgements)

---

## Overview

Cassava (*Manihot esculenta*) is a critical food crop across Sub-Saharan Africa. Early detection of leaf diseases is essential for protecting yields. This project trains three convolutional neural network models to classify cassava leaf images into four categories and deploys them as an interactive web app that supports both image upload and live camera capture.

---

## Dataset

| Property | Value |
|----------|-------|
| Total images | 9,726 |
| Number of classes | 4 |
| Split | 80% train / 20% validation |
| Train images | 7,780 |
| Val images | 1,946 |

**Class distribution:**

| Class | Images |
|-------|--------|
| Cassava___brown_streak_disease | 2,448 |
| Cassava___mosaic_disease | 2,448 |
| Cassava___healthy | 2,444 |
| Cassava___green_mottle | 2,386 |

The dataset is nearly perfectly balanced across all four classes, requiring minimal class weighting during training.

---

## Models

All three models use a two-phase transfer learning strategy:

**Phase 1 — Head training (base frozen)**
- Only the custom classification head is trained
- Allows the head to learn task-specific features before fine-tuning

**Phase 2 — Fine-tuning (top layers unfrozen)**
- Top 30–50% of the base model is unfrozen
- BatchNormalization layers kept frozen throughout to preserve pretrained statistics
- Lower learning rate to prevent catastrophic forgetting

### EfficientNetB0
- Base: EfficientNetB0 pretrained on ImageNet
- Head: GlobalAveragePooling → BatchNorm → Dense(512) → Dropout(0.4) → Dense(256) → Dropout(0.3) → Dense(4, softmax)
- Phase 1 LR: `3e-4` | Phase 2 LR: `5e-5`
- Preprocessing: `efficientnet.preprocess_input` (scales to `[-1, 1]`)

### ResNet50
- Base: ResNet50 pretrained on ImageNet
- Head: GlobalAveragePooling → BatchNorm → Dense(512) → Dropout(0.4) → Dense(256) → Dropout(0.3) → Dense(4, softmax)
- Phase 1 LR: `3e-4` | Phase 2 LR: `1e-4`
- Preprocessing: `resnet50.preprocess_input`
- BatchNorm layers explicitly frozen in Phase 2

### MobileNetV2
- Base: MobileNetV2 (alpha=1.0) pretrained on ImageNet
- Head: GlobalAveragePooling → BatchNorm → Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Dense(4, softmax)
- Phase 1 LR: `1e-3` | Phase 2 LR: `1e-4`
- Preprocessing: `mobilenet_v2.preprocess_input`
- Last 50 layers unfrozen in Phase 2

---

## Results

| Model | Phase 1 Val Accuracy | Phase 2 Val Accuracy | Val Loss | Training Speed |
|-------|---------------------|---------------------|----------|----------------|
| EfficientNetB0 | 68.55% | 74.67% | 0.778 | Medium |
| MobileNetV2 | 65.26% | 77.24% | 0.745 | Fastest |
| **ResNet50** | **69.37%** | **81.50%** | **0.624** | Slowest |

**ResNet50 is the best performing model**, achieving 81.5% validation accuracy and the lowest validation loss of 0.624. All models were trained on CPU — GPU training is expected to yield higher accuracy with more epochs.

---

## Project Structure

```
cassava-leaf-disease/
│
├── Cassava_EfficientNetB0.ipynb      # EfficientNetB0 training notebook
├── Cassava_ResNet50.ipynb            # ResNet50 training notebook
├── Cassava_MobileNetV2.ipynb         # MobileNetV2 training notebook
│
├── cassava_app.py                    # Streamlit web application
│
├── cassava_efficientnetb0_final.keras  # Trained EfficientNetB0 model
├── cassava_resnet50_final.keras        # Trained ResNet50 model
├── cassava_mobilenetv2_final.h5        # Trained MobileNetV2 model
│
├── dataset/
│   └── (your cassava leaf images organised by class)
│
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/cassava-leaf-disease.git
cd cassava-leaf-disease
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install tensorflow streamlit pillow plotly numpy pandas scikit-learn matplotlib seaborn
```

---

## Usage

### Training the models

Open any of the three notebooks in Jupyter and run cells from top to bottom:

```bash
jupyter notebook Cassava_ResNet50.ipynb
```

Before running, update the `DATA_DIR` path in the Config cell to point to your dataset:

```python
DATA_DIR = r'path/to/your/cassava/dataset'
```

The dataset folder should be structured as:

```
dataset/
├── Cassava___healthy/
├── Cassava___mosaic_disease/
├── Cassava___green_mottle/
└── Cassava___brown_streak_disease/
```

The notebooks handle the train/val split automatically (80/20 stratified).

---

## Streamlit App

### Running the app

Place your trained model files in the same folder as `cassava_app.py`, then run:

```bash
streamlit run cassava_app.py
```

### App features

- **Model selector** — switch between EfficientNetB0, ResNet50, and MobileNetV2 from the sidebar
- **Camera capture** — take a live photo directly from your device camera
- **Image upload** — upload one or more images from your device
- **Disease prediction** — predicted class with confidence score
- **Severity rating** — colour-coded severity level (Moderate / High / Very High)
- **Treatment advice** — actionable treatment recommendations for each disease
- **Probability chart** — horizontal bar chart showing confidence across all four classes

### Supported classes

| Class | Severity | Description |
|-------|----------|-------------|
| Healthy | None | No disease detected |
| Cassava Green Mottle | Moderate | Caused by CGMV, green mottling on leaves |
| Cassava Mosaic Disease | High | Caused by CMV via whiteflies, mosaic patterns |
| Cassava Brown Streak Disease | Very High | Caused by CBSV, destroys tubers |

---

## Acknowledgements

- Dataset curated and cleaned manually for balanced class distribution
- Transfer learning architectures sourced from `tensorflow.keras.applications`
- Model training approach inspired by collaborative work with peers on a parallel maize leaf disease classification project
- Streamlit used for rapid deployment of the inference interface

---

## License

This project is for academic and research purposes.
