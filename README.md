# 🌿 Cassava Leaf Disease Detection

A deep learning system for detecting cassava leaf diseases from images, built with TensorFlow and deployed as a live Streamlit web application. Three transfer learning models were trained and compared — EfficientNetB0, ResNet50, and MobileNetV2.

🔗 **Live App:** [Streamlit Cloud](https://cassava-disease-classification-with-deep-learning.streamlit.app)

🤗 **Models:** [Hugging Face — Ebaju-Ed4/cassava-disease-models](https://huggingface.co/Ebaju-Ed4/cassava-disease-models)

💻 **Code:** [GitHub — Edward-Ed4/Cassava-Disease-Classification-with-Deep-learning](https://github.com/Edward-Ed4/Cassava-Disease-Classification-with-Deep-learning)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Streamlit App](#streamlit-app)
- [Acknowledgements](#acknowledgements)

---

## Overview

Cassava (_Manihot esculenta_) is a critical food crop across Sub-Saharan Africa. Early detection of leaf diseases is essential for protecting yields. This project trains three convolutional neural network models to classify cassava leaf images into four categories and deploys them as an interactive web app that supports both image upload and live camera capture.

The models are hosted on Hugging Face Hub and downloaded automatically at app startup — no manual model placement required.

---

## Dataset

| Property          | Value                      |
| ----------------- | -------------------------- |
| Total images      | 9,726                      |
| Number of classes | 4                          |
| Split             | 80% train / 20% validation |
| Train images      | 7,780                      |
| Val images        | 1,946                      |

**Class distribution:**

| Class                             | Images |
| --------------------------------- | ------ |
| Cassava\_\_\_brown_streak_disease | 2,448  |
| Cassava\_\_\_mosaic_disease       | 2,448  |
| Cassava\_\_\_healthy              | 2,444  |
| Cassava\_\_\_green_mottle         | 2,386  |

The dataset is nearly perfectly balanced across all four classes, requiring minimal class weighting during training. Images were manually reviewed and cleaned before training.

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

| Model          | Phase 1 Val Accuracy | Phase 2 Val Accuracy | Val Loss  | Training Speed |
| -------------- | -------------------- | -------------------- | --------- | -------------- |
| EfficientNetB0 | 68.55%               | 74.67%               | 0.778     | Medium         |
| MobileNetV2    | 65.26%               | 77.24%               | 0.745     | Fastest        |
| **ResNet50**   | **69.37%**           | **81.50%**           | **0.624** | Slowest        |

> All models were trained on CPU. GPU training is expected to yield higher accuracy with more epochs.

**ResNet50 is the best performing model**, achieving 81.5% validation accuracy and the lowest validation loss of 0.624.

---

## Project Structure

```
cassava-leaf-disease/
│
├── Cassava_EfficientNetB0.ipynb       # EfficientNetB0 training notebook
├── Cassava_ResNet50.ipynb             # ResNet50 training notebook
├── Cassava_MobileNetV2.ipynb          # MobileNetV2 training notebook
│
├── cassava_app.py                     # Streamlit web application
├── requirements.txt                   # Python dependencies for deployment
├── .gitignore                         # Excludes models, venv, cache
│
├── training_history_b0.png            # EfficientNetB0 training curves
├── training_history_resnet50.png      # ResNet50 training curves
├── training_history_mobilenet.png     # MobileNetV2 training curves
│
├── cassava_b0_phase1_log.csv          # EfficientNetB0 epoch log
├── cassava_resnet50_phase1_log.csv    # ResNet50 epoch log
├── cassava_mobilenetv2_log.csv        # MobileNetV2 epoch log
│
└── README.md

# Model files are hosted on Hugging Face (too large for GitHub):
# huggingface.co/Ebaju-Ed4/cassava-disease-models
#   ├── cassava_efficientnetb0_final.keras   (51 MB)
#   ├── cassava_resnet50_final.keras         (280 MB)
#   └── cassava_mobilenetv2_final.h5         (29 MB)
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Edward-Ed4/Cassava-Disease-Classification-with-Deep-learning.git
cd Cassava-Disease-Classification-with-Deep-learning
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
pip install -r requirements.txt
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

## Deployment

### Architecture

```
GitHub (code)  →  Streamlit Cloud (app)
                        ↓
              Hugging Face Hub (models)
```

The app downloads models from Hugging Face at startup using `hf_hub_download`. Models are cached after the first download so subsequent loads are instant.

### Model hosting — Hugging Face

Models are too large for GitHub (ResNet50 is 280 MB). They are hosted at:

```
huggingface.co/Ebaju-Ed4/cassava-disease-models
```

The app downloads them automatically — no manual setup needed.

### App hosting — Streamlit Cloud

The app is deployed at [share.streamlit.io](https://share.streamlit.io) connected to this GitHub repository.

**Deployment settings:**

- Python version: `3.11`
- Main file: `cassava_app.py`
- Dependencies: `requirements.txt`

### Running locally

```bash
streamlit run cassava_app.py
```

Models will be downloaded from Hugging Face on first run and cached locally.

---

## Streamlit App

### Features

- **Model selector** — switch between EfficientNetB0, ResNet50, and MobileNetV2 from the sidebar
- **Camera capture** — take a live photo directly from your device camera; camera closes automatically after capture
- **Image upload** — upload one or more images from your device
- **Disease prediction** — predicted class with confidence score
- **Severity rating** — colour-coded severity level per disease
- **Treatment advice** — actionable treatment recommendations for each disease
- **Probability chart** — horizontal bar chart showing model confidence across all four classes

### Supported classes

| Class                        | Severity  | Description                                                                       |
| ---------------------------- | --------- | --------------------------------------------------------------------------------- |
| Healthy                      | None      | No disease detected                                                               |
| Cassava Green Mottle         | Moderate  | Caused by CGMV, presents as green mottling on leaves                              |
| Cassava Mosaic Disease       | High      | Caused by CMV via whiteflies, mosaic patterns on leaves                           |
| Cassava Brown Streak Disease | Very High | Caused by CBSV, destroys tubers — most destructive cassava disease in East Africa |

---

## Acknowledgements

- Dataset curated and cleaned manually for balanced class distribution
- Transfer learning architectures sourced from `tensorflow.keras.applications`
- Model training approach inspired by collaborative work with peers on a parallel maize leaf disease classification project (EfficientNetB0, ResNet50, MobileNetV2)
- Models hosted on [Hugging Face Hub](https://huggingface.co)
- App deployed on [Streamlit Cloud](https://streamlit.io/cloud)

---

## License

This project is for academic and research purposes.
