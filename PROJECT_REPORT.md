# Project Report — Cassava Leaf Disease Detection with Deep Learning & AWS Cloud

**Author:** Weyanga Richard Shadrack
**Student Number:** 2400720749

---

## 1. Project Overview

A deep learning system that detects cassava leaf diseases from images. Three convolutional neural network models were trained using transfer learning, compared for performance, and deployed as a live web application accessible from any device globally. The final architecture uses AWS S3 for cloud model storage and Streamlit Cloud for application hosting.

---

## 2. Machine Learning — Model Training

Three models were trained and compared, all using a two-phase transfer learning strategy.

### Dataset

| Property          | Value                      |
| ----------------- | -------------------------- |
| Total images      | 9,726                      |
| Number of classes | 4                          |
| Split             | 80% train / 20% validation |
| Train images      | 7,780                      |
| Val images        | 1,946                      |

The dataset is nearly perfectly balanced at approximately 2,400 images per class. Images were manually reviewed and cleaned before training.

### Classes

| Class                        | Severity  |
| ---------------------------- | --------- |
| Cassava Brown Streak Disease | Very High |
| Cassava Mosaic Disease       | High      |
| Cassava Green Mottle         | Moderate  |
| Healthy                      | None      |

### Training Strategy

All three models used a two-phase transfer learning approach:

**Phase 1 — Head training (base frozen)**

- Only the custom classification head is trained
- Allows the head to learn task-specific features before fine-tuning

**Phase 2 — Fine-tuning (top layers unfrozen)**

- Top 30–50% of the base model is unfrozen
- BatchNormalization layers kept frozen throughout to preserve pretrained statistics
- Lower learning rate applied to prevent catastrophic forgetting

### Model Architectures

**EfficientNetB0**

- Base: EfficientNetB0 pretrained on ImageNet
- Head: GlobalAveragePooling → BatchNorm → Dense(512) → Dropout(0.4) → Dense(256) → Dropout(0.3) → Dense(4, softmax)
- Phase 1 LR: `3e-4` | Phase 2 LR: `5e-5`

**ResNet50**

- Base: ResNet50 pretrained on ImageNet
- Head: GlobalAveragePooling → BatchNorm → Dense(512) → Dropout(0.4) → Dense(256) → Dropout(0.3) → Dense(4, softmax)
- Phase 1 LR: `3e-4` | Phase 2 LR: `1e-4`
- BatchNorm layers explicitly frozen in Phase 2

**MobileNetV2**

- Base: MobileNetV2 (alpha=1.0) pretrained on ImageNet
- Head: GlobalAveragePooling → BatchNorm → Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Dense(4, softmax)
- Phase 1 LR: `1e-3` | Phase 2 LR: `1e-4`
- Last 50 layers unfrozen in Phase 2

### Results

| Model          | Phase 1 Val Accuracy | Phase 2 Val Accuracy | Val Loss  | Training Speed |
| -------------- | -------------------- | -------------------- | --------- | -------------- |
| EfficientNetB0 | 68.55%               | 74.67%               | 0.778     | Medium         |
| MobileNetV2    | 65.26%               | 77.24%               | 0.745     | Fastest        |
| **ResNet50**   | **69.37%**           | **81.50%**           | **0.624** | Slowest        |

> All models were trained on CPU. GPU training is expected to yield higher accuracy with more epochs.

**ResNet50 is the best performing model**, achieving 81.5% validation accuracy and the lowest validation loss of 0.624.

---

## 3. Cloud Architecture

### Architecture Diagram

```
Developer pushes code
        ↓
GitHub (cloud source control + CI/CD trigger)
        ↓  auto-deploy on every push to main
Streamlit Cloud (cloud application hosting — free tier)
        ↓  downloads models at startup via boto3
AWS S3 — eu-north-1 Stockholm (cloud object storage)
   ├── cassava_efficientnetb0_final.keras  (51 MB)
   ├── cassava_resnet50_final.keras        (280 MB)
   └── cassava_mobilenetv2_final.h5        (29 MB)
```

### AWS Services Used

**Amazon S3 (Simple Storage Service)**

- Bucket name: `cassava-disease-models`
- Region: `eu-north-1` (Europe — Stockholm)
- Stores all three trained model files totalling approximately 360 MB
- Models are downloaded by the application at startup using the `boto3` Python SDK
- S3 was chosen because GitHub has a 100 MB file size limit, making it unsuitable for storing large ML model files

**AWS IAM (Identity and Access Management)**

- Created a dedicated IAM user: `cassava-app-user`
- Attached policy: `AmazonS3ReadOnlyAccess` (least-privilege principle)
- Generated programmatic access keys for the Streamlit application to authenticate with S3
- AWS credentials stored securely as Streamlit Cloud secrets — never hardcoded in source code or committed to GitHub

### Key Cloud Design Decisions

- **Separation of concerns** — code, models, and compute are hosted on three separate platforms
- **Least-privilege IAM** — the app user has read-only S3 access, no other AWS permissions
- **Secrets management** — AWS credentials injected as environment variables via Streamlit secrets, not stored in the codebase
- **Serverless deployment** — no server management required; Streamlit Cloud handles all infrastructure
- **Continuous deployment** — every git push to the main branch triggers an automatic redeploy

---

## 4. Deployment Pipeline

| Step               | Tool                     | Purpose                                      |
| ------------------ | ------------------------ | -------------------------------------------- |
| Source control     | GitHub                   | Stores all code, notebooks, charts, and logs |
| Model storage      | AWS S3                   | Hosts trained model files in the cloud       |
| App hosting        | Streamlit Cloud          | Runs and serves the web application          |
| Secrets management | Streamlit Secrets        | Stores AWS credentials securely              |
| CI/CD              | GitHub → Streamlit Cloud | Auto-redeploy on every git push              |

---

## 5. Application Features

The Streamlit web application provides the following functionality:

- **Model selector** — switch between EfficientNetB0, ResNet50, and MobileNetV2 from the sidebar
- **Camera capture** — take a live photo directly from the device camera
- **Image upload** — upload one or more images from the device
- **Disease prediction** — predicted class with confidence score
- **Severity rating** — colour-coded severity level per disease
- **Treatment advice** — actionable treatment recommendations for each detected disease
- **Probability chart** — horizontal bar chart showing model confidence across all four classes

---

## 6. Project Links

| Resource                    | Link                                                                                 |
| --------------------------- | ------------------------------------------------------------------------------------ |
| Live App                    | https://cassava-disease-classification-with-deep-learning-t7jyapv6cayz.streamlit.app |
| GitHub Repository           | https://github.com/Edward-Ed4/Cassava-Disease-Classification-with-Deep-learning      |
| Hugging Face (model backup) | https://huggingface.co/Ebaju-Ed4/cassava-disease-models                              |
| AWS S3 Bucket               | `cassava-disease-models` — eu-north-1 (Stockholm)                                    |

---

## 7. Repository Structure

```
cassava-disease-classification/
│
├── cassava_app.py                     # Streamlit web application (loads models from AWS S3)
├── requirements.txt                   # Python dependencies
├── runtime.txt                        # Python version pin (3.11) for Streamlit Cloud
├── .gitignore                         # Excludes model files, venv, cache
├── README.md                          # Project documentation
├── PROJECT_REPORT.md                  # This report
│
├── Cassava_ResNet50.ipynb             # ResNet50 training notebook
├── Cassava_EfficientNetB0.ipynb       # EfficientNetB0 training notebook
├── Cassava_MobileNetV2.ipynb          # MobileNetV2 training notebook
│
├── confusion_matrix_resnet50.png      # ResNet50 confusion matrix
├── confusion_matrix_b0.png            # EfficientNetB0 confusion matrix
├── confusion_matrix_mobilenetv2.png   # MobileNetV2 confusion matrix
│
├── training_history_resnet50.png      # ResNet50 training curves
├── training_history_b0.png            # EfficientNetB0 training curves
├── training_history_mobilenet.png     # MobileNetV2 training curves
│
├── cassava_resnet50_phase1_log.csv    # ResNet50 epoch-by-epoch training log
├── cassava_b0_phase1_log.csv          # EfficientNetB0 epoch log
├── cassava_mobilenetv2_log.csv        # MobileNetV2 epoch log
│
├── app_screenshot_1.jpeg              # Live app screenshot
├── app_screenshot_2.jpeg              # Live app screenshot
├── app_screenshot_3.jpeg              # Live app screenshot
├── app_screenshot_4.jpeg              # Live app screenshot
└── app_screenshot_5.jpeg              # Live app screenshot

# Model files hosted on AWS S3 (too large for GitHub):
#   cassava_efficientnetb0_final.keras   (51 MB)
#   cassava_resnet50_final.keras         (280 MB)
#   cassava_mobilenetv2_final.h5         (29 MB)
```

---

## 8. Technologies Used

| Technology         | Purpose                                                      |
| ------------------ | ------------------------------------------------------------ |
| Python 3.11        | Programming language                                         |
| TensorFlow / Keras | Model training and inference                                 |
| Streamlit          | Web application framework                                    |
| boto3              | AWS SDK — S3 model download                                  |
| AWS S3             | Cloud object storage for model files                         |
| AWS IAM            | Cloud identity and access management                         |
| GitHub             | Source control and CI/CD pipeline                            |
| Streamlit Cloud    | Cloud application hosting                                    |
| Plotly             | Interactive confidence charts                                |
| scikit-learn       | Evaluation metrics (classification report, confusion matrix) |

---

## 9. Security Notes

- AWS IAM user `cassava-app-user` uses least-privilege access (S3 read-only)
- AWS credentials are stored as Streamlit Cloud secrets and injected as environment variables at runtime
- Credentials are never committed to the GitHub repository
- It is recommended to rotate AWS access keys periodically via IAM → Users → Security credentials

---

## 10. Acknowledgements

- Transfer learning architectures sourced from `tensorflow.keras.applications`
- Models hosted on AWS S3 (primary) and Hugging Face Hub (backup)
- Application deployed on Streamlit Community Cloud
- Dataset curated and cleaned manually for balanced class distribution
