# MAKERERE UNIVERSITY

## COLLEGE OF COMPUTING AND INFORMATION TECHNOLOGY

### DEPARTMENT OF COMPUTER SCIENCE

---

# ☁ CLOUD COMPUTING PROJECT REPORT

## Cassava Leaf Disease Detection

### A Deep Learning Web Application Deployed on AWS Cloud

---

|                       |                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------ |
| **Student Name**      | Ebaju Edward                                                                         |
| **Student Number**    | 2400723929                                                                           |
| **GitHub Repository** | github.com/Edward-Ed4/Cassava-Disease-Classification-with-Deep-learning              |
| **Live Application**  | https://cassava-disease-classification-with-deep-learning-t7jyapv6cayz.streamlit.app |

---

## Table of Contents

1. Introduction
2. Project Objectives
3. Cloud Architecture
4. Services Used
   - 4.1 Amazon S3 (Simple Storage Service)
   - 4.2 AWS IAM (Identity and Access Management)
   - 4.3 Streamlit Cloud
   - 4.4 GitHub
5. Application Features
6. Implementation
   - 6.1 Model Training
   - 6.2 Model Storage on AWS S3
   - 6.3 Application (Streamlit)
   - 6.4 Security
   - 6.5 Source Code
7. Deployment Process
8. Cloud Computing Concepts Demonstrated
9. Challenges Encountered
   - 9.1 Python Version Incompatibility
   - 9.2 Model File Size Exceeding GitHub Limits
   - 9.3 AWS Credentials Not Reaching the App
   - 9.4 Incorrect Hugging Face Repository ID
10. Live Application
11. Conclusion
12. References

---

## 1. Introduction

Cloud computing has transformed how machine learning applications are built, deployed, and scaled. Instead of managing physical servers or requiring users to install local software, developers can leverage cloud platforms to host models and applications that are globally accessible, highly scalable, and cost-effective.

This project demonstrates the practical application of cloud computing concepts by training deep learning models for cassava leaf disease detection and deploying them as a fully functional web application using Amazon Web Services and Streamlit Cloud.

Cassava (_Manihot esculenta_) is a critical food crop across Sub-Saharan Africa. Early and accurate detection of leaf diseases is essential for protecting yields and food security. This project trains three convolutional neural network models — EfficientNetB0, ResNet50, and MobileNetV2 — to classify cassava leaf images into four disease categories and deploys them as an interactive web application that supports both image upload and live camera capture.

The trained models are stored on AWS S3 and downloaded automatically by the application at startup, demonstrating a production-grade cloud ML deployment pattern.

---

## 2. Project Objectives

- Design and deploy a cloud-native machine learning web application using AWS services
- Demonstrate the use of AWS S3 as a managed cloud object storage service for ML model artifacts
- Apply AWS IAM for secure, least-privilege programmatic access to cloud resources
- Host a publicly accessible web application on Streamlit Cloud with automatic deployment via GitHub
- Implement secure secrets management by storing AWS credentials as environment variables
- Train and compare three transfer learning models (EfficientNetB0, ResNet50, MobileNetV2) for cassava disease classification
- Apply cloud computing concepts including managed services, separation of concerns, and continuous deployment

---

## 3. Cloud Architecture

The application follows a three-tier cloud architecture with complete separation between code, compute, and data storage.

### 3.1 Architecture Summary

```
Developer pushes code
        ↓
GitHub (cloud source control + CI/CD trigger)
        ↓  auto-deploy on every push to main branch
Streamlit Cloud (cloud application hosting)
        ↓  downloads models at startup via boto3 SDK
AWS S3 — eu-north-1 Stockholm (cloud object storage)
   ├── cassava_efficientnetb0_final.keras  (51 MB)
   ├── cassava_resnet50_final.keras        (280 MB)
   └── cassava_mobilenetv2_final.h5        (29 MB)
```

### 3.2 Architecture Layer Table

| Layer               | Service           | Role                                                           |
| ------------------- | ----------------- | -------------------------------------------------------------- |
| Source Control      | GitHub            | Hosts application code and triggers automatic redeployment     |
| Application Hosting | Streamlit Cloud   | Runs the web application and serves it globally                |
| Model Storage       | AWS S3            | Stores trained ML model files as cloud objects                 |
| Access Control      | AWS IAM           | Controls programmatic access to S3 with least-privilege policy |
| Secrets Management  | Streamlit Secrets | Injects AWS credentials as environment variables at runtime    |

---

## 4. Services Used

### 4.1 Amazon S3 (Simple Storage Service)

Amazon S3 is an object storage service that offers industry-leading scalability, data availability, security, and performance. In this project, S3 is used as a cloud ML artifact registry — storing the three trained model files that are too large to commit to GitHub.

**Bucket configuration:**

- Bucket name: `cassava-disease-models`
- Region: `eu-north-1` (Europe — Stockholm)
- Access: Private (accessed programmatically via IAM credentials)

**Objects stored:**

| File                                 | Size   | Description                  |
| ------------------------------------ | ------ | ---------------------------- |
| `cassava_efficientnetb0_final.keras` | 51 MB  | Trained EfficientNetB0 model |
| `cassava_resnet50_final.keras`       | 280 MB | Trained ResNet50 model       |
| `cassava_mobilenetv2_final.h5`       | 29 MB  | Trained MobileNetV2 model    |

**Key features used:**

- Object storage for large ML model files
- Private bucket with IAM-controlled access
- Regional storage in eu-north-1 for low latency
- Programmatic access via the `boto3` Python SDK

S3 was chosen because GitHub enforces a 100 MB file size limit, making it unsuitable for storing large trained model files. S3 provides durable, scalable, and cost-effective storage — the 360 MB of model files costs approximately $0.01 per month.

### 4.2 AWS IAM (Identity and Access Management)

AWS IAM is used to securely control access to AWS services and resources. A dedicated IAM user was created for the application following the principle of least privilege.

**IAM configuration:**

- IAM user: `cassava-app-user`
- Attached policy: `AmazonS3ReadOnlyAccess`
- Access type: Programmatic access (access key ID + secret access key)

The IAM user has read-only access to S3 and no other AWS permissions. This means even if the credentials were compromised, an attacker could only read from S3 — they could not write, delete, or access any other AWS service.

**Key features used:**

- Least-privilege access control
- Programmatic access keys for application authentication
- Policy-based permissions management

### 4.3 Streamlit Cloud

Streamlit Community Cloud is a free managed platform for deploying Streamlit web applications. It hosts the application, manages the runtime environment, and automatically redeploys when changes are pushed to GitHub.

**Configuration:**

- Python version: 3.11
- Main file: `cassava_app.py`
- Dependencies: `requirements.txt`
- Secrets: AWS credentials injected as environment variables

**Key features used:**

- Serverless application hosting — no server management required
- Automatic redeployment on GitHub push
- Secrets management for secure credential storage
- Global accessibility via public URL

### 4.4 GitHub

GitHub serves as the source control and continuous deployment pipeline for the project.

**Key features used:**

- Cloud-hosted version control for all source code, notebooks, and assets
- Automatic deployment trigger — every push to the `main` branch triggers a redeploy on Streamlit Cloud
- Public repository for open access to the project codebase

---

## 5. Application Features

- **Model selector** — switch between EfficientNetB0, ResNet50, and MobileNetV2 from the sidebar
- **Camera capture** — take a live photo directly from the device camera; camera closes automatically after capture
- **Image upload** — upload one or more images from the device
- **Disease prediction** — predicted class with confidence score displayed prominently
- **Severity rating** — colour-coded severity level (None / Moderate / High / Very High) per disease
- **Treatment advice** — actionable, disease-specific treatment recommendations
- **Probability chart** — horizontal bar chart showing model confidence across all four classes
- **Student identification** — author name and student number displayed in the sidebar

---

## 6. Implementation

### 6.1 Model Training

Three models were trained using a two-phase transfer learning strategy on a dataset of 9,726 cassava leaf images across four classes.

**Dataset:**

| Property     | Value                      |
| ------------ | -------------------------- |
| Total images | 9,726                      |
| Classes      | 4                          |
| Split        | 80% train / 20% validation |
| Train images | 7,780                      |
| Val images   | 1,946                      |

**Classes:**

- Cassava Brown Streak Disease (Very High severity)
- Cassava Mosaic Disease (High severity)
- Cassava Green Mottle (Moderate severity)
- Healthy

**Training phases:**

- Phase 1 — Train only the classification head with the base model frozen
- Phase 2 — Unfreeze the top 30–50% of the base model and fine-tune at a lower learning rate

**Results:**

| Model          | Phase 1 Val Accuracy | Phase 2 Val Accuracy | Val Loss  |
| -------------- | -------------------- | -------------------- | --------- |
| EfficientNetB0 | 68.55%               | 74.67%               | 0.778     |
| MobileNetV2    | 65.26%               | 77.24%               | 0.745     |
| **ResNet50**   | **69.37%**           | **81.50%**           | **0.624** |

ResNet50 achieved the best performance at 81.5% validation accuracy with the lowest validation loss of 0.624.

### 6.2 Model Storage on AWS S3

After training, the three model files were uploaded to the S3 bucket `cassava-disease-models` in eu-north-1. The application downloads them at startup using the `boto3` SDK:

```python
s3 = boto3.client(
    "s3",
    region_name="eu-north-1",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
)
s3.download_file(S3_BUCKET, filename, local_path)
```

Models are cached using `@st.cache_resource` so they are only downloaded once per app session, not on every user request.

### 6.3 Application (Streamlit)

The web application is built with Streamlit and runs the following workflow on each prediction:

1. User selects a model from the sidebar
2. User uploads an image or takes a photo
3. Image is resized to 224×224 and preprocessed using the model-specific preprocessing function
4. Model runs inference and returns class probabilities
5. The top prediction is displayed with confidence score, severity, treatment advice, and a probability chart

### 6.4 Security

- AWS credentials are stored as Streamlit Cloud secrets and injected as environment variables at runtime
- Credentials are never hardcoded in source code or committed to GitHub
- IAM user has read-only S3 access — least-privilege principle applied
- S3 bucket is private — not publicly accessible

### 6.5 Source Code

The full source code is publicly available on GitHub:
`https://github.com/Edward-Ed4/Cassava-Disease-Classification-with-Deep-learning`

---

## 7. Deployment Process

1. Trained three deep learning models locally using Jupyter notebooks
2. Created an AWS account and navigated to the S3 console
3. Created S3 bucket `cassava-disease-models` in region eu-north-1 (Stockholm)
4. Uploaded three model files to the S3 bucket (51 MB, 280 MB, 29 MB)
5. Created IAM user `cassava-app-user` with `AmazonS3ReadOnlyAccess` policy
6. Generated programmatic access keys (Access Key ID + Secret Access Key)
7. Updated `cassava_app.py` to download models from S3 using `boto3` instead of local file paths
8. Updated `requirements.txt` to include `boto3`
9. Initialised a Git repository and pushed all code to GitHub
10. Connected the GitHub repository to Streamlit Cloud
11. Set Python version to 3.11 in Streamlit Cloud app settings
12. Added AWS credentials to Streamlit Cloud secrets
13. Deployed the application — Streamlit Cloud installs dependencies, the app downloads models from S3 at startup

---

## 8. Cloud Computing Concepts Demonstrated

| Concept                   | How it is Demonstrated                                                          |
| ------------------------- | ------------------------------------------------------------------------------- |
| Managed Cloud Storage     | AWS S3 stores ML model files without any server or file system management       |
| Least-Privilege Access    | IAM user has read-only S3 access — no unnecessary permissions granted           |
| Serverless Hosting        | Streamlit Cloud hosts the app with no server provisioning or management         |
| Secrets Management        | AWS credentials injected as environment variables, never in source code         |
| Continuous Deployment     | GitHub push automatically triggers redeploy on Streamlit Cloud                  |
| Separation of Concerns    | Code (GitHub), models (S3), and compute (Streamlit Cloud) on separate platforms |
| Pay-as-you-go Storage     | S3 charges only for storage used — ~$0.01/month for 360 MB of models            |
| Cloud SDK Integration     | `boto3` Python SDK used to programmatically access AWS S3 from the application  |
| Global Accessibility      | Application accessible from any device, anywhere in the world via public URL    |
| Infrastructure Separation | No physical server management — all infrastructure managed by cloud providers   |

---

## 9. Challenges Encountered

### 9.1 Python Version Incompatibility

Streamlit Cloud defaulted to Python 3.14.4, which TensorFlow does not yet support. TensorFlow had no compatible wheel files for Python 3.14, causing the dependency installation to fail with an unsatisfiable requirements error.

**Resolution:** Set the Python version to 3.11 in the Streamlit Cloud app settings and pinned `tensorflow-cpu==2.17.1` in `requirements.txt`.

### 9.2 Model File Size Exceeding GitHub Limits

The three trained model files total approximately 360 MB, with the ResNet50 model alone being 280 MB. GitHub enforces a 100 MB file size limit per file, making it impossible to commit the model files to the repository.

**Resolution:** Uploaded the model files to AWS S3 and updated the application to download them programmatically at startup using the `boto3` SDK.

### 9.3 AWS Credentials Not Reaching the App

After switching from Hugging Face to AWS S3, the application threw a `botocore.exceptions.NoCredentialsError` on deployment. The app could not find the AWS access keys.

**Resolution:** Added the AWS credentials (`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`) to the Streamlit Cloud secrets panel. Streamlit injects these as environment variables at runtime, which `boto3` reads automatically.

### 9.4 Incorrect Hugging Face Repository ID

During an earlier deployment phase using Hugging Face Hub for model storage, the application threw a `RepositoryNotFoundError`. The repository ID hardcoded in the app was `EdwardEbaju-Ed4/cassava-disease-models` but the actual Hugging Face username was `Ebaju-Ed4`.

**Resolution:** Corrected the repository ID in `cassava_app.py` to `Ebaju-Ed4/cassava-disease-models` and pushed the fix to GitHub.

---

## 10. Live Application

| Resource                    | URL                                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------ |
| Live App (Streamlit Cloud)  | https://cassava-disease-classification-with-deep-learning-t7jyapv6cayz.streamlit.app |
| GitHub Repository           | https://github.com/Edward-Ed4/Cassava-Disease-Classification-with-Deep-learning      |
| AWS S3 Bucket               | `cassava-disease-models` — eu-north-1 (Stockholm)                                    |
| Model Backup (Hugging Face) | https://huggingface.co/Ebaju-Ed4/cassava-disease-models                              |

---

## 11. Conclusion

This project successfully demonstrates the design and deployment of a cloud-native machine learning web application using Amazon Web Services. The Cassava Leaf Disease Detection application leverages AWS S3 for scalable cloud object storage of trained model artifacts, AWS IAM for secure least-privilege access control, Streamlit Cloud for serverless application hosting, and GitHub for source control and continuous deployment.

The project illustrates key cloud computing concepts including managed services, separation of concerns, secrets management, pay-as-you-go storage, and continuous deployment pipelines. The application is fully functional, publicly accessible from anywhere in the world, and requires zero server management — demonstrating the power of modern cloud platforms for machine learning deployment.

The best performing model, ResNet50, achieved 81.5% validation accuracy on the four-class cassava disease classification task. The complete system — from model training to live deployment — showcases a production-grade cloud ML architecture that scales automatically and costs near zero when idle.

---

## 12. References

- Amazon Web Services Documentation: https://docs.aws.amazon.com
- Amazon S3 Documentation: https://docs.aws.amazon.com/s3/
- AWS IAM Documentation: https://docs.aws.amazon.com/iam/
- boto3 Documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
- Streamlit Documentation: https://docs.streamlit.io
- Streamlit Cloud Deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud
- TensorFlow Keras Documentation: https://www.tensorflow.org/api_docs/python/tf/keras
- GitHub Repository: https://github.com/Edward-Ed4/Cassava-Disease-Classification-with-Deep-learning

---

_Ebaju Edward | Student No: 2400723929 | Cloud Computing Project_
