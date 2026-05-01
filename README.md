# 🚀 Customer Segmentation MLOps Pipeline : 
Application link : [https://customer-segmentation-mlops-pipeline.streamlit.app/]

An end-to-end **MLOps project** that builds, tracks, deploys, and serves a customer segmentation model using modern industry tools like **MLflow, DagsHub, Docker, FastAPI, Streamlit, and GitHub Actions**.

---

## 📌 Project Overview

This project implements a complete machine learning lifecycle:

* 📥 MongoDb Data ingestion & validation
* 🔄 Feature engineering & transformation
* 🤖 Model training (KMeans clustering)
* 📊 Experiment tracking with MLflow + DagsHub
* 🧠 Model evaluation with multiple clustering metrics
* 🌐 Model serving via FastAPI
* 🎨 Interactive UI using Streamlit
* 🐳 Dockerized deployment
* 🔁 CI/CD pipeline using GitHub Actions

---

## 🧠 Business Problem

Customer segmentation helps businesses:

* Identify high-value customers
* Improve marketing strategies
* Personalize customer experience
* Increase retention and revenue

This project clusters customers based on behavioral features like recency, frequency, monetary value, and more.

---

## 🏗️ Project Architecture

```
Raw Data → Ingestion → Validation → Transformation → Model Training
                                                            ↓
                                                    MLflow Tracking
                                                            ↓
                                                    Model Registry
                                                            ↓
                                                    ┌───────────────┬
                                                    ↓               ↓
                                                FastAPI API     Streamlit UI
                                                            ↓
                                                    Docker Container
                                                            ↓
                                                    Docker Hub + CI/CD
```

---

## 📁 Project Structure

```
Customer-segmentation-MLOps-Pipeline/
│
├── artifacts/                # Processed data, models, logs
├── configs/                  # Configuration files
├── src/
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   ├── model_training/
│   ├── model_tracking/
│   ├── pipelines/
│   └── utils/
│
├── tests/                    # Pipeline tests
├── app.py                    # FastAPI app
├── streamlit_app.py          # Streamlit UI
├── main.py                   # Pipeline entry point
├── Dockerfile
├── requirements.txt
└── .github/workflows/        # CI/CD pipeline
```

---

## ⚙️ Tech Stack

* **Python** (Core language)
* **Scikit-learn** (Modeling)
* **MongoDb** (Database)
* **MLflow + DagsHub** (Experiment tracking & registry)
* **FastAPI** (Model serving)
* **Streamlit** (UI)
* **Docker** (Containerization)
* **GitHub Actions** (CI/CD)

---

## 📊 Model Details

* Algorithm: **KMeans Clustering**
* Features:

  * Recency
  * Frequency
  * Monetary
  * Quantity
  * Discount
  * Delivery Days
  * Customer Rating

### 📈 Evaluation Metrics

* Silhouette Score
* Davies-Bouldin Score
* Calinski-Harabasz Score

---

## 🔍 MLflow Tracking

All experiments are tracked using MLflow integrated with DagsHub:

* Parameters (e.g., `n_clusters`)
* Metrics (multiple clustering scores)
* Model artifacts
* Dataset info

👉 Enables reproducibility and comparison across runs.

---

## 🌐 Applications

### 1. FastAPI (Optional Serving Layer)

```bash
uvicorn app:app --reload
```

* REST API for predictions
* Swagger UI available at `/docs`

---

### 2. Streamlit UI (Primary Interface) : https://customer-segmentation-mlops-pipeline.streamlit.app/

```bash
streamlit run streamlit_app.py
```

* Interactive sliders for inputs
* Real-time cluster prediction
* User-friendly visualization

---

## 🐳 Dockerization

Build and run the app:

```bash
docker build -t customer-segmentation-app .
docker run -p 8501:8501 customer-segmentation-app
```

Access:

```
http://localhost:8501
```

---

## 🔁 CI/CD Pipeline

Implemented using **GitHub Actions**:

* Trigger: Push to `main`
* Steps:

  * Checkout code
  * Build Docker image
  * Push image to Docker Hub

👉 Ensures automated and reproducible builds.

---

## 🧪 Testing

Basic pipeline validation using `pytest`:

```bash
pytest tests/
```

---

## 🚀 Future Enhancements

Planned production-grade improvements:

### ☁️ Cloud Integration

* Store model artifacts in **AWS S3**
* Load model dynamically during runtime

### 🖥️ Deployment Automation

* Deploy Docker container on **AWS EC2**
* Automate deployment via CI/CD pipeline

### 🔄 Advanced MLOps

* Add data & model validation gates
* Introduce model versioning strategies
* Enable monitoring & logging

---

## 🎯 Key Highlights

* End-to-end MLOps pipeline
* Modular and scalable code structure
* Real-world deployment practices
* Clean separation of concerns
* Production-ready architecture

---

## 👨‍💻 Author

**Harshith Narasimhamurthy**

* 📧 [harshithnchandan@gmail.com](mailto:harshithnchandan@gmail.com)
* 📱 +91 9663918804
* 🔗 [LinkedIn](https://www.linkedin.com/in/harshithnarasimhamurthy69/)

---

