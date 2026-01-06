# Heart Disease Prediction - MLOps Project

A complete MLOps pipeline for predicting heart disease risk based on patient health data, featuring automated training, testing, deployment, and monitoring.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Training](#model-training)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Monitoring](#monitoring)
- [CI/CD Pipeline](#cicd-pipeline)
- [Testing](#testing)

## Overview

This project implements an end-to-end machine learning pipeline for heart disease prediction using the UCI Heart Disease dataset. The solution includes:

- **Data Processing**: Automated data download, cleaning, and feature engineering
- **Model Training**: Multiple classification models with hyperparameter tuning
- **Experiment Tracking**: MLflow integration for tracking experiments
- **API Serving**: FastAPI-based REST API for predictions
- **Containerization**: Docker containers for consistent deployment
- **Orchestration**: Kubernetes manifests for scalable deployment
- **Monitoring**: Prometheus and Grafana integration
- **CI/CD**: Automated testing and deployment pipeline

## Features

- ✅ Comprehensive EDA with professional visualizations
- ✅ Feature engineering and preprocessing pipeline
- ✅ Multiple ML models (Logistic Regression, Random Forest, Gradient Boosting)
- ✅ MLflow experiment tracking
- ✅ RESTful API with FastAPI
- ✅ Docker containerization
- ✅ Kubernetes deployment manifests
- ✅ Prometheus metrics and Grafana dashboards
- ✅ Automated testing with pytest
- ✅ GitHub Actions CI/CD pipeline
- ✅ Complete documentation

## Project Structure

```
heart_disease_mlops/
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # GitHub Actions workflow
├── data/
│   ├── heart_disease_raw.csv       # Raw dataset
│   └── heart_disease_cleaned.csv   # Cleaned dataset
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile              # Docker image definition
│   │   └── docker-compose.yml      # Docker Compose configuration
│   ├── kubernetes/
│   │   ├── deployment.yaml         # Kubernetes deployment
│   │   ├── service.yaml            # Kubernetes service
│   │   ├── configmap.yaml          # Configuration
│   │   ├── hpa.yaml                # Horizontal Pod Autoscaler
│   │   └── ingress.yaml            # Ingress configuration
│   └── monitoring/
│       └── prometheus.yml          # Prometheus configuration
├── models/
│   ├── best_model.pkl              # Trained model
│   ├── preprocessor.pkl            # Preprocessing pipeline
│   └── best_model_info.pkl         # Model metadata
├── notebooks/
│   └── 01_EDA.ipynb                # Exploratory Data Analysis
├── screenshots/                     # Visualizations and screenshots
├── src/
│   ├── api/
│   │   └── app.py                  # FastAPI application
│   ├── data_processing/
│   │   ├── download_data.py        # Data download script
│   │   └── preprocessing.py        # Preprocessing pipeline
│   └── model/
│       ├── train.py                # Model training script
│       └── inference.py            # Inference module
├── tests/
│   ├── test_preprocessing.py       # Preprocessing tests
│   └── test_api.py                 # API tests
├── .dockerignore
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- Docker (for containerization)
- Kubernetes/Minikube (for local K8s deployment)
- Git

### Local Installation

1. **Clone the repository** (or navigate to the project directory):
```bash
cd heart_disease_mlops
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download the dataset**:
```bash
python src/data_processing/download_data.py
```

## Usage

### 1. Exploratory Data Analysis

Run the Jupyter notebook:
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### 2. Train Models

Train multiple models with MLflow tracking:
```bash
python src/model/train.py
```

This will:
- Load and preprocess the data
- Train Logistic Regression, Random Forest, and Gradient Boosting models
- Log experiments to MLflow
- Save the best model

View MLflow experiments:
```bash
mlflow ui
```
Open http://localhost:5000 in your browser.

### 3. Test Inference

Test the trained model:
```bash
python src/model/inference.py
```

## API Documentation

### Start the API Locally

```bash
python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

- **GET /** - Root endpoint with API information
- **GET /health** - Health check
- **POST /predict** - Make predictions
- **GET /metrics** - Prometheus metrics
- **GET /stats** - Prediction statistics
- **GET /docs** - Interactive API documentation (Swagger UI)

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```

### Example Response

```json
{
  "prediction": 1,
  "prediction_label": "Disease",
  "probability_no_disease": 0.23,
  "probability_disease": 0.77,
  "confidence": 0.77,
  "timestamp": "2025-12-31T10:30:00"
}
```

## Docker Deployment

### Build the Docker Image

```bash
docker build -t heart-disease-api:latest -f deployment/docker/Dockerfile .
```

### Run the Container

```bash
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest
```

### Using Docker Compose

```bash
cd deployment/docker
docker-compose up -d
```

This will start:
- API service on port 8000
- Prometheus on port 9090
- Grafana on port 3000

### Test the Container

```bash
curl http://localhost:8000/health
```

### Stop the Container

```bash
docker stop heart-disease-api
# Or with docker-compose
docker-compose down
```

## Kubernetes Deployment

### Prerequisites

- Minikube or Docker Desktop with Kubernetes enabled
- kubectl installed

### Start Minikube (if using Minikube)

```bash
minikube start
```

### Build and Load Image

```bash
# Build the image
docker build -t heart-disease-api:latest -f deployment/docker/Dockerfile .

# Load into Minikube (if using Minikube)
minikube image load heart-disease-api:latest
```

### Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f deployment/kubernetes/

# Or apply individually
kubectl apply -f deployment/kubernetes/configmap.yaml
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/hpa.yaml
kubectl apply -f deployment/kubernetes/ingress.yaml
```

### Check Deployment Status

```bash
# Check pods
kubectl get pods

# Check services
kubectl get services

# Check deployment
kubectl get deployments
```

### Access the API

```bash
# Get the service URL (Minikube)
minikube service heart-disease-api-service --url

# Or use port forwarding
kubectl port-forward service/heart-disease-api-service 8000:80
```

### View Logs

```bash
kubectl logs -f deployment/heart-disease-api
```

### Scale the Deployment

```bash
kubectl scale deployment heart-disease-api --replicas=5
```

### Clean Up

```bash
kubectl delete -f deployment/kubernetes/
```

## Monitoring

### Prometheus Metrics

Access Prometheus at http://localhost:9090 (when using Docker Compose)

Available metrics:
- `predictions_total` - Total number of predictions by type
- `prediction_latency_seconds` - Prediction latency histogram
- `api_requests_total` - Total API requests by endpoint and status
- `model_loaded` - Whether the model is loaded (0 or 1)

### Grafana Dashboards

Access Grafana at http://localhost:3000 (when using Docker Compose)

Default credentials:
- Username: admin
- Password: admin

### View API Statistics

```bash
curl http://localhost:8000/stats
```

## CI/CD Pipeline

The GitHub Actions workflow automates:

1. **Linting**: Code quality checks with flake8, black, and pylint
2. **Testing**: Unit tests with pytest and coverage reporting
3. **Training**: Model training and artifact generation
4. **Docker Build**: Container image creation and testing
5. **Security**: Vulnerability scanning with Trivy

### Workflow Triggers

- Push to `main` or `develop` branches
- Pull requests to `main`

### View Workflow Results

Check the Actions tab in your GitHub repository.

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=src --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # On macOS
# Or on Linux: xdg-open htmlcov/index.html
```

### Run Specific Tests

```bash
# Test preprocessing only
pytest tests/test_preprocessing.py -v

# Test API only
pytest tests/test_api.py -v
```

### Linting

```bash
# Flake8
flake8 src tests

# Black (formatting)
black src tests --check

# Pylint
pylint src
```

## Model Information

### Dataset

- **Source**: UCI Machine Learning Repository
- **Samples**: 303 patients
- **Features**: 14 (13 features + 1 target)
- **Target**: Binary classification (Disease/No Disease)

### Models Trained

1. **Logistic Regression**
   - Fast baseline model
   - Interpretable coefficients

2. **Random Forest**
   - Ensemble method
   - Feature importance analysis

3. **Gradient Boosting**
   - Advanced ensemble technique
   - Often best performance

### Model Evaluation

All models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Cross-validation

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests: `pytest tests/`
4. Run linting: `flake8 src tests`
5. Submit a pull request

## License

This project is for educational purposes as part of MLOps coursework.

## Contact

For questions or issues, please open an issue in the repository.

## Acknowledgments

- UCI Machine Learning Repository for the dataset
- BITS Pilani for the MLOps course
- FastAPI, MLflow, and scikit-learn communities
