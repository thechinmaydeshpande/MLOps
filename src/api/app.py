"""
FastAPI application for Heart Disease Prediction
"""
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
from pathlib import Path
import sys
import logging
from datetime import datetime
import json
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from model.inference import HeartDiseasePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total number of predictions', ['prediction'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency in seconds')
api_requests = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
model_loaded = Gauge('model_loaded', 'Whether the model is loaded')

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease risk based on patient health data",
    version="1.0.0"
)

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    api_requests.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    return response

# Request/Response models
class PatientData(BaseModel):
    """
    Patient health data for prediction
    """
    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1=male, 0=female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=0, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=0, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: float = Field(..., ge=0, le=4, description="Number of major vessels (0-3)")
    thal: float = Field(..., ge=0, description="Thalassemia")

    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }


class PredictionResponse(BaseModel):
    """
    Prediction response
    """
    prediction: int
    prediction_label: str
    probability_no_disease: float
    probability_disease: float
    confidence: float
    timestamp: str


class HealthResponse(BaseModel):
    """
    Health check response
    """
    status: str
    timestamp: str
    model_loaded: bool


# Global predictor instance
predictor = None


@app.on_event("startup")
async def load_model():
    """
    Load model and preprocessor on startup
    """
    global predictor

    try:
        # Paths (adjust for Docker deployment)
        model_path = Path('/app/models/best_model.pkl') if Path('/app/models').exists() else \
                     Path(__file__).parent.parent.parent / 'models' / 'best_model.pkl'

        preprocessor_path = Path('/app/models/preprocessor.pkl') if Path('/app/models').exists() else \
                           Path(__file__).parent.parent.parent / 'models' / 'preprocessor.pkl'

        predictor = HeartDiseasePredictor(model_path, preprocessor_path)
        model_loaded.set(1)
        logger.info("Model and preprocessor loaded successfully")

    except Exception as e:
        model_loaded.set(0)
        logger.error(f"Error loading model: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=predictor is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_data: PatientData):
    """
    Make prediction for patient data
    """
    if predictor is None:
        logger.error("Predictor not initialized")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Log request
        logger.info(f"Prediction request received: {patient_data.dict()}")

        # Convert to dict
        input_data = patient_data.dict()

        # Make prediction with timing
        start_time = time.time()
        result = predictor.predict(input_data)
        prediction_latency.observe(time.time() - start_time)

        # Track prediction
        prediction_counter.labels(prediction=result['prediction_label']).inc()

        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()

        # Log result
        logger.info(f"Prediction result: {result}")

        # Log to file for monitoring
        log_prediction(input_data, result)

        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def log_prediction(input_data, result):
    """
    Log predictions for monitoring
    """
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input': input_data,
            'output': result
        }

        log_file = Path('predictions_log.jsonl')
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    except Exception as e:
        logger.error(f"Error logging prediction: {e}")


@app.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/stats")
async def get_stats():
    """
    Get API statistics for monitoring
    """
    try:
        # Read prediction logs
        log_file = Path('predictions_log.jsonl')

        if not log_file.exists():
            return {
                "total_predictions": 0,
                "disease_predictions": 0,
                "no_disease_predictions": 0
            }

        total = 0
        disease_count = 0
        no_disease_count = 0

        with open(log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                total += 1
                if entry['output']['prediction'] == 1:
                    disease_count += 1
                else:
                    no_disease_count += 1

        return {
            "total_predictions": total,
            "disease_predictions": disease_count,
            "no_disease_predictions": no_disease_count,
            "disease_rate": disease_count / total if total > 0 else 0
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
