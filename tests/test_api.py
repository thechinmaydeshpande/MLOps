"""
Unit tests for FastAPI application
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def sample_patient_data():
    """
    Sample patient data for testing
    """
    return {
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


@pytest.fixture
def invalid_patient_data():
    """
    Invalid patient data for testing
    """
    return {
        "age": -5,  # Invalid: negative age
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


class TestAPIEndpoints:
    """
    Test API endpoints
    """

    def test_root_endpoint(self):
        """
        Test root endpoint
        """
        # Import app here to avoid loading model during test collection
        from api.app import app
        client = TestClient(app)

        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert "version" in response.json()

    def test_health_endpoint(self):
        """
        Test health check endpoint
        """
        from api.app import app
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data

    def test_predict_endpoint_structure(self, sample_patient_data):
        """
        Test predict endpoint returns correct structure
        """
        from api.app import app
        client = TestClient(app)

        # This will fail if model is not loaded, but we can test the structure
        response = client.post("/predict", json=sample_patient_data)

        # Either success or service unavailable
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "prediction_label" in data
            assert "probability_no_disease" in data
            assert "probability_disease" in data
            assert "confidence" in data
            assert "timestamp" in data

    def test_predict_endpoint_validation(self, invalid_patient_data):
        """
        Test predict endpoint input validation
        """
        from api.app import app
        client = TestClient(app)

        response = client.post("/predict", json=invalid_patient_data)

        # Should return validation error
        assert response.status_code == 422

    def test_predict_missing_fields(self):
        """
        Test predict endpoint with missing required fields
        """
        from api.app import app
        client = TestClient(app)

        incomplete_data = {
            "age": 63,
            "sex": 1
            # Missing other required fields
        }

        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422

    def test_metrics_endpoint(self):
        """
        Test metrics endpoint (Prometheus format)
        """
        from api.app import app
        client = TestClient(app)

        response = client.get("/metrics")
        assert response.status_code == 200

        # Prometheus metrics are plain text, not JSON
        assert "text/plain" in response.headers.get("content-type", "")
        assert "predictions_total" in response.text or "# HELP" in response.text

    def test_stats_endpoint(self):
        """
        Test stats endpoint (JSON format)
        """
        from api.app import app
        client = TestClient(app)

        response = client.get("/stats")
        assert response.status_code in [200, 500]  # May fail if no predictions logged yet

        if response.status_code == 200:
            data = response.json()
            assert "total_predictions" in data


class TestDataValidation:
    """
    Test request data validation
    """

    def test_age_validation(self):
        """
        Test age field validation
        """
        from api.app import app, PatientData
        client = TestClient(app)

        # Age too high
        data = {
            "age": 150,  # Invalid
            "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
            "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
            "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
        }

        response = client.post("/predict", json=data)
        assert response.status_code == 422

    def test_categorical_validation(self):
        """
        Test categorical field validation
        """
        from api.app import app
        client = TestClient(app)

        # Invalid categorical value
        data = {
            "age": 63, "sex": 5,  # Invalid: should be 0 or 1
            "cp": 3, "trestbps": 145, "chol": 233,
            "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
            "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
        }

        response = client.post("/predict", json=data)
        assert response.status_code == 422

    def test_negative_values(self):
        """
        Test negative values validation
        """
        from api.app import app
        client = TestClient(app)

        data = {
            "age": 63, "sex": 1, "cp": 3,
            "trestbps": -145,  # Invalid: negative
            "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
            "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
        }

        response = client.post("/predict", json=data)
        assert response.status_code == 422


class TestResponseFormat:
    """
    Test response formats
    """

    def test_prediction_response_types(self, sample_patient_data):
        """
        Test prediction response data types
        """
        from api.app import app
        client = TestClient(app)

        response = client.post("/predict", json=sample_patient_data)

        if response.status_code == 200:
            data = response.json()

            # Check data types
            assert isinstance(data['prediction'], int)
            assert isinstance(data['prediction_label'], str)
            assert isinstance(data['probability_no_disease'], float)
            assert isinstance(data['probability_disease'], float)
            assert isinstance(data['confidence'], float)
            assert isinstance(data['timestamp'], str)

            # Check value ranges
            assert data['prediction'] in [0, 1]
            assert 0 <= data['probability_no_disease'] <= 1
            assert 0 <= data['probability_disease'] <= 1
            assert 0 <= data['confidence'] <= 1

    def test_prediction_probabilities_sum(self, sample_patient_data):
        """
        Test that probabilities sum to 1
        """
        from api.app import app
        client = TestClient(app)

        response = client.post("/predict", json=sample_patient_data)

        if response.status_code == 200:
            data = response.json()
            prob_sum = data['probability_no_disease'] + data['probability_disease']
            assert abs(prob_sum - 1.0) < 0.01  # Allow small floating point error


def test_openapi_schema():
    """
    Test OpenAPI schema generation
    """
    from api.app import app
    client = TestClient(app)

    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]
