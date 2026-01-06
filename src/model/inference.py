"""
Model inference module
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from data_processing.preprocessing import HeartDiseasePreprocessor


class HeartDiseasePredictor:
    """
    Inference class for heart disease prediction
    """

    def __init__(self, model_path, preprocessor_path):
        """
        Initialize the predictor

        Args:
            model_path: Path to the trained model pickle file
            preprocessor_path: Path to the preprocessor pickle file
        """
        self.model = self.load_model(model_path)
        self.preprocessor = self.load_preprocessor(preprocessor_path)

    def load_model(self, model_path):
        """
        Load the trained model
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return model

    def load_preprocessor(self, preprocessor_path):
        """
        Load the preprocessor
        """
        preprocessor = HeartDiseasePreprocessor()
        preprocessor.load_preprocessor(preprocessor_path)
        return preprocessor

    def preprocess_input(self, input_data):
        """
        Preprocess input data for prediction

        Args:
            input_data: Dictionary or DataFrame with patient data

        Returns:
            Preprocessed features ready for prediction
        """
        # Convert dict to DataFrame if needed
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()

        # Preprocess
        X, _ = self.preprocessor.preprocess(df, fit=False)

        return X

    def predict(self, input_data):
        """
        Make prediction for input data

        Args:
            input_data: Dictionary or DataFrame with patient data

        Returns:
            Dictionary with prediction and probability
        """
        # Preprocess
        X = self.preprocess_input(input_data)

        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]

        result = {
            'prediction': int(prediction),
            'prediction_label': 'Disease' if prediction == 1 else 'No Disease',
            'probability_no_disease': float(probability[0]),
            'probability_disease': float(probability[1]),
            'confidence': float(max(probability))
        }

        return result

    def predict_batch(self, input_data):
        """
        Make predictions for multiple inputs

        Args:
            input_data: DataFrame with patient data

        Returns:
            List of prediction dictionaries
        """
        # Preprocess
        X = self.preprocess_input(input_data)

        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        results = []
        for i in range(len(predictions)):
            result = {
                'prediction': int(predictions[i]),
                'prediction_label': 'Disease' if predictions[i] == 1 else 'No Disease',
                'probability_no_disease': float(probabilities[i][0]),
                'probability_disease': float(probabilities[i][1]),
                'confidence': float(max(probabilities[i]))
            }
            results.append(result)

        return results


def test_inference():
    """
    Test the inference pipeline
    """
    # Paths
    model_path = Path(__file__).parent.parent.parent / 'models' / 'best_model.pkl'
    preprocessor_path = Path(__file__).parent.parent.parent / 'models' / 'preprocessor.pkl'

    # Initialize predictor
    predictor = HeartDiseasePredictor(model_path, preprocessor_path)

    # Test sample
    test_sample = {
        'age': 63,
        'sex': 1,
        'cp': 3,
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1
    }

    print("Test Input:")
    print(test_sample)
    print("\nPrediction:")
    result = predictor.predict(test_sample)
    print(result)


if __name__ == "__main__":
    test_inference()
