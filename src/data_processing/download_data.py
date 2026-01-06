"""
Script to download Heart Disease UCI Dataset
"""
import pandas as pd
import os
from pathlib import Path

def download_heart_disease_data():
    """
    Download the Heart Disease UCI dataset
    """
    # UCI Heart Disease dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    # Column names as per UCI documentation
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download and save dataset
    try:
        print("Downloading Heart Disease UCI dataset...")
        df = pd.read_csv(url, names=column_names, na_values='?')

        output_path = data_dir / 'heart_disease_raw.csv'
        df.to_csv(output_path, index=False)
        print(f"Dataset downloaded successfully to {output_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values:\n{df.isnull().sum()}")

        return df
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    download_heart_disease_data()
