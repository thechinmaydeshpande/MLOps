"""
Data preprocessing and feature engineering module
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path


class HeartDiseasePreprocessor:
    """
    Preprocessing pipeline for Heart Disease dataset
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        self.categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    def clean_data(self, df):
        """
        Clean the dataset by handling missing values
        """
        df_clean = df.copy()

        # Fill missing values with mode for categorical features
        for col in ['ca', 'thal']:
            if col in df_clean.columns and df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

        return df_clean

    def engineer_features(self, df):
        """
        Create additional features
        """
        df_engineered = df.copy()

        # Age groups
        df_engineered['age_group'] = pd.cut(df_engineered['age'],
                                             bins=[0, 45, 55, 65, 100],
                                             labels=[0, 1, 2, 3])

        # Blood pressure categories
        df_engineered['bp_category'] = pd.cut(df_engineered['trestbps'],
                                               bins=[0, 120, 140, 200],
                                               labels=[0, 1, 2])

        # Cholesterol categories
        df_engineered['chol_category'] = pd.cut(df_engineered['chol'],
                                                 bins=[0, 200, 240, 600],
                                                 labels=[0, 1, 2])

        # Convert categorical features to numeric if needed
        df_engineered['age_group'] = df_engineered['age_group'].astype(float)
        df_engineered['bp_category'] = df_engineered['bp_category'].astype(float)
        df_engineered['chol_category'] = df_engineered['chol_category'].astype(float)

        return df_engineered

    def prepare_features(self, df, fit=True):
        """
        Prepare features for modeling (scaling)
        """
        df_prep = df.copy()

        # Separate features and target
        if 'target' in df_prep.columns:
            X = df_prep.drop('target', axis=1)
            y = df_prep['target']
        else:
            X = df_prep
            y = None

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Scale numerical features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        # Convert back to DataFrame
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)

        return X_scaled, y

    def preprocess(self, df, fit=True):
        """
        Complete preprocessing pipeline
        """
        # Clean data
        df_clean = self.clean_data(df)

        # Engineer features
        df_engineered = self.engineer_features(df_clean)

        # Prepare features
        X, y = self.prepare_features(df_engineered, fit=fit)

        return X, y

    def save_preprocessor(self, filepath):
        """
        Save the preprocessor (scaler and feature names)
        """
        preprocessor_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)

        print(f"Preprocessor saved to {filepath}")

    def load_preprocessor(self, filepath):
        """
        Load the preprocessor
        """
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)

        self.scaler = preprocessor_data['scaler']
        self.feature_names = preprocessor_data['feature_names']
        self.numerical_features = preprocessor_data['numerical_features']
        self.categorical_features = preprocessor_data['categorical_features']

        print(f"Preprocessor loaded from {filepath}")


def load_and_split_data(filepath, test_size=0.2, random_state=42):
    """
    Load data and split into train/test sets
    """
    df = pd.read_csv(filepath)

    # Convert target to binary if needed
    if 'target' in df.columns and df['target'].max() > 1:
        df['target'] = (df['target'] > 0).astype(int)

    # Initialize preprocessor
    preprocessor = HeartDiseasePreprocessor()

    # Preprocess data
    X, y = preprocessor.preprocess(df, fit=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Test the preprocessing pipeline
    data_path = Path(__file__).parent.parent.parent / 'data' / 'heart_disease_raw.csv'

    X_train, X_test, y_train, y_test, preprocessor = load_and_split_data(data_path)

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Feature names: {preprocessor.feature_names}")

    # Save preprocessor
    preprocessor_path = Path(__file__).parent.parent.parent / 'models' / 'preprocessor.pkl'
    preprocessor.save_preprocessor(preprocessor_path)
