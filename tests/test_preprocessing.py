"""
Unit tests for data preprocessing module
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from data_processing.preprocessing import HeartDiseasePreprocessor, load_and_split_data


@pytest.fixture
def sample_data():
    """
    Create sample data for testing
    """
    data = {
        'age': [63, 37, 41, 56, 57],
        'sex': [1, 1, 0, 1, 0],
        'cp': [3, 2, 1, 1, 0],
        'trestbps': [145, 130, 130, 120, 120],
        'chol': [233, 250, 204, 236, 354],
        'fbs': [1, 0, 0, 0, 0],
        'restecg': [0, 1, 0, 1, 1],
        'thalach': [150, 187, 172, 178, 163],
        'exang': [0, 0, 0, 0, 1],
        'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6],
        'slope': [0, 0, 2, 2, 2],
        'ca': [0, 0, 0, 0, 0],
        'thal': [1, 2, 2, 2, 2],
        'target': [1, 1, 1, 1, 1]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_missing():
    """
    Create sample data with missing values
    """
    data = {
        'age': [63, 37, 41, 56, 57],
        'sex': [1, 1, 0, 1, 0],
        'cp': [3, 2, 1, 1, 0],
        'trestbps': [145, 130, 130, 120, 120],
        'chol': [233, 250, 204, 236, 354],
        'fbs': [1, 0, 0, 0, 0],
        'restecg': [0, 1, 0, 1, 1],
        'thalach': [150, 187, 172, 178, 163],
        'exang': [0, 0, 0, 0, 1],
        'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6],
        'slope': [0, 0, 2, 2, 2],
        'ca': [0, np.nan, 0, 0, np.nan],
        'thal': [1, 2, np.nan, 2, 2],
        'target': [1, 1, 1, 1, 1]
    }
    return pd.DataFrame(data)


class TestHeartDiseasePreprocessor:
    """
    Test cases for HeartDiseasePreprocessor
    """

    def test_initialization(self):
        """
        Test preprocessor initialization
        """
        preprocessor = HeartDiseasePreprocessor()
        assert preprocessor.scaler is not None
        assert len(preprocessor.numerical_features) == 5
        assert len(preprocessor.categorical_features) == 8

    def test_clean_data(self, sample_data_with_missing):
        """
        Test data cleaning
        """
        preprocessor = HeartDiseasePreprocessor()
        df_clean = preprocessor.clean_data(sample_data_with_missing)

        # Check no missing values
        assert df_clean.isnull().sum().sum() == 0
        assert len(df_clean) == len(sample_data_with_missing)

    def test_engineer_features(self, sample_data):
        """
        Test feature engineering
        """
        preprocessor = HeartDiseasePreprocessor()
        df_engineered = preprocessor.engineer_features(sample_data)

        # Check new features created
        assert 'age_group' in df_engineered.columns
        assert 'bp_category' in df_engineered.columns
        assert 'chol_category' in df_engineered.columns

        # Check no NaN in new features
        assert df_engineered['age_group'].notna().all()

    def test_prepare_features(self, sample_data):
        """
        Test feature preparation and scaling
        """
        preprocessor = HeartDiseasePreprocessor()
        X, y = preprocessor.prepare_features(sample_data, fit=True)

        # Check shapes
        assert X.shape[0] == sample_data.shape[0]
        assert y.shape[0] == sample_data.shape[0]

        # Check scaling (mean should be close to 0, std close to 1)
        assert abs(X.mean().mean()) < 1
        assert abs(X.std().mean() - 1) < 1

    def test_preprocess_pipeline(self, sample_data_with_missing):
        """
        Test complete preprocessing pipeline
        """
        preprocessor = HeartDiseasePreprocessor()
        X, y = preprocessor.preprocess(sample_data_with_missing, fit=True)

        # Check no missing values
        assert X.isnull().sum().sum() == 0

        # Check shapes
        assert X.shape[0] == sample_data_with_missing.shape[0]
        assert y.shape[0] == sample_data_with_missing.shape[0]

        # Check feature names stored
        assert preprocessor.feature_names is not None
        assert len(preprocessor.feature_names) > 0

    def test_save_load_preprocessor(self, sample_data, tmp_path):
        """
        Test saving and loading preprocessor
        """
        # Create and fit preprocessor
        preprocessor1 = HeartDiseasePreprocessor()
        X1, y1 = preprocessor1.preprocess(sample_data, fit=True)

        # Save preprocessor
        save_path = tmp_path / "test_preprocessor.pkl"
        preprocessor1.save_preprocessor(save_path)

        # Load preprocessor
        preprocessor2 = HeartDiseasePreprocessor()
        preprocessor2.load_preprocessor(save_path)

        # Test on same data
        X2, y2 = preprocessor2.preprocess(sample_data, fit=False)

        # Check results are the same
        assert preprocessor1.feature_names == preprocessor2.feature_names
        np.testing.assert_array_almost_equal(X1.values, X2.values)


class TestDataLoading:
    """
    Test data loading and splitting
    """

    def test_load_and_split_data(self, sample_data, tmp_path):
        """
        Test data loading and splitting function
        """
        # Save sample data
        data_path = tmp_path / "test_data.csv"
        sample_data.to_csv(data_path, index=False)

        # Load and split
        X_train, X_test, y_train, y_test, preprocessor = load_and_split_data(
            data_path, test_size=0.2, random_state=42
        )

        # Check shapes
        assert X_train.shape[0] + X_test.shape[0] == len(sample_data)
        assert y_train.shape[0] + y_test.shape[0] == len(sample_data)

        # Check preprocessor
        assert preprocessor is not None
        assert preprocessor.feature_names is not None

    def test_target_conversion(self, tmp_path):
        """
        Test binary target conversion
        """
        # Create data with multi-class target (need enough samples for stratification)
        data = {
            'age': [63, 37, 41, 56, 57, 44, 52, 60],
            'sex': [1, 1, 0, 1, 0, 1, 0, 1],
            'cp': [3, 2, 1, 1, 0, 3, 2, 1],
            'trestbps': [145, 130, 130, 120, 120, 140, 135, 150],
            'chol': [233, 250, 204, 236, 354, 240, 220, 260],
            'fbs': [1, 0, 0, 0, 0, 1, 0, 1],
            'restecg': [0, 1, 0, 1, 1, 0, 1, 0],
            'thalach': [150, 187, 172, 178, 163, 155, 165, 145],
            'exang': [0, 0, 0, 0, 1, 0, 1, 0],
            'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6, 1.2, 2.0, 1.5],
            'slope': [0, 0, 2, 2, 2, 1, 1, 0],
            'ca': [0, 0, 0, 0, 0, 1, 0, 1],
            'thal': [1, 2, 2, 2, 2, 1, 2, 1],
            'target': [0, 2, 3, 1, 0, 2, 3, 1]  # Multi-class - will convert to [0,1,1,1,0,1,1,1]
        }
        df = pd.DataFrame(data)

        # Save
        data_path = tmp_path / "test_data.csv"
        df.to_csv(data_path, index=False)

        # Load and split
        X_train, X_test, y_train, y_test, preprocessor = load_and_split_data(
            data_path, test_size=0.25, random_state=42
        )

        # Check target is binary
        assert y_train.isin([0, 1]).all()
        assert y_test.isin([0, 1]).all()


class TestEdgeCases:
    """
    Test edge cases and error handling
    """

    def test_empty_dataframe(self):
        """
        Test with empty DataFrame
        """
        preprocessor = HeartDiseasePreprocessor()
        df_empty = pd.DataFrame()

        # Should handle gracefully
        with pytest.raises(Exception):
            preprocessor.preprocess(df_empty, fit=True)

    def test_missing_columns(self):
        """
        Test with missing required columns
        """
        preprocessor = HeartDiseasePreprocessor()
        df_incomplete = pd.DataFrame({'age': [63], 'sex': [1]})

        # Should raise error
        with pytest.raises(Exception):
            preprocessor.engineer_features(df_incomplete)

    def test_transform_before_fit(self, sample_data):
        """
        Test transform before fitting scaler
        """
        preprocessor = HeartDiseasePreprocessor()

        # Try to transform without fitting
        with pytest.raises(Exception):
            preprocessor.prepare_features(sample_data, fit=False)
