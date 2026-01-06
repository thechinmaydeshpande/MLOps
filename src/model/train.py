"""
Model training with MLflow experiment tracking
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import pickle

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from data_processing.preprocessing import load_and_split_data


class ModelTrainer:
    """
    Train and evaluate multiple classification models
    """

    def __init__(self, experiment_name="heart_disease_prediction"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.models = {}
        self.results = {}

    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """
        Train Logistic Regression with hyperparameter tuning
        """
        with mlflow.start_run(run_name="Logistic_Regression"):
            # Hyperparameter grid
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs'],
                'max_iter': [1000]
            }

            # Grid search with cross-validation
            lr = LogisticRegression(random_state=42)
            grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc',
                                      n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            # Best model
            best_model = grid_search.best_estimator_

            # Log parameters
            mlflow.log_params(grid_search.best_params_)

            # Evaluate
            metrics = self.evaluate_model(best_model, X_train, y_train, X_test, y_test,
                                         model_name="Logistic_Regression")

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model
            mlflow.sklearn.log_model(best_model, "model")

            # Save feature importance (coefficients)
            self.plot_feature_importance(best_model.coef_[0], X_train.columns,
                                        "Logistic_Regression")

            self.models['Logistic_Regression'] = best_model
            self.results['Logistic_Regression'] = metrics

            print(f"\nLogistic Regression - Best Parameters: {grid_search.best_params_}")
            print(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")

            return best_model, metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Train Random Forest with hyperparameter tuning
        """
        with mlflow.start_run(run_name="Random_Forest"):
            # Hyperparameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            # Grid search with cross-validation
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc',
                                      n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            # Best model
            best_model = grid_search.best_estimator_

            # Log parameters
            mlflow.log_params(grid_search.best_params_)

            # Evaluate
            metrics = self.evaluate_model(best_model, X_train, y_train, X_test, y_test,
                                         model_name="Random_Forest")

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model
            mlflow.sklearn.log_model(best_model, "model")

            # Save feature importance
            self.plot_feature_importance(best_model.feature_importances_, X_train.columns,
                                        "Random_Forest")

            self.models['Random_Forest'] = best_model
            self.results['Random_Forest'] = metrics

            print(f"\nRandom Forest - Best Parameters: {grid_search.best_params_}")
            print(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")

            return best_model, metrics

    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """
        Train Gradient Boosting Classifier
        """
        with mlflow.start_run(run_name="Gradient_Boosting"):
            # Hyperparameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }

            # Grid search with cross-validation
            gb = GradientBoostingClassifier(random_state=42)
            grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='roc_auc',
                                      n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            # Best model
            best_model = grid_search.best_estimator_

            # Log parameters
            mlflow.log_params(grid_search.best_params_)

            # Evaluate
            metrics = self.evaluate_model(best_model, X_train, y_train, X_test, y_test,
                                         model_name="Gradient_Boosting")

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model
            mlflow.sklearn.log_model(best_model, "model")

            # Save feature importance
            self.plot_feature_importance(best_model.feature_importances_, X_train.columns,
                                        "Gradient_Boosting")

            self.models['Gradient_Boosting'] = best_model
            self.results['Gradient_Boosting'] = metrics

            print(f"\nGradient Boosting - Best Parameters: {grid_search.best_params_}")
            print(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")

            return best_model, metrics

    def evaluate_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """
        Comprehensive model evaluation
        """
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Probabilities for ROC-AUC
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'train_precision': precision_score(y_train, y_train_pred),
            'test_precision': precision_score(y_test, y_test_pred),
            'train_recall': recall_score(y_train, y_train_pred),
            'test_recall': recall_score(y_test, y_test_pred),
            'train_f1': f1_score(y_train, y_train_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'train_roc_auc': roc_auc_score(y_train, y_train_proba),
            'test_roc_auc': roc_auc_score(y_test, y_test_proba)
        }

        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        metrics['cv_roc_auc_mean'] = cv_scores.mean()
        metrics['cv_roc_auc_std'] = cv_scores.std()

        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_test_pred, model_name)

        # Plot ROC curve
        self.plot_roc_curve(y_test, y_test_proba, model_name)

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """
        Plot and save confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        # Save plot
        save_path = Path(__file__).parent.parent.parent / 'screenshots' / f'confusion_matrix_{model_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(save_path)
        plt.close()

    def plot_roc_curve(self, y_true, y_proba, model_name):
        """
        Plot and save ROC curve
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Save plot
        save_path = Path(__file__).parent.parent.parent / 'screenshots' / f'roc_curve_{model_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(save_path)
        plt.close()

    def plot_feature_importance(self, importances, feature_names, model_name):
        """
        Plot and save feature importance
        """
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(importances)
        }).sort_values('importance', ascending=False).head(15)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
        plt.title(f'Top 15 Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()

        # Save plot
        save_path = Path(__file__).parent.parent.parent / 'screenshots' / f'feature_importance_{model_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(save_path)
        plt.close()

    def compare_models(self):
        """
        Compare all trained models
        """
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df[['test_accuracy', 'test_precision', 'test_recall',
                                       'test_f1', 'test_roc_auc', 'cv_roc_auc_mean']]

        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(comparison_df.to_string())
        print("="*80)

        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Test metrics comparison
        metrics_to_plot = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']
        comparison_df[metrics_to_plot].plot(kind='bar', ax=axes[0])
        axes[0].set_title('Model Comparison - Test Metrics', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Score')
        axes[0].set_xlabel('Models')
        axes[0].legend(loc='lower right')
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)

        # ROC-AUC comparison
        roc_data = comparison_df['test_roc_auc'].sort_values(ascending=False)
        roc_data.plot(kind='barh', ax=axes[1], color='skyblue')
        axes[1].set_title('Model Comparison - ROC-AUC Score', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('ROC-AUC Score')
        axes[1].set_ylabel('Models')
        axes[1].set_xlim([0, 1])
        axes[1].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        save_path = Path(__file__).parent.parent.parent / 'screenshots' / 'model_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return comparison_df

    def save_best_model(self, save_dir):
        """
        Save the best performing model
        """
        # Find best model based on test ROC-AUC
        best_model_name = max(self.results, key=lambda k: self.results[k]['test_roc_auc'])
        best_model = self.models[best_model_name]

        save_path = Path(save_dir) / 'best_model.pkl'
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(best_model, f)

        print(f"\nBest model ({best_model_name}) saved to {save_path}")
        print(f"Test ROC-AUC: {self.results[best_model_name]['test_roc_auc']:.4f}")

        # Save model info
        model_info = {
            'model_name': best_model_name,
            'metrics': self.results[best_model_name]
        }

        info_path = Path(save_dir) / 'best_model_info.pkl'
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)

        return best_model, best_model_name


def main():
    """
    Main training pipeline
    """
    print("="*80)
    print("HEART DISEASE PREDICTION - MODEL TRAINING")
    print("="*80)

    # Load and split data
    data_path = Path(__file__).parent.parent.parent / 'data' / 'heart_disease_raw.csv'
    print(f"\nLoading data from {data_path}")

    X_train, X_test, y_train, y_test, preprocessor = load_and_split_data(data_path)

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")

    # Save preprocessor
    preprocessor_path = Path(__file__).parent.parent.parent / 'models' / 'preprocessor.pkl'
    preprocessor.save_preprocessor(preprocessor_path)

    # Initialize trainer
    trainer = ModelTrainer()

    # Train models
    print("\n" + "="*80)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*80)
    trainer.train_logistic_regression(X_train, y_train, X_test, y_test)

    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST")
    print("="*80)
    trainer.train_random_forest(X_train, y_train, X_test, y_test)

    print("\n" + "="*80)
    print("TRAINING GRADIENT BOOSTING")
    print("="*80)
    trainer.train_gradient_boosting(X_train, y_train, X_test, y_test)

    # Compare models
    comparison_df = trainer.compare_models()

    # Save best model
    models_dir = Path(__file__).parent.parent.parent / 'models'
    best_model, best_model_name = trainer.save_best_model(models_dir)

    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()
