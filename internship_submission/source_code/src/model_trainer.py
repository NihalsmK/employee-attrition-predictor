"""
Model Trainer for Employee Attrition Predictor

This module provides comprehensive machine learning model training capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import pickle
import yaml
import logging
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    feature_importance: Dict[str, float]
    cross_val_scores: List[float]
    confusion_matrix: np.ndarray
    classification_report: str

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    train_metrics: ModelPerformance
    test_metrics: ModelPerformance
    overfitting_score: float
    model_complexity: Dict[str, Any]

@dataclass
class ComparisonReport:
    """Model comparison report."""
    models: List[str]
    metrics_comparison: pd.DataFrame
    best_model: str
    best_metric: str
    recommendations: List[str]

@dataclass
class FeatureImportance:
    """Feature importance analysis."""
    feature_names: List[str]
    importance_scores: List[float]
    importance_ranking: List[Tuple[str, float]]
    top_features: List[str]
    visualization: Optional[plt.Figure]

class BaseModel:
    """Base class for all models."""
    
    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.is_fitted = False
        self.feature_names = []
        self.training_time = None
        self.metadata = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model."""
        start_time = datetime.now()
        self.model.fit(X, y)
        self.training_time = (datetime.now() - start_time).total_seconds()
        self.is_fitted = True
        self.feature_names = X.columns.tolist()
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return binary predictions
            predictions = self.predict(X)
            proba = np.zeros((len(predictions), 2))
            proba[predictions == 0, 0] = 1
            proba[predictions == 1, 1] = 1
            return proba
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        else:
            return None

class LogisticModel(BaseModel):
    """Logistic Regression model wrapper."""
    
    def __init__(self, **kwargs):
        model = LogisticRegression(**kwargs)
        super().__init__(model, "Logistic Regression")
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients with interpretation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting coefficients")
        
        coefficients = {}
        for feature, coef in zip(self.feature_names, self.model.coef_[0]):
            coefficients[feature] = coef
        
        return coefficients
    
    def interpret_coefficients(self) -> Dict[str, str]:
        """Interpret coefficients for business understanding."""
        coefficients = self.get_coefficients()
        interpretations = {}
        
        for feature, coef in coefficients.items():
            if coef > 0:
                interpretations[feature] = f"Increases attrition risk (coef: {coef:.3f})"
            else:
                interpretations[feature] = f"Decreases attrition risk (coef: {coef:.3f})"
        
        return interpretations

class RandomForestModel(BaseModel):
    """Random Forest model wrapper."""
    
    def __init__(self, **kwargs):
        model = RandomForestClassifier(**kwargs)
        super().__init__(model, "Random Forest")
    
    def get_tree_count(self) -> int:
        """Get number of trees in the forest."""
        return self.model.n_estimators
    
    def get_feature_interactions(self) -> Dict[str, Any]:
        """Analyze feature interactions (simplified)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analyzing interactions")
        
        # Get feature importance
        importance = self.get_feature_importance()
        
        # Return top interacting features (simplified analysis)
        top_indices = np.argsort(importance)[-5:]  # Top 5 features
        top_features = [self.feature_names[i] for i in top_indices]
        
        return {
            'top_interacting_features': top_features,
            'interaction_strength': importance[top_indices].tolist(),
            'note': 'Simplified interaction analysis based on feature importance'
        }

class DecisionTreeModel(BaseModel):
    """Decision Tree model wrapper."""
    
    def __init__(self, **kwargs):
        model = DecisionTreeClassifier(**kwargs)
        super().__init__(model, "Decision Tree")
    
    def get_tree_depth(self) -> int:
        """Get actual tree depth."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting tree depth")
        return self.model.tree_.max_depth
    
    def get_decision_paths(self, X: pd.DataFrame, max_samples: int = 5) -> List[str]:
        """Get interpretable decision paths for sample predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting decision paths")
        
        # Get decision paths for first few samples
        sample_X = X.head(max_samples)
        decision_paths = self.model.decision_path(sample_X)
        
        paths = []
        for i in range(min(max_samples, len(sample_X))):
            path_nodes = decision_paths[i].indices
            path_description = f"Sample {i+1}: "
            
            # Simplified path description
            if len(path_nodes) > 0:
                path_description += f"Follows {len(path_nodes)} decision nodes"
            else:
                path_description += "Direct classification"
            
            paths.append(path_description)
        
        return paths

class ModelTrainer:
    """
    Comprehensive model trainer for employee attrition prediction.
    
    Supports multiple algorithms with evaluation and comparison capabilities.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize the ModelTrainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.is_target_encoded = False
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file is not found."""
        return {
            'model_training': {
                'test_size': 0.2,
                'random_state': 42,
                'cross_validation_folds': 5,
                'algorithms': {
                    'logistic_regression': {
                        'max_iter': 1000,
                        'random_state': 42
                    },
                    'random_forest': {
                        'n_estimators': 100,
                        'max_depth': 10,
                        'random_state': 42
                    },
                    'decision_tree': {
                        'max_depth': 8,
                        'random_state': 42
                    }
                }
            }
        }
    
    def _encode_target(self, y: pd.Series) -> np.ndarray:
        """Encode target variable if needed."""
        if y.dtype == 'object':
            if not self.is_target_encoded:
                y_encoded = self.label_encoder.fit_transform(y)
                self.is_target_encoded = True
            else:
                y_encoded = self.label_encoder.transform(y)
            return y_encoded
        return y.values
    
    def train_logistic_regression(self, X: pd.DataFrame, y: pd.Series) -> LogisticModel:
        """
        Train logistic regression model with coefficient interpretation.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Trained LogisticModel
        """
        logger.info("Training Logistic Regression model...")
        
        # Encode target if needed
        y_encoded = self._encode_target(y)
        
        # Get configuration
        lr_config = self.config['model_training']['algorithms']['logistic_regression']
        
        # Create and train model
        model = LogisticModel(**lr_config)
        model.fit(X, y_encoded)
        
        # Store model
        self.models['logistic_regression'] = model
        
        logger.info("Logistic Regression training completed.")
        return model
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> RandomForestModel:
        """
        Train random forest model for non-linear relationship capture.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Trained RandomForestModel
        """
        logger.info("Training Random Forest model...")
        
        # Encode target if needed
        y_encoded = self._encode_target(y)
        
        # Get configuration
        rf_config = self.config['model_training']['algorithms']['random_forest']
        
        # Create and train model
        model = RandomForestModel(**rf_config)
        model.fit(X, y_encoded)
        
        # Store model
        self.models['random_forest'] = model
        
        logger.info("Random Forest training completed.")
        return model
    
    def train_decision_tree(self, X: pd.DataFrame, y: pd.Series) -> DecisionTreeModel:
        """
        Train decision tree model with interpretable decision paths.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Trained DecisionTreeModel
        """
        logger.info("Training Decision Tree model...")
        
        # Encode target if needed
        y_encoded = self._encode_target(y)
        
        # Get configuration
        dt_config = self.config['model_training']['algorithms']['decision_tree']
        
        # Create and train model
        model = DecisionTreeModel(**dt_config)
        model.fit(X, y_encoded)
        
        # Store model
        self.models['decision_tree'] = model
        
        logger.info("Decision Tree training completed.")
        return model
    
    def evaluate_model(self, model: BaseModel, X_test: pd.DataFrame, y_test: pd.Series) -> ModelPerformance:
        """
        Evaluate model performance with comprehensive metrics.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test target
            
        Returns:
            ModelPerformance with evaluation metrics
        """
        logger.info(f"Evaluating {model.model_name} model...")
        
        # Encode target if needed
        y_test_encoded = self._encode_target(y_test)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred, average='weighted')
        recall = recall_score(y_test_encoded, y_pred, average='weighted')
        f1 = f1_score(y_test_encoded, y_pred, average='weighted')
        
        # AUC score (for binary classification)
        if len(np.unique(y_test_encoded)) == 2:
            auc = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
        else:
            auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr', average='weighted')
        
        # Feature importance
        importance = model.get_feature_importance()
        if importance is not None:
            feature_importance = dict(zip(model.feature_names, importance))
        else:
            feature_importance = {}
        
        # Cross-validation scores
        cv_scores = self._get_cross_validation_scores(model, X_test, y_test_encoded)
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        # Classification report
        class_report = classification_report(y_test_encoded, y_pred)
        
        return ModelPerformance(
            model_name=model.model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            feature_importance=feature_importance,
            cross_val_scores=cv_scores,
            confusion_matrix=cm,
            classification_report=class_report
        )
    
    def _get_cross_validation_scores(self, model: BaseModel, X: pd.DataFrame, y: np.ndarray) -> List[float]:
        """Get cross-validation scores for the model."""
        try:
            cv_folds = self.config['model_training']['cross_validation_folds']
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model.model, X, y, cv=cv, scoring='accuracy')
            return scores.tolist()
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return []
    
    def compare_models(self, models: List[BaseModel], X_test: pd.DataFrame, y_test: pd.Series) -> ComparisonReport:
        """
        Compare multiple models with ROC curves and performance metrics.
        
        Args:
            models: List of trained models to compare
            X_test: Test features
            y_test: Test target
            
        Returns:
            ComparisonReport with comparison results
        """
        logger.info("Comparing models...")
        
        # Evaluate all models
        model_performances = []
        for model in models:
            performance = self.evaluate_model(model, X_test, y_test)
            model_performances.append(performance)
        
        # Create comparison DataFrame
        comparison_data = []
        for perf in model_performances:
            comparison_data.append({
                'Model': perf.model_name,
                'Accuracy': perf.accuracy,
                'Precision': perf.precision,
                'Recall': perf.recall,
                'F1-Score': perf.f1_score,
                'AUC': perf.auc_score,
                'CV_Mean': np.mean(perf.cross_val_scores) if perf.cross_val_scores else 0,
                'CV_Std': np.std(perf.cross_val_scores) if perf.cross_val_scores else 0
            })
        
        metrics_df = pd.DataFrame(comparison_data)
        
        # Determine best model (based on F1-score)
        best_idx = metrics_df['F1-Score'].idxmax()
        best_model = metrics_df.loc[best_idx, 'Model']
        
        # Generate recommendations
        recommendations = self._generate_model_recommendations(metrics_df, model_performances)
        
        # Create ROC curve comparison
        self._plot_roc_comparison(models, X_test, y_test)
        
        return ComparisonReport(
            models=[perf.model_name for perf in model_performances],
            metrics_comparison=metrics_df,
            best_model=best_model,
            best_metric='F1-Score',
            recommendations=recommendations
        )
    
    def _generate_model_recommendations(self, metrics_df: pd.DataFrame, 
                                      performances: List[ModelPerformance]) -> List[str]:
        """Generate recommendations based on model comparison."""
        recommendations = []
        
        # Best overall model
        best_model = metrics_df.loc[metrics_df['F1-Score'].idxmax(), 'Model']
        recommendations.append(f"Best overall model: {best_model}")
        
        # Interpretability recommendation
        if 'Logistic Regression' in metrics_df['Model'].values:
            lr_f1 = metrics_df[metrics_df['Model'] == 'Logistic Regression']['F1-Score'].iloc[0]
            best_f1 = metrics_df['F1-Score'].max()
            if abs(lr_f1 - best_f1) < 0.05:  # Within 5% of best
                recommendations.append("Logistic Regression recommended for interpretability with minimal performance loss")
        
        # Overfitting check
        for perf in performances:
            if perf.cross_val_scores:
                cv_mean = np.mean(perf.cross_val_scores)
                if abs(perf.accuracy - cv_mean) > 0.1:  # 10% difference
                    recommendations.append(f"{perf.model_name} may be overfitting (train-CV gap: {abs(perf.accuracy - cv_mean):.3f})")
        
        # Performance insights
        high_precision_models = metrics_df[metrics_df['Precision'] > 0.8]['Model'].tolist()
        if high_precision_models:
            recommendations.append(f"High precision models (low false positives): {', '.join(high_precision_models)}")
        
        high_recall_models = metrics_df[metrics_df['Recall'] > 0.8]['Model'].tolist()
        if high_recall_models:
            recommendations.append(f"High recall models (low false negatives): {', '.join(high_recall_models)}")
        
        return recommendations
    
    def _plot_roc_comparison(self, models: List[BaseModel], X_test: pd.DataFrame, y_test: pd.Series):
        """Plot ROC curves for model comparison."""
        plt.figure(figsize=(10, 8))
        
        y_test_encoded = self._encode_target(y_test)
        
        for model in models:
            if len(np.unique(y_test_encoded)) == 2:  # Binary classification
                y_pred_proba = model.predict_proba(X_test)
                fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_proba[:, 1])
                auc = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
                plt.plot(fpr, tpr, label=f'{model.model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reports/roc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def extract_feature_importance(self, model: BaseModel, top_n: int = 10) -> FeatureImportance:
        """
        Extract and analyze feature importance for business insights.
        
        Args:
            model: Trained model
            top_n: Number of top features to return
            
        Returns:
            FeatureImportance with analysis results
        """
        logger.info(f"Extracting feature importance from {model.model_name}...")
        
        importance_scores = model.get_feature_importance()
        if importance_scores is None:
            logger.warning(f"Feature importance not available for {model.model_name}")
            return FeatureImportance(
                feature_names=[],
                importance_scores=[],
                importance_ranking=[],
                top_features=[],
                visualization=None
            )
        
        # Create feature importance ranking
        feature_importance_pairs = list(zip(model.feature_names, importance_scores))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top features
        top_features = [pair[0] for pair in feature_importance_pairs[:top_n]]
        
        # Create visualization
        fig = self._plot_feature_importance(feature_importance_pairs[:top_n], model.model_name)
        
        return FeatureImportance(
            feature_names=model.feature_names,
            importance_scores=importance_scores.tolist(),
            importance_ranking=feature_importance_pairs,
            top_features=top_features,
            visualization=fig
        )
    
    def _plot_feature_importance(self, importance_pairs: List[Tuple[str, float]], 
                                model_name: str) -> plt.Figure:
        """Plot feature importance visualization."""
        features, scores = zip(*importance_pairs)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(features))
        
        bars = ax.barh(y_pos, scores)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Feature Importance - {model_name}')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', va='center', ha='left')
        
        plt.tight_layout()
        return fig
    
    def save_model(self, model: BaseModel, filepath: str):
        """Save trained model to disk."""
        model_data = {
            'model': model.model,
            'model_name': model.model_name,
            'feature_names': model.feature_names,
            'training_time': model.training_time,
            'metadata': model.metadata,
            'label_encoder': self.label_encoder if self.is_target_encoded else None,
            'is_target_encoded': self.is_target_encoded
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> BaseModel:
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Recreate model wrapper
        if 'Logistic' in model_data['model_name']:
            model = LogisticModel()
        elif 'Random Forest' in model_data['model_name']:
            model = RandomForestModel()
        elif 'Decision Tree' in model_data['model_name']:
            model = DecisionTreeModel()
        else:
            model = BaseModel(model_data['model'], model_data['model_name'])
        
        # Restore model state
        model.model = model_data['model']
        model.feature_names = model_data['feature_names']
        model.training_time = model_data['training_time']
        model.metadata = model_data['metadata']
        model.is_fitted = True
        
        # Restore label encoder
        if model_data['is_target_encoded']:
            self.label_encoder = model_data['label_encoder']
            self.is_target_encoded = True
        
        logger.info(f"Model loaded from {filepath}")
        return model

if __name__ == "__main__":
    # Test the ModelTrainer with sample data
    from feature_encoder import FeatureEncoder
    
    trainer = ModelTrainer()
    
    # Load and prepare data
    df = pd.read_csv('data/hr_employee_data.csv')
    
    # Encode features
    encoder = FeatureEncoder()
    X = encoder.fit_transform(df.drop('Attrition', axis=1))
    y = df['Attrition']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("=== Model Training Test ===")
    
    # Train models
    lr_model = trainer.train_logistic_regression(X_train, y_train)
    rf_model = trainer.train_random_forest(X_train, y_train)
    dt_model = trainer.train_decision_tree(X_train, y_train)
    
    # Evaluate models
    models = [lr_model, rf_model, dt_model]
    comparison = trainer.compare_models(models, X_test, y_test)
    
    print(f"\nModel Comparison Results:")
    print(comparison.metrics_comparison)
    print(f"\nBest Model: {comparison.best_model}")
    print(f"\nRecommendations:")
    for rec in comparison.recommendations:
        print(f"  - {rec}")
    
    # Feature importance
    importance = trainer.extract_feature_importance(rf_model)
    print(f"\nTop 5 Important Features (Random Forest):")
    for i, (feature, score) in enumerate(importance.importance_ranking[:5]):
        print(f"  {i+1}. {feature}: {score:.3f}")
    
    # Save best model
    best_model_name = comparison.best_model.lower().replace(' ', '_')
    best_model = next(m for m in models if m.model_name == comparison.best_model)
    trainer.save_model(best_model, f'models/{best_model_name}_model.pkl')
    
    print(f"\nBest model saved as models/{best_model_name}_model.pkl")