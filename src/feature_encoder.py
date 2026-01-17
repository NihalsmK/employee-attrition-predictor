"""
Feature Encoder for Employee Attrition Predictor

This module provides comprehensive feature encoding capabilities for HR datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import pickle
import yaml
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EncodingResult:
    """Result of feature encoding operations."""
    encoded_data: pd.DataFrame
    feature_names: List[str]
    encoding_info: Dict[str, Any]
    original_shape: Tuple[int, int]
    encoded_shape: Tuple[int, int]

class FeatureEncoder:
    """
    Comprehensive feature encoder for HR datasets.
    
    Handles ordinal encoding, one-hot encoding, binary encoding, and numerical scaling.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize the FeatureEncoder.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names_ = []
        self.encoding_strategy = self._get_encoding_strategy()
        
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
            'data_processing': {
                'categorical_columns': {
                    'ordinal': {
                        'JobSatisfaction': ['Low', 'Medium', 'High']
                    },
                    'nominal': ['Department', 'EducationField', 'JobRole'],
                    'binary': ['OverTime']
                },
                'numerical_columns': [
                    'Age', 'DistanceFromHome', 'MonthlyIncome', 
                    'NumCompaniesWorked', 'YearsAtCompany'
                ]
            }
        }
    
    def _get_encoding_strategy(self) -> Dict[str, str]:
        """Get encoding strategy for each column type."""
        strategy = {}
        
        # Ordinal columns
        if 'ordinal' in self.config['data_processing']['categorical_columns']:
            for col in self.config['data_processing']['categorical_columns']['ordinal']:
                strategy[col] = 'ordinal'
        
        # Nominal columns
        if 'nominal' in self.config['data_processing']['categorical_columns']:
            for col in self.config['data_processing']['categorical_columns']['nominal']:
                strategy[col] = 'onehot'
        
        # Binary columns
        if 'binary' in self.config['data_processing']['categorical_columns']:
            for col in self.config['data_processing']['categorical_columns']['binary']:
                strategy[col] = 'binary'
        
        # Numerical columns
        for col in self.config['data_processing']['numerical_columns']:
            strategy[col] = 'numerical'
        
        return strategy
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit encoders and transform the data.
        
        Args:
            df: DataFrame to encode
            
        Returns:
            Encoded DataFrame
        """
        logger.info("Fitting encoders and transforming data...")
        
        # Make a copy to avoid modifying original data
        df_encoded = df.copy()
        
        # Store original feature names
        self.original_columns = df.columns.tolist()
        
        # Encode each column based on its strategy
        for column, strategy in self.encoding_strategy.items():
            if column in df_encoded.columns:
                if strategy == 'ordinal':
                    df_encoded = self._fit_transform_ordinal(df_encoded, column)
                elif strategy == 'onehot':
                    df_encoded = self._fit_transform_onehot(df_encoded, column)
                elif strategy == 'binary':
                    df_encoded = self._fit_transform_binary(df_encoded, column)
                elif strategy == 'numerical':
                    df_encoded = self._fit_transform_numerical(df_encoded, column)
        
        # Store feature names
        self.feature_names_ = df_encoded.columns.tolist()
        self.is_fitted = True
        
        logger.info(f"Encoding completed. Shape: {df.shape} -> {df_encoded.shape}")
        return df_encoded
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted encoders.
        
        Args:
            df: DataFrame to encode
            
        Returns:
            Encoded DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Encoders must be fitted before transform. Use fit_transform() first.")
        
        logger.info("Transforming data using fitted encoders...")
        
        # Make a copy to avoid modifying original data
        df_encoded = df.copy()
        
        # Transform each column based on its strategy
        for column, strategy in self.encoding_strategy.items():
            if column in df_encoded.columns:
                if strategy == 'ordinal':
                    df_encoded = self._transform_ordinal(df_encoded, column)
                elif strategy == 'onehot':
                    df_encoded = self._transform_onehot(df_encoded, column)
                elif strategy == 'binary':
                    df_encoded = self._transform_binary(df_encoded, column)
                elif strategy == 'numerical':
                    df_encoded = self._transform_numerical(df_encoded, column)
        
        # Ensure consistent column order and presence
        df_encoded = self._align_columns(df_encoded)
        
        logger.info(f"Transform completed. Shape: {df.shape} -> {df_encoded.shape}")
        return df_encoded
    
    def _fit_transform_ordinal(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Fit and transform ordinal categorical column."""
        if column == 'JobSatisfaction':
            # Custom ordinal mapping for JobSatisfaction
            ordinal_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
            self.encoders[column] = ordinal_mapping
            df[column] = df[column].map(ordinal_mapping)
        else:
            # Generic ordinal encoding
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column].astype(str))
            self.encoders[column] = encoder
        
        return df
    
    def _transform_ordinal(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Transform ordinal categorical column using fitted encoder."""
        if isinstance(self.encoders[column], dict):
            # Custom mapping
            df[column] = df[column].map(self.encoders[column])
            # Handle unknown values
            df[column] = df[column].fillna(2)  # Default to 'Medium' for JobSatisfaction
        else:
            # LabelEncoder
            # Handle unknown values by mapping to most frequent class
            known_classes = self.encoders[column].classes_
            df[column] = df[column].astype(str)
            unknown_mask = ~df[column].isin(known_classes)
            if unknown_mask.any():
                df.loc[unknown_mask, column] = known_classes[0]  # Default to first class
            df[column] = self.encoders[column].transform(df[column])
        
        return df
    
    def _fit_transform_onehot(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Fit and transform nominal categorical column using one-hot encoding."""
        # Get unique values
        unique_values = df[column].unique()
        
        # Create one-hot encoded columns
        for value in unique_values:
            new_column = f"{column}_{value}"
            df[new_column] = (df[column] == value).astype(int)
        
        # Store encoder info
        self.encoders[column] = {
            'type': 'onehot',
            'categories': unique_values.tolist(),
            'columns': [f"{column}_{value}" for value in unique_values]
        }
        
        # Drop original column
        df = df.drop(columns=[column])
        
        return df
    
    def _transform_onehot(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Transform nominal categorical column using fitted one-hot encoder."""
        encoder_info = self.encoders[column]
        categories = encoder_info['categories']
        
        # Create one-hot encoded columns
        for value in categories:
            new_column = f"{column}_{value}"
            df[new_column] = (df[column] == value).astype(int)
        
        # Drop original column
        df = df.drop(columns=[column])
        
        return df
    
    def _fit_transform_binary(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Fit and transform binary categorical column."""
        # Map Yes/No to 1/0
        if column == 'OverTime':
            binary_mapping = {'Yes': 1, 'No': 0}
        else:
            # Generic binary mapping
            unique_values = df[column].unique()
            if len(unique_values) == 2:
                binary_mapping = {unique_values[0]: 0, unique_values[1]: 1}
            else:
                # Default mapping for non-binary
                binary_mapping = {val: i for i, val in enumerate(unique_values)}
        
        self.encoders[column] = binary_mapping
        df[column] = df[column].map(binary_mapping)
        
        return df
    
    def _transform_binary(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Transform binary categorical column using fitted encoder."""
        binary_mapping = self.encoders[column]
        df[column] = df[column].map(binary_mapping)
        
        # Handle unknown values
        df[column] = df[column].fillna(0)  # Default to 0 for unknown values
        
        return df
    
    def _fit_transform_numerical(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Fit and transform numerical column using standardization."""
        # Store original values for potential inverse transform
        if column not in self.encoders:
            self.encoders[column] = {
                'type': 'numerical',
                'mean': df[column].mean(),
                'std': df[column].std()
            }
        
        # Standardize the column
        df[column] = (df[column] - self.encoders[column]['mean']) / self.encoders[column]['std']
        
        return df
    
    def _transform_numerical(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Transform numerical column using fitted scaler."""
        encoder_info = self.encoders[column]
        df[column] = (df[column] - encoder_info['mean']) / encoder_info['std']
        
        return df
    
    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure transformed DataFrame has consistent columns with training data."""
        # Add missing columns with zeros
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0
        
        # Remove extra columns and reorder
        df = df[self.feature_names_]
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get names of encoded features."""
        if not self.is_fitted:
            raise ValueError("Encoders must be fitted before getting feature names.")
        return self.feature_names_.copy()
    
    def save_encoders(self, path: str) -> None:
        """
        Save fitted encoders to disk.
        
        Args:
            path: Path to save encoders
        """
        if not self.is_fitted:
            raise ValueError("Encoders must be fitted before saving.")
        
        encoder_data = {
            'encoders': self.encoders,
            'feature_names': self.feature_names_,
            'original_columns': self.original_columns,
            'encoding_strategy': self.encoding_strategy,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(encoder_data, f)
        
        logger.info(f"Encoders saved to {path}")
    
    def load_encoders(self, path: str) -> None:
        """
        Load fitted encoders from disk.
        
        Args:
            path: Path to load encoders from
        """
        with open(path, 'rb') as f:
            encoder_data = pickle.load(f)
        
        self.encoders = encoder_data['encoders']
        self.feature_names_ = encoder_data['feature_names']
        self.original_columns = encoder_data['original_columns']
        self.encoding_strategy = encoder_data['encoding_strategy']
        self.is_fitted = encoder_data['is_fitted']
        
        logger.info(f"Encoders loaded from {path}")
    
    def get_encoding_info(self) -> Dict[str, Any]:
        """Get information about the encoding process."""
        if not self.is_fitted:
            raise ValueError("Encoders must be fitted before getting encoding info.")
        
        info = {
            'original_columns': self.original_columns,
            'encoded_columns': self.feature_names_,
            'encoding_strategy': self.encoding_strategy,
            'num_original_features': len(self.original_columns),
            'num_encoded_features': len(self.feature_names_),
            'encoders_info': {}
        }
        
        # Add detailed encoder information
        for column, encoder in self.encoders.items():
            if isinstance(encoder, dict):
                if encoder.get('type') == 'onehot':
                    info['encoders_info'][column] = {
                        'type': 'one-hot',
                        'categories': encoder['categories'],
                        'num_categories': len(encoder['categories'])
                    }
                elif encoder.get('type') == 'numerical':
                    info['encoders_info'][column] = {
                        'type': 'standardization',
                        'mean': encoder['mean'],
                        'std': encoder['std']
                    }
                else:
                    info['encoders_info'][column] = {
                        'type': 'mapping',
                        'mapping': encoder
                    }
            else:
                info['encoders_info'][column] = {
                    'type': 'label_encoder',
                    'classes': encoder.classes_.tolist() if hasattr(encoder, 'classes_') else 'unknown'
                }
        
        return info

if __name__ == "__main__":
    # Test the FeatureEncoder with sample data
    encoder = FeatureEncoder()
    
    # Load sample data
    df = pd.read_csv('data/hr_employee_data.csv')
    
    # Remove target column for encoding
    X = df.drop('Attrition', axis=1)
    
    # Fit and transform
    X_encoded = encoder.fit_transform(X)
    
    print("=== Feature Encoding Results ===")
    print(f"Original shape: {X.shape}")
    print(f"Encoded shape: {X_encoded.shape}")
    print(f"Original columns: {len(X.columns)}")
    print(f"Encoded columns: {len(X_encoded.columns)}")
    
    # Show encoding info
    encoding_info = encoder.get_encoding_info()
    print(f"\nEncoding strategies used:")
    for col, strategy in encoding_info['encoding_strategy'].items():
        print(f"  {col}: {strategy}")
    
    # Test transform on new data (first 5 rows)
    X_test = X.head(5)
    X_test_encoded = encoder.transform(X_test)
    print(f"\nTest transform shape: {X_test.shape} -> {X_test_encoded.shape}")
    
    # Save encoders
    encoder.save_encoders('models/feature_encoders.pkl')
    print("\nEncoders saved successfully!")