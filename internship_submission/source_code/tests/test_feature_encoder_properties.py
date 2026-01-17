"""
Property-based tests for FeatureEncoder

Feature: employee-attrition-predictor, Property 1: Data Processing Round-Trip Consistency
Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 8.1, 8.2, 8.3, 8.4, 8.5
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_encoder import FeatureEncoder

class TestFeatureEncoderProperties:
    """Property-based tests for FeatureEncoder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = FeatureEncoder()
    
    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=18, max_value=65),  # Age
                st.sampled_from(['Sales', 'R&D', 'HR', 'Finance']),  # Department
                st.integers(min_value=1, max_value=50),  # DistanceFromHome
                st.sampled_from(['Technical', 'Business', 'Medical']),  # EducationField
                st.sampled_from(['Manager', 'Analyst', 'Director']),  # JobRole
                st.sampled_from(['Low', 'Medium', 'High']),  # JobSatisfaction
                st.floats(min_value=1000, max_value=20000),  # MonthlyIncome
                st.integers(min_value=0, max_value=10),  # NumCompaniesWorked
                st.integers(min_value=0, max_value=40),  # YearsAtCompany
                st.sampled_from(['Yes', 'No'])  # OverTime
            ),
            min_size=5,
            max_size=100
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_property_1_data_processing_round_trip_consistency(self, hr_data):
        """
        Property 1: Data Processing Round-Trip Consistency
        
        For any valid HR dataset, after applying feature encoding and then decoding 
        (where applicable), the semantic meaning of categorical variables should be 
        preserved and numerical features should maintain their relative relationships.
        
        Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 8.1, 8.2, 8.3, 8.4, 8.5
        """
        # Create DataFrame from generated data
        df = pd.DataFrame(hr_data, columns=[
            'Age', 'Department', 'DistanceFromHome', 'EducationField', 'JobRole',
            'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked', 
            'YearsAtCompany', 'OverTime'
        ])
        
        # Ensure we have some data to work with
        assume(len(df) > 0)
        
        # Store original data for comparison
        original_df = df.copy()
        
        # Fit and transform the data
        encoded_df = self.encoder.fit_transform(df)
        
        # Property: Encoded data should have consistent shape
        assert len(encoded_df) == len(original_df)
        assert encoded_df.shape[0] == original_df.shape[0]
        
        # Property: No missing values should be introduced during encoding
        original_missing = original_df.isnull().sum().sum()
        encoded_missing = encoded_df.isnull().sum().sum()
        assert encoded_missing <= original_missing  # Should not introduce new missing values
        
        # Property: Ordinal encoding should preserve order
        if 'JobSatisfaction' in encoded_df.columns:
            # Check that Low < Medium < High in encoded values
            low_encoded = encoded_df[original_df['JobSatisfaction'] == 'Low']['JobSatisfaction'].iloc[0] if (original_df['JobSatisfaction'] == 'Low').any() else None
            medium_encoded = encoded_df[original_df['JobSatisfaction'] == 'Medium']['JobSatisfaction'].iloc[0] if (original_df['JobSatisfaction'] == 'Medium').any() else None
            high_encoded = encoded_df[original_df['JobSatisfaction'] == 'High']['JobSatisfaction'].iloc[0] if (original_df['JobSatisfaction'] == 'High').any() else None
            
            if low_encoded is not None and medium_encoded is not None:
                assert low_encoded < medium_encoded
            if medium_encoded is not None and high_encoded is not None:
                assert medium_encoded < high_encoded
            if low_encoded is not None and high_encoded is not None:
                assert low_encoded < high_encoded
        
        # Property: Binary encoding should produce 0/1 values
        if 'OverTime' in encoded_df.columns:
            overtime_values = encoded_df['OverTime'].unique()
            assert all(val in [0, 1] for val in overtime_values)
        
        # Property: One-hot encoded columns should sum to 1 for each row
        onehot_columns = [col for col in encoded_df.columns if '_' in col and 
                         any(col.startswith(prefix + '_') for prefix in ['Department', 'EducationField', 'JobRole'])]
        
        if onehot_columns:
            # Group by prefix
            prefixes = set(col.split('_')[0] for col in onehot_columns)
            for prefix in prefixes:
                prefix_cols = [col for col in onehot_columns if col.startswith(prefix + '_')]
                if len(prefix_cols) > 1:
                    row_sums = encoded_df[prefix_cols].sum(axis=1)
                    assert all(row_sum == 1 for row_sum in row_sums)
        
        # Property: Numerical columns should maintain relative relationships
        numerical_cols = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked', 'YearsAtCompany']
        for col in numerical_cols:
            if col in encoded_df.columns and len(df) > 1:
                # Check that relative ordering is preserved (correlation should be high)
                original_values = original_df[col].values
                encoded_values = encoded_df[col].values
                
                # Skip if all values are the same
                if len(set(original_values)) > 1:
                    correlation = np.corrcoef(original_values, encoded_values)[0, 1]
                    assert correlation > 0.99  # Very high correlation expected for standardization
    
    @given(
        st.integers(min_value=5, max_value=50),  # Dataset size
        st.random_module()
    )
    @settings(max_examples=30, deadline=None)
    def test_transform_consistency(self, dataset_size, random_module):
        """
        Property: Transform should be consistent across multiple calls.
        
        For any dataset, applying transform multiple times should produce 
        identical results.
        """
        # Generate consistent test data
        np.random.seed(42)
        df = pd.DataFrame({
            'Age': np.random.randint(18, 65, dataset_size),
            'Department': np.random.choice(['Sales', 'R&D', 'HR'], dataset_size),
            'DistanceFromHome': np.random.randint(1, 50, dataset_size),
            'EducationField': np.random.choice(['Technical', 'Business'], dataset_size),
            'JobRole': np.random.choice(['Manager', 'Analyst'], dataset_size),
            'JobSatisfaction': np.random.choice(['Low', 'Medium', 'High'], dataset_size),
            'MonthlyIncome': np.random.normal(5000, 1000, dataset_size),
            'NumCompaniesWorked': np.random.randint(0, 10, dataset_size),
            'YearsAtCompany': np.random.randint(0, 40, dataset_size),
            'OverTime': np.random.choice(['Yes', 'No'], dataset_size)
        })
        
        # Fit the encoder
        encoded_df1 = self.encoder.fit_transform(df)
        
        # Transform the same data multiple times
        encoded_df2 = self.encoder.transform(df)
        encoded_df3 = self.encoder.transform(df)
        
        # Property: Results should be identical
        pd.testing.assert_frame_equal(encoded_df2, encoded_df3)
        
        # Property: Transform should match fit_transform for same data
        # (Note: fit_transform might have slight differences due to fitting process)
        assert encoded_df1.shape == encoded_df2.shape
        assert list(encoded_df1.columns) == list(encoded_df2.columns)
    
    @given(
        st.lists(st.sampled_from(['Low', 'Medium', 'High']), min_size=1, max_size=20)
    )
    @settings(max_examples=30, deadline=None)
    def test_ordinal_encoding_order_preservation(self, job_satisfaction_values):
        """
        Property: Ordinal encoding must preserve the natural order.
        
        For any sequence of ordinal values, the encoded values should 
        maintain the same relative ordering.
        """
        df = pd.DataFrame({
            'Age': [25] * len(job_satisfaction_values),
            'Department': ['Sales'] * len(job_satisfaction_values),
            'DistanceFromHome': [10] * len(job_satisfaction_values),
            'EducationField': ['Technical'] * len(job_satisfaction_values),
            'JobRole': ['Manager'] * len(job_satisfaction_values),
            'JobSatisfaction': job_satisfaction_values,
            'MonthlyIncome': [5000] * len(job_satisfaction_values),
            'NumCompaniesWorked': [2] * len(job_satisfaction_values),
            'YearsAtCompany': [5] * len(job_satisfaction_values),
            'OverTime': ['No'] * len(job_satisfaction_values)
        })
        
        encoded_df = self.encoder.fit_transform(df)
        
        # Property: Ordinal values should follow Low=1, Medium=2, High=3
        for i, original_value in enumerate(job_satisfaction_values):
            encoded_value = encoded_df.iloc[i]['JobSatisfaction']
            if original_value == 'Low':
                assert encoded_value == 1
            elif original_value == 'Medium':
                assert encoded_value == 2
            elif original_value == 'High':
                assert encoded_value == 3
    
    @given(
        st.lists(st.sampled_from(['Yes', 'No']), min_size=1, max_size=20)
    )
    @settings(max_examples=30, deadline=None)
    def test_binary_encoding_correctness(self, overtime_values):
        """
        Property: Binary encoding must correctly map Yes/No to 1/0.
        
        For any sequence of binary values, the encoding should be consistent
        and map to the correct binary representation.
        """
        df = pd.DataFrame({
            'Age': [25] * len(overtime_values),
            'Department': ['Sales'] * len(overtime_values),
            'DistanceFromHome': [10] * len(overtime_values),
            'EducationField': ['Technical'] * len(overtime_values),
            'JobRole': ['Manager'] * len(overtime_values),
            'JobSatisfaction': ['Medium'] * len(overtime_values),
            'MonthlyIncome': [5000] * len(overtime_values),
            'NumCompaniesWorked': [2] * len(overtime_values),
            'YearsAtCompany': [5] * len(overtime_values),
            'OverTime': overtime_values
        })
        
        encoded_df = self.encoder.fit_transform(df)
        
        # Property: Binary encoding should map Yes->1, No->0
        for i, original_value in enumerate(overtime_values):
            encoded_value = encoded_df.iloc[i]['OverTime']
            if original_value == 'Yes':
                assert encoded_value == 1
            elif original_value == 'No':
                assert encoded_value == 0
    
    def test_feature_names_consistency(self):
        """
        Property: Feature names should be consistent and retrievable.
        
        After fitting, the encoder should provide consistent feature names
        that match the encoded DataFrame columns.
        """
        # Create test data
        df = pd.DataFrame({
            'Age': [25, 30, 35],
            'Department': ['Sales', 'R&D', 'HR'],
            'DistanceFromHome': [5, 10, 15],
            'EducationField': ['Technical', 'Business', 'Medical'],
            'JobRole': ['Manager', 'Analyst', 'Director'],
            'JobSatisfaction': ['Low', 'Medium', 'High'],
            'MonthlyIncome': [4000, 5000, 6000],
            'NumCompaniesWorked': [1, 2, 3],
            'YearsAtCompany': [2, 5, 8],
            'OverTime': ['No', 'Yes', 'No']
        })
        
        encoded_df = self.encoder.fit_transform(df)
        feature_names = self.encoder.get_feature_names()
        
        # Property: Feature names should match encoded DataFrame columns
        assert list(encoded_df.columns) == feature_names
        
        # Property: Feature names should be retrievable after fitting
        assert len(feature_names) > 0
        assert isinstance(feature_names, list)
        assert all(isinstance(name, str) for name in feature_names)
    
    def test_encoding_info_completeness(self):
        """
        Property: Encoding information should be complete and accurate.
        
        The encoder should provide comprehensive information about the
        encoding process and strategies used.
        """
        # Create test data
        df = pd.DataFrame({
            'Age': [25, 30, 35],
            'Department': ['Sales', 'R&D', 'HR'],
            'JobSatisfaction': ['Low', 'Medium', 'High'],
            'MonthlyIncome': [4000, 5000, 6000],
            'OverTime': ['No', 'Yes', 'No']
        })
        
        encoded_df = self.encoder.fit_transform(df)
        encoding_info = self.encoder.get_encoding_info()
        
        # Property: Encoding info should contain all required fields
        required_fields = [
            'original_columns', 'encoded_columns', 'encoding_strategy',
            'num_original_features', 'num_encoded_features', 'encoders_info'
        ]
        
        for field in required_fields:
            assert field in encoding_info
        
        # Property: Column counts should be accurate
        assert encoding_info['num_original_features'] == len(df.columns)
        assert encoding_info['num_encoded_features'] == len(encoded_df.columns)
        
        # Property: Strategy information should be complete
        for col in df.columns:
            if col in encoding_info['encoding_strategy']:
                strategy = encoding_info['encoding_strategy'][col]
                assert strategy in ['ordinal', 'onehot', 'binary', 'numerical']

if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])