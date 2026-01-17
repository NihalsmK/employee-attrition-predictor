"""
Property-based tests for DataValidator

Feature: employee-attrition-predictor, Property 11: Data Quality Validation Completeness
Validates: Requirements 15.1, 15.2, 15.5
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.pandas import data_frames, columns
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_validator import DataValidator, ValidationResult, QualityReport, OutlierReport, BusinessRuleReport

class TestDataValidatorProperties:
    """Property-based tests for DataValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
    
    @given(
        data_frames([
            columns(['Age', 'Department', 'DistanceFromHome', 'EducationField',
                    'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
                    'YearsAtCompany', 'OverTime', 'Attrition'], dtype=object)
        ])
    )
    @settings(max_examples=100, deadline=None)
    def test_property_11_data_quality_validation_completeness(self, df):
        """
        Property 11: Data Quality Validation Completeness
        
        For any input dataset, the data validator must detect all instances of 
        missing required fields, invalid data types, and values outside acceptable ranges.
        
        Validates: Requirements 15.1, 15.2, 15.5
        """
        # Ensure we have some data to work with
        assume(len(df) > 0)
        
        # Test schema validation
        schema_result = self.validator.validate_schema(df)
        
        # Property: Schema validation must always return a ValidationResult
        assert isinstance(schema_result, ValidationResult)
        assert hasattr(schema_result, 'is_valid')
        assert hasattr(schema_result, 'errors')
        assert hasattr(schema_result, 'warnings')
        assert hasattr(schema_result, 'summary')
        
        # Property: All required columns must be checked
        required_columns = set(self.validator.required_columns)
        present_columns = set(df.columns)
        missing_columns = required_columns - present_columns
        
        if missing_columns:
            # If columns are missing, validation should fail
            assert not schema_result.is_valid
            assert any("Missing required columns" in error for error in schema_result.errors)
        
        # Test completeness check
        quality_report = self.validator.check_completeness(df)
        
        # Property: Quality report must always be complete and consistent
        assert isinstance(quality_report, QualityReport)
        assert 0 <= quality_report.completeness_score <= 100
        assert quality_report.total_records == len(df)
        assert quality_report.complete_records <= quality_report.total_records
        
        # Property: Missing value counts must be accurate
        for column in df.columns:
            if column in quality_report.missing_values:
                expected_missing = df[column].isnull().sum()
                assert quality_report.missing_values[column] == expected_missing
    
    @given(
        st.integers(min_value=1, max_value=1000),
        st.random_module()
    )
    @settings(max_examples=50, deadline=None)
    def test_outlier_detection_consistency(self, n_rows, random_module):
        """
        Property: Outlier detection must be consistent and deterministic.
        
        For any numerical data, outlier detection should produce consistent results
        when run multiple times on the same data.
        """
        # Generate numerical data with known outliers
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, n_rows - 2)
        outliers = [0, 200]  # Clear outliers
        data = np.concatenate([normal_data, outliers])
        
        df = pd.DataFrame({
            'Age': data,
            'DistanceFromHome': np.random.normal(10, 5, n_rows),
            'MonthlyIncome': np.random.normal(5000, 1000, n_rows),
            'NumCompaniesWorked': np.random.poisson(2, n_rows),
            'YearsAtCompany': np.random.exponential(5, n_rows)
        })
        
        # Run outlier detection multiple times
        report1 = self.validator.detect_outliers(df)
        report2 = self.validator.detect_outliers(df)
        
        # Property: Results should be identical
        assert isinstance(report1, OutlierReport)
        assert isinstance(report2, OutlierReport)
        assert report1.outlier_counts == report2.outlier_counts
        assert report1.total_outliers == report2.total_outliers
    
    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=18, max_value=65),  # Valid age
                st.floats(min_value=1000, max_value=20000),  # Valid income
                st.integers(min_value=0, max_value=40),  # Valid years at company
                st.integers(min_value=0, max_value=50),  # Valid distance
                st.sampled_from(['Low', 'Medium', 'High']),  # Valid job satisfaction
                st.sampled_from(['Yes', 'No'])  # Valid attrition
            ),
            min_size=1,
            max_size=100
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_business_rules_validation_accuracy(self, valid_data):
        """
        Property: Business rules validation must accurately identify violations.
        
        For any dataset with valid business data, business rule validation
        should not report false violations.
        """
        # Create DataFrame from valid data
        df = pd.DataFrame(valid_data, columns=[
            'Age', 'MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome',
            'JobSatisfaction', 'Attrition'
        ])
        
        # Ensure YearsAtCompany doesn't exceed Age - 16 (business rule)
        df['YearsAtCompany'] = np.minimum(df['YearsAtCompany'], df['Age'] - 16)
        
        business_report = self.validator.validate_business_rules(df)
        
        # Property: Valid data should not have business rule violations
        assert isinstance(business_report, BusinessRuleReport)
        assert business_report.total_violations == 0
        
        # Property: All expected rules should be checked
        expected_rules = [
            'Age_Range_18_65',
            'Positive_Monthly_Income', 
            'Years_At_Company_Logical',
            'Non_Negative_Distance',
            'Valid_Job_Satisfaction',
            'Valid_Attrition_Values'
        ]
        
        for rule in expected_rules:
            assert rule in business_report.rules_checked
    
    @given(
        st.integers(min_value=0, max_value=100),  # Percentage of missing values
        st.integers(min_value=10, max_value=500)  # Dataset size
    )
    @settings(max_examples=50, deadline=None)
    def test_completeness_score_accuracy(self, missing_percentage, dataset_size):
        """
        Property: Completeness score must accurately reflect missing data percentage.
        
        For any dataset with a known percentage of missing values,
        the completeness score should accurately reflect the data quality.
        """
        # Create dataset with controlled missing values
        df = pd.DataFrame({
            'Age': range(dataset_size),
            'Department': ['Sales'] * dataset_size,
            'MonthlyIncome': [5000] * dataset_size
        })
        
        # Introduce missing values
        n_missing = int(dataset_size * missing_percentage / 100)
        if n_missing > 0:
            missing_indices = np.random.choice(dataset_size, n_missing, replace=False)
            df.loc[missing_indices, 'Age'] = np.nan
        
        quality_report = self.validator.check_completeness(df)
        
        # Property: Completeness score should be accurate
        expected_complete_records = dataset_size - n_missing
        expected_completeness = (expected_complete_records / dataset_size) * 100
        
        assert abs(quality_report.completeness_score - expected_completeness) < 0.01
        assert quality_report.complete_records == expected_complete_records
        assert quality_report.total_records == dataset_size
    
    def test_validation_result_structure_consistency(self):
        """
        Property: Validation results must have consistent structure.
        
        For any validation operation, the result structure should be consistent
        and contain all required fields.
        """
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = self.validator.validate_schema(empty_df)
        
        # Property: Result structure must be consistent
        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.summary, dict)
        
        # Test with valid DataFrame
        valid_df = pd.DataFrame({
            'Age': [25, 30, 35],
            'Department': ['Sales', 'HR', 'IT'],
            'DistanceFromHome': [5, 10, 15],
            'EducationField': ['Technical', 'Business', 'Technical'],
            'JobSatisfaction': ['High', 'Medium', 'Low'],
            'MonthlyIncome': [5000, 6000, 7000],
            'NumCompaniesWorked': [1, 2, 3],
            'YearsAtCompany': [2, 5, 8],
            'OverTime': ['No', 'Yes', 'No'],
            'Attrition': ['No', 'No', 'Yes']
        })
        
        result = self.validator.validate_schema(valid_df)
        
        # Property: Valid data should pass schema validation
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.summary['schema_valid']

if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])