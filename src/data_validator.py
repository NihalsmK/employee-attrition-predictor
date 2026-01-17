"""
Data Validator for Employee Attrition Predictor

This module provides comprehensive data validation capabilities for HR datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation operations."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]

@dataclass
class QualityReport:
    """Data quality assessment report."""
    completeness_score: float
    missing_values: Dict[str, int]
    missing_percentage: Dict[str, float]
    total_records: int
    complete_records: int

@dataclass
class OutlierReport:
    """Outlier detection report."""
    outliers_detected: Dict[str, List[int]]
    outlier_counts: Dict[str, int]
    outlier_methods: Dict[str, str]
    total_outliers: int

@dataclass
class BusinessRuleReport:
    """Business rule validation report."""
    rule_violations: Dict[str, List[int]]
    violation_counts: Dict[str, int]
    rules_checked: List[str]
    total_violations: int

class DataValidator:
    """
    Comprehensive data validator for HR datasets.
    
    Validates schema, data quality, outliers, and business rules.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize the DataValidator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.required_columns = self.config['data_processing']['required_columns']
        
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
                'required_columns': [
                    'Age', 'Department', 'DistanceFromHome', 'EducationField',
                    'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
                    'YearsAtCompany', 'OverTime', 'Attrition'
                ]
            }
        }
    
    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate dataset schema against required columns and data types.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with schema validation details
        """
        errors = []
        warnings = []
        
        # Check required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {list(missing_columns)}")
        
        # Check for extra columns
        extra_columns = set(df.columns) - set(self.required_columns)
        if extra_columns:
            warnings.append(f"Extra columns found: {list(extra_columns)}")
        
        # Validate data types
        type_errors = self._validate_data_types(df)
        errors.extend(type_errors)
        
        # Check for empty dataset
        if len(df) == 0:
            errors.append("Dataset is empty")
        
        # Check for duplicate columns
        if len(df.columns) != len(set(df.columns)):
            errors.append("Duplicate column names detected")
        
        summary = {
            'total_columns': len(df.columns),
            'required_columns_present': len(set(self.required_columns) & set(df.columns)),
            'total_records': len(df),
            'schema_valid': len(errors) == 0
        }
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            summary=summary
        )
    
    def _validate_data_types(self, df: pd.DataFrame) -> List[str]:
        """Validate expected data types for each column."""
        errors = []
        
        # Expected data types
        expected_types = {
            'Age': ['int64', 'int32', 'float64'],
            'DistanceFromHome': ['int64', 'int32', 'float64'],
            'MonthlyIncome': ['int64', 'int32', 'float64'],
            'NumCompaniesWorked': ['int64', 'int32', 'float64'],
            'YearsAtCompany': ['int64', 'int32', 'float64'],
            'Department': ['object', 'category'],
            'EducationField': ['object', 'category'],
            'JobSatisfaction': ['object', 'category'],
            'OverTime': ['object', 'category'],
            'Attrition': ['object', 'category']
        }
        
        for column, expected in expected_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if actual_type not in expected:
                    errors.append(f"Column '{column}' has type '{actual_type}', expected one of {expected}")
        
        return errors
    
    def check_completeness(self, df: pd.DataFrame) -> QualityReport:
        """
        Check data completeness and missing value patterns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            QualityReport with completeness metrics
        """
        total_records = len(df)
        missing_values = df.isnull().sum().to_dict()
        missing_percentage = (df.isnull().sum() / total_records * 100).to_dict()
        
        # Calculate completeness score (percentage of complete records)
        complete_records = df.dropna().shape[0]
        completeness_score = (complete_records / total_records) * 100 if total_records > 0 else 0
        
        return QualityReport(
            completeness_score=completeness_score,
            missing_values=missing_values,
            missing_percentage=missing_percentage,
            total_records=total_records,
            complete_records=complete_records
        )
    
    def detect_outliers(self, df: pd.DataFrame) -> OutlierReport:
        """
        Detect outliers in numerical columns using IQR method.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            OutlierReport with outlier detection results
        """
        numerical_columns = ['Age', 'DistanceFromHome', 'MonthlyIncome', 
                           'NumCompaniesWorked', 'YearsAtCompany']
        
        outliers_detected = {}
        outlier_counts = {}
        outlier_methods = {}
        
        for column in numerical_columns:
            if column in df.columns:
                outlier_indices = self._detect_iqr_outliers(df[column])
                outliers_detected[column] = outlier_indices
                outlier_counts[column] = len(outlier_indices)
                outlier_methods[column] = "IQR"
        
        total_outliers = sum(outlier_counts.values())
        
        return OutlierReport(
            outliers_detected=outliers_detected,
            outlier_counts=outlier_counts,
            outlier_methods=outlier_methods,
            total_outliers=total_outliers
        )
    
    def _detect_iqr_outliers(self, series: pd.Series) -> List[int]:
        """Detect outliers using Interquartile Range method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        return series[outlier_mask].index.tolist()
    
    def validate_business_rules(self, df: pd.DataFrame) -> BusinessRuleReport:
        """
        Validate business logic constraints.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            BusinessRuleReport with business rule validation results
        """
        rule_violations = {}
        violation_counts = {}
        rules_checked = []
        
        # Rule 1: Age should be between 18 and 65
        if 'Age' in df.columns:
            rule_name = "Age_Range_18_65"
            violations = df[(df['Age'] < 18) | (df['Age'] > 65)].index.tolist()
            rule_violations[rule_name] = violations
            violation_counts[rule_name] = len(violations)
            rules_checked.append(rule_name)
        
        # Rule 2: MonthlyIncome should be positive
        if 'MonthlyIncome' in df.columns:
            rule_name = "Positive_Monthly_Income"
            violations = df[df['MonthlyIncome'] <= 0].index.tolist()
            rule_violations[rule_name] = violations
            violation_counts[rule_name] = len(violations)
            rules_checked.append(rule_name)
        
        # Rule 3: YearsAtCompany should not exceed Age - 16
        if 'Age' in df.columns and 'YearsAtCompany' in df.columns:
            rule_name = "Years_At_Company_Logical"
            violations = df[df['YearsAtCompany'] > (df['Age'] - 16)].index.tolist()
            rule_violations[rule_name] = violations
            violation_counts[rule_name] = len(violations)
            rules_checked.append(rule_name)
        
        # Rule 4: DistanceFromHome should be non-negative
        if 'DistanceFromHome' in df.columns:
            rule_name = "Non_Negative_Distance"
            violations = df[df['DistanceFromHome'] < 0].index.tolist()
            rule_violations[rule_name] = violations
            violation_counts[rule_name] = len(violations)
            rules_checked.append(rule_name)
        
        # Rule 5: JobSatisfaction should be valid values
        if 'JobSatisfaction' in df.columns:
            rule_name = "Valid_Job_Satisfaction"
            valid_values = ['Low', 'Medium', 'High']
            violations = df[~df['JobSatisfaction'].isin(valid_values)].index.tolist()
            rule_violations[rule_name] = violations
            violation_counts[rule_name] = len(violations)
            rules_checked.append(rule_name)
        
        # Rule 6: Attrition should be Yes/No
        if 'Attrition' in df.columns:
            rule_name = "Valid_Attrition_Values"
            valid_values = ['Yes', 'No']
            violations = df[~df['Attrition'].isin(valid_values)].index.tolist()
            rule_violations[rule_name] = violations
            violation_counts[rule_name] = len(violations)
            rules_checked.append(rule_name)
        
        total_violations = sum(violation_counts.values())
        
        return BusinessRuleReport(
            rule_violations=rule_violations,
            violation_counts=violation_counts,
            rules_checked=rules_checked,
            total_violations=total_violations
        )
    
    def generate_validation_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary containing all validation results
        """
        logger.info("Starting comprehensive data validation...")
        
        schema_result = self.validate_schema(df)
        quality_report = self.check_completeness(df)
        outlier_report = self.detect_outliers(df)
        business_rule_report = self.validate_business_rules(df)
        
        report = {
            'schema_validation': schema_result,
            'data_quality': quality_report,
            'outlier_detection': outlier_report,
            'business_rules': business_rule_report,
            'overall_status': {
                'schema_valid': schema_result.is_valid,
                'data_complete': quality_report.completeness_score > 95,
                'outliers_detected': outlier_report.total_outliers > 0,
                'business_rules_passed': business_rule_report.total_violations == 0
            }
        }
        
        logger.info("Data validation completed.")
        return report

if __name__ == "__main__":
    # Test the DataValidator with sample data
    validator = DataValidator()
    
    # Load sample data
    df = pd.read_csv('data/hr_employee_data.csv')
    
    # Generate validation report
    report = validator.generate_validation_report(df)
    
    print("=== Data Validation Report ===")
    print(f"Schema Valid: {report['overall_status']['schema_valid']}")
    print(f"Data Completeness: {report['data_quality'].completeness_score:.2f}%")
    print(f"Outliers Detected: {report['outlier_detection'].total_outliers}")
    print(f"Business Rule Violations: {report['business_rules'].total_violations}")