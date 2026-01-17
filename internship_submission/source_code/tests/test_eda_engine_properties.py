"""
Property-based tests for EDAEngine

Feature: employee-attrition-predictor, Property 5: EDA Visualization Generation Completeness
Feature: employee-attrition-predictor, Property 6: Hypothesis Testing Statistical Validity
Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 10.1, 10.2, 10.5
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hypothesis import given, strategies as st, settings, assume
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from eda_engine import EDAEngine, AnalysisResult, HypothesisResult, StatsSummary

class TestEDAEngineProperties:
    """Property-based tests for EDAEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.eda = EDAEngine()
    
    @given(
        st.integers(min_value=10, max_value=100),  # Dataset size
        st.integers(min_value=2, max_value=5),     # Number of numerical columns
        st.random_module()
    )
    @settings(max_examples=30, deadline=None)
    def test_property_5_eda_visualization_generation_completeness(self, n_rows, n_numerical_cols, random_module):
        """
        Property 5: EDA Visualization Generation Completeness
        
        For any valid HR dataset, the EDA engine must successfully generate all 
        required visualizations (correlation matrices, distribution plots, 
        cross-tabulations) without errors.
        
        Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
        """
        # Generate numerical data
        np.random.seed(42)
        numerical_data = {}
        for i in range(n_numerical_cols):
            col_name = f'numerical_col_{i}'
            numerical_data[col_name] = np.random.normal(50, 10, n_rows)
        
        # Add categorical data
        categorical_data = {
            'Department': np.random.choice(['Sales', 'R&D', 'HR'], n_rows),
            'JobSatisfaction': np.random.choice(['Low', 'Medium', 'High'], n_rows),
            'Attrition': np.random.choice(['Yes', 'No'], n_rows, p=[0.3, 0.7])
        }
        
        # Combine data
        df = pd.DataFrame({**numerical_data, **categorical_data})
        
        # Property: Correlation matrix generation should always succeed
        corr_fig = self.eda.generate_correlation_matrix(df)
        assert isinstance(corr_fig, plt.Figure)
        assert len(corr_fig.axes) > 0
        plt.close(corr_fig)
        
        # Property: Distribution plots should be generated for numerical columns
        numerical_cols = list(numerical_data.keys())
        dist_figs = self.eda.create_distribution_plots(df, numerical_cols[:2])  # Test with first 2 columns
        assert isinstance(dist_figs, list)
        assert len(dist_figs) > 0
        for fig in dist_figs:
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        
        # Property: Feature analysis should work for both numerical and categorical features
        for feature in ['Department', numerical_cols[0]]:
            analysis_result = self.eda.analyze_attrition_by_feature(df, feature)
            assert isinstance(analysis_result, AnalysisResult)
            assert analysis_result.feature_name == feature
            assert isinstance(analysis_result.visualizations, list)
            assert len(analysis_result.visualizations) > 0
            assert isinstance(analysis_result.insights, list)
            
            # Clean up figures
            for fig in analysis_result.visualizations:
                plt.close(fig)
    
    @given(
        st.integers(min_value=50, max_value=200),  # Dataset size for statistical power
        st.floats(min_value=0.1, max_value=0.9),  # Correlation strength
        st.random_module()
    )
    @settings(max_examples=20, deadline=None)
    def test_property_6_hypothesis_testing_statistical_validity(self, n_rows, correlation_strength, random_module):
        """
        Property 6: Hypothesis Testing Statistical Validity
        
        For any HR dataset with sufficient sample size, correlation tests between 
        DistanceFromHome and Attrition, and attrition rate comparisons between 
        overtime groups, must produce statistically valid results.
        
        Validates: Requirements 10.1, 10.2, 10.5
        """
        # Generate data with controlled relationships
        np.random.seed(42)
        
        # Create distance data with known correlation to attrition
        distance = np.random.exponential(10, n_rows).astype(int).clip(1, 50)
        
        # Create attrition with correlation to distance
        attrition_prob = 0.2 + correlation_strength * (distance - distance.min()) / (distance.max() - distance.min())
        attrition_binary = np.random.binomial(1, attrition_prob, n_rows)
        attrition = ['Yes' if x == 1 else 'No' for x in attrition_binary]
        
        # Create overtime data with different attrition rates
        overtime = np.random.choice(['Yes', 'No'], n_rows, p=[0.3, 0.7])
        
        # Adjust attrition based on overtime (create association)
        for i in range(n_rows):
            if overtime[i] == 'Yes' and np.random.random() < 0.3:  # Increase attrition for overtime
                attrition[i] = 'Yes'
        
        df = pd.DataFrame({
            'DistanceFromHome': distance,
            'OverTime': overtime,
            'Attrition': attrition,
            'MonthlyIncome': np.random.normal(5000, 1000, n_rows),
            'JobSatisfaction': np.random.choice(['Low', 'Medium', 'High'], n_rows)
        })
        
        # Property: Distance-attrition correlation test should produce valid results
        distance_test = self.eda.test_hypothesis(df, "distance_attrition_correlation")
        assert isinstance(distance_test, HypothesisResult)
        assert isinstance(distance_test.test_statistic, (int, float))
        assert isinstance(distance_test.p_value, (int, float))
        assert 0 <= distance_test.p_value <= 1
        assert isinstance(distance_test.is_significant, bool)
        assert distance_test.effect_size is not None
        assert isinstance(distance_test.interpretation, str)
        assert len(distance_test.interpretation) > 0
        
        # Property: Overtime-attrition association test should produce valid results
        overtime_test = self.eda.test_hypothesis(df, "overtime_attrition_association")
        assert isinstance(overtime_test, HypothesisResult)
        assert isinstance(overtime_test.test_statistic, (int, float))
        assert overtime_test.test_statistic >= 0  # Chi-square is always non-negative
        assert isinstance(overtime_test.p_value, (int, float))
        assert 0 <= overtime_test.p_value <= 1
        assert isinstance(overtime_test.is_significant, bool)
        assert overtime_test.effect_size is not None
        assert isinstance(overtime_test.interpretation, str)
        assert len(overtime_test.interpretation) > 0
        
        # Property: Income-attrition relationship test should produce valid results
        income_test = self.eda.test_hypothesis(df, "income_attrition_relationship")
        assert isinstance(income_test, HypothesisResult)
        assert isinstance(income_test.test_statistic, (int, float))
        assert isinstance(income_test.p_value, (int, float))
        assert 0 <= income_test.p_value <= 1
        assert isinstance(income_test.is_significant, bool)
        assert income_test.effect_size is not None
        assert isinstance(income_test.interpretation, str)
        assert len(income_test.interpretation) > 0
        
        # Property: Job satisfaction-attrition relationship test should produce valid results
        satisfaction_test = self.eda.test_hypothesis(df, "satisfaction_attrition_relationship")
        assert isinstance(satisfaction_test, HypothesisResult)
        assert isinstance(satisfaction_test.test_statistic, (int, float))
        assert satisfaction_test.test_statistic >= 0  # Chi-square is always non-negative
        assert isinstance(satisfaction_test.p_value, (int, float))
        assert 0 <= satisfaction_test.p_value <= 1
        assert isinstance(satisfaction_test.is_significant, bool)
        assert satisfaction_test.effect_size is not None
        assert isinstance(satisfaction_test.interpretation, str)
        assert len(satisfaction_test.interpretation) > 0
    
    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=18, max_value=65),  # Age
                st.floats(min_value=1000, max_value=20000),  # Income
                st.integers(min_value=1, max_value=50),  # Distance
                st.integers(min_value=0, max_value=40),  # Years at company
                st.sampled_from(['Sales', 'R&D', 'HR', 'Finance']),  # Department
                st.sampled_from(['Yes', 'No'])  # Attrition
            ),
            min_size=20,
            max_size=100
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_summary_statistics_completeness(self, hr_data):
        """
        Property: Summary statistics should be complete and accurate.
        
        For any HR dataset, the summary statistics should include all 
        required components and be mathematically consistent.
        """
        # Create DataFrame from generated data
        df = pd.DataFrame(hr_data, columns=[
            'Age', 'MonthlyIncome', 'DistanceFromHome', 'YearsAtCompany',
            'Department', 'Attrition'
        ])
        
        # Generate summary statistics
        summary = self.eda.generate_summary_statistics(df)
        
        # Property: Summary should contain all required components
        assert isinstance(summary, StatsSummary)
        assert isinstance(summary.numerical_summary, pd.DataFrame)
        assert isinstance(summary.categorical_summary, dict)
        assert isinstance(summary.correlation_matrix, pd.DataFrame)
        assert isinstance(summary.missing_values, pd.Series)
        assert isinstance(summary.data_types, pd.Series)
        
        # Property: Numerical summary should include standard statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            assert len(summary.numerical_summary.columns) == len(numerical_cols)
            required_stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
            for stat in required_stats:
                assert stat in summary.numerical_summary.index
        
        # Property: Categorical summary should include value counts
        categorical_cols = df.select_dtypes(include=['object']).columns
        assert len(summary.categorical_summary) == len(categorical_cols)
        for col in categorical_cols:
            assert col in summary.categorical_summary
            assert isinstance(summary.categorical_summary[col], pd.Series)
            assert summary.categorical_summary[col].sum() == len(df)
        
        # Property: Missing values count should be accurate
        for col in df.columns:
            expected_missing = df[col].isnull().sum()
            assert summary.missing_values[col] == expected_missing
    
    def test_correlation_matrix_properties(self):
        """
        Property: Correlation matrix should have mathematical properties.
        
        Correlation matrices should be symmetric, have 1s on diagonal,
        and values between -1 and 1.
        """
        # Create test data with known correlations
        n_rows = 100
        np.random.seed(42)
        
        x1 = np.random.normal(0, 1, n_rows)
        x2 = 0.8 * x1 + 0.6 * np.random.normal(0, 1, n_rows)  # Correlated with x1
        x3 = np.random.normal(0, 1, n_rows)  # Independent
        
        df = pd.DataFrame({
            'Variable1': x1,
            'Variable2': x2,
            'Variable3': x3,
            'Category': np.random.choice(['A', 'B'], n_rows)
        })
        
        # Generate correlation matrix
        corr_fig = self.eda.generate_correlation_matrix(df)
        
        # Get the correlation matrix data
        summary = self.eda.generate_summary_statistics(df)
        corr_matrix = summary.correlation_matrix
        
        # Property: Correlation matrix should be symmetric
        assert np.allclose(corr_matrix.values, corr_matrix.values.T, rtol=1e-10)
        
        # Property: Diagonal should be 1s
        np.testing.assert_array_almost_equal(np.diag(corr_matrix.values), 1.0)
        
        # Property: All values should be between -1 and 1
        assert np.all(corr_matrix.values >= -1)
        assert np.all(corr_matrix.values <= 1)
        
        # Property: Known correlation should be detected
        assert abs(corr_matrix.loc['Variable1', 'Variable2']) > 0.5  # Should be correlated
        
        plt.close(corr_fig)
    
    def test_analysis_result_structure_consistency(self):
        """
        Property: Analysis results should have consistent structure.
        
        For any feature analysis, the result structure should be consistent
        and contain all required fields.
        """
        # Create test data
        df = pd.DataFrame({
            'Age': [25, 30, 35, 40, 45],
            'Department': ['Sales', 'R&D', 'HR', 'Sales', 'R&D'],
            'MonthlyIncome': [4000, 5000, 6000, 7000, 8000],
            'Attrition': ['No', 'Yes', 'No', 'Yes', 'No']
        })
        
        # Test numerical feature analysis
        numerical_result = self.eda.analyze_attrition_by_feature(df, 'Age')
        
        # Property: Result structure should be consistent
        assert isinstance(numerical_result, AnalysisResult)
        assert numerical_result.feature_name == 'Age'
        assert numerical_result.analysis_type == 'attrition_analysis'
        assert isinstance(numerical_result.results, dict)
        assert isinstance(numerical_result.visualizations, list)
        assert isinstance(numerical_result.insights, list)
        assert len(numerical_result.visualizations) > 0
        assert len(numerical_result.insights) > 0
        
        # Test categorical feature analysis
        categorical_result = self.eda.analyze_attrition_by_feature(df, 'Department')
        
        # Property: Result structure should be consistent for categorical features
        assert isinstance(categorical_result, AnalysisResult)
        assert categorical_result.feature_name == 'Department'
        assert categorical_result.analysis_type == 'attrition_analysis'
        assert isinstance(categorical_result.results, dict)
        assert isinstance(categorical_result.visualizations, list)
        assert isinstance(categorical_result.insights, list)
        assert len(categorical_result.visualizations) > 0
        assert len(categorical_result.insights) > 0
        
        # Property: Categorical analysis should include cross-tabulation
        assert 'crosstab' in categorical_result.results
        assert 'chi2_test' in categorical_result.results
        
        # Clean up figures
        for fig in numerical_result.visualizations + categorical_result.visualizations:
            plt.close(fig)
    
    def test_hypothesis_test_error_handling(self):
        """
        Property: Hypothesis tests should handle edge cases gracefully.
        
        Tests should fail gracefully when required columns are missing
        or data is insufficient.
        """
        # Test with missing columns
        df_missing_cols = pd.DataFrame({
            'Age': [25, 30, 35],
            'Department': ['Sales', 'R&D', 'HR']
        })
        
        # Property: Should raise appropriate errors for missing columns
        with pytest.raises(ValueError, match="Required columns not found"):
            self.eda.test_hypothesis(df_missing_cols, "distance_attrition_correlation")
        
        with pytest.raises(ValueError, match="Required columns not found"):
            self.eda.test_hypothesis(df_missing_cols, "overtime_attrition_association")
        
        # Property: Should raise error for unknown hypothesis
        df_complete = pd.DataFrame({
            'DistanceFromHome': [5, 10, 15],
            'Attrition': ['Yes', 'No', 'Yes']
        })
        
        with pytest.raises(ValueError, match="Unknown hypothesis"):
            self.eda.test_hypothesis(df_complete, "unknown_hypothesis")

if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])