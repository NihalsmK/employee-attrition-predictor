"""
Exploratory Data Analysis Engine for Employee Attrition Predictor

This module provides comprehensive EDA capabilities for HR datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import scipy.stats as stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import warnings
import logging

# Configure plotting
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Result of feature analysis operations."""
    feature_name: str
    analysis_type: str
    results: Dict[str, Any]
    visualizations: List[plt.Figure]
    insights: List[str]

@dataclass
class HypothesisResult:
    """Result of hypothesis testing."""
    hypothesis: str
    test_statistic: float
    p_value: float
    is_significant: bool
    effect_size: Optional[float]
    interpretation: str

@dataclass
class StatsSummary:
    """Statistical summary of dataset."""
    numerical_summary: pd.DataFrame
    categorical_summary: Dict[str, pd.Series]
    correlation_matrix: pd.DataFrame
    missing_values: pd.Series
    data_types: pd.Series

class EDAEngine:
    """
    Comprehensive Exploratory Data Analysis engine for HR datasets.
    
    Provides visualization, statistical analysis, and hypothesis testing capabilities.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the EDAEngine.
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 10)
        
    def generate_correlation_matrix(self, df: pd.DataFrame, method: str = 'pearson') -> plt.Figure:
        """
        Generate correlation matrix for numerical variables.
        
        Args:
            df: DataFrame to analyze
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Matplotlib figure with correlation heatmap
        """
        logger.info(f"Generating {method} correlation matrix...")
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            logger.warning("Less than 2 numerical columns found for correlation analysis.")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'Insufficient numerical columns for correlation analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Correlation Matrix - Insufficient Data')
            return fig
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr(method=method)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title(f'{method.capitalize()} Correlation Matrix - Numerical Variables', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def analyze_attrition_by_feature(self, df: pd.DataFrame, feature: str, 
                                   target: str = 'Attrition') -> AnalysisResult:
        """
        Analyze attrition patterns by a specific feature.
        
        Args:
            df: DataFrame to analyze
            feature: Feature to analyze
            target: Target variable (default: 'Attrition')
            
        Returns:
            AnalysisResult with analysis details and visualizations
        """
        logger.info(f"Analyzing attrition patterns by {feature}...")
        
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame")
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found in DataFrame")
        
        visualizations = []
        insights = []
        results = {}
        
        # Determine if feature is numerical or categorical
        is_numerical = df[feature].dtype in ['int64', 'float64', 'int32', 'float32']
        
        if is_numerical:
            # Numerical feature analysis
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Distribution by attrition
            for i, attrition_value in enumerate(df[target].unique()):
                subset = df[df[target] == attrition_value][feature]
                axes[0, 0].hist(subset, alpha=0.7, label=f'{target}={attrition_value}', bins=20)
            axes[0, 0].set_title(f'{feature} Distribution by {target}')
            axes[0, 0].set_xlabel(feature)
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            
            # Box plot
            df.boxplot(column=feature, by=target, ax=axes[0, 1])
            axes[0, 1].set_title(f'{feature} Box Plot by {target}')
            
            # Violin plot
            sns.violinplot(data=df, x=target, y=feature, ax=axes[1, 0])
            axes[1, 0].set_title(f'{feature} Violin Plot by {target}')
            
            # Statistical summary
            summary_stats = df.groupby(target)[feature].agg(['mean', 'median', 'std', 'count'])
            axes[1, 1].axis('tight')
            axes[1, 1].axis('off')
            table = axes[1, 1].table(cellText=summary_stats.round(2).values,
                                   rowLabels=summary_stats.index,
                                   colLabels=summary_stats.columns,
                                   cellLoc='center',
                                   loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            axes[1, 1].set_title(f'{feature} Summary Statistics by {target}')
            
            plt.tight_layout()
            visualizations.append(fig)
            
            # Statistical tests
            groups = [df[df[target] == val][feature].dropna() for val in df[target].unique()]
            if len(groups) == 2:
                # T-test for two groups
                stat, p_value = stats.ttest_ind(groups[0], groups[1])
                results['t_test'] = {'statistic': stat, 'p_value': p_value}
                
                if p_value < 0.05:
                    insights.append(f"Significant difference in {feature} between {target} groups (p={p_value:.4f})")
                else:
                    insights.append(f"No significant difference in {feature} between {target} groups (p={p_value:.4f})")
            
            # Effect size (Cohen's d)
            if len(groups) == 2:
                pooled_std = np.sqrt(((len(groups[0])-1)*groups[0].var() + (len(groups[1])-1)*groups[1].var()) / 
                                   (len(groups[0]) + len(groups[1]) - 2))
                cohens_d = (groups[0].mean() - groups[1].mean()) / pooled_std
                results['cohens_d'] = cohens_d
                
                if abs(cohens_d) > 0.8:
                    insights.append(f"Large effect size (Cohen's d = {cohens_d:.3f})")
                elif abs(cohens_d) > 0.5:
                    insights.append(f"Medium effect size (Cohen's d = {cohens_d:.3f})")
                elif abs(cohens_d) > 0.2:
                    insights.append(f"Small effect size (Cohen's d = {cohens_d:.3f})")
        
        else:
            # Categorical feature analysis
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Cross-tabulation
            crosstab = pd.crosstab(df[feature], df[target])
            results['crosstab'] = crosstab
            
            # Stacked bar chart
            crosstab.plot(kind='bar', stacked=True, ax=axes[0, 0])
            axes[0, 0].set_title(f'{feature} by {target} (Stacked)')
            axes[0, 0].set_xlabel(feature)
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].legend(title=target)
            plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
            
            # Percentage stacked bar chart
            crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
            crosstab_pct.plot(kind='bar', stacked=True, ax=axes[0, 1])
            axes[0, 1].set_title(f'{feature} by {target} (Percentage)')
            axes[0, 1].set_xlabel(feature)
            axes[0, 1].set_ylabel('Percentage')
            axes[0, 1].legend(title=target)
            plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
            
            # Heatmap
            sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
            axes[1, 0].set_title(f'{feature} vs {target} Heatmap')
            
            # Chi-square test
            chi2, p_value, dof, expected = chi2_contingency(crosstab)
            results['chi2_test'] = {
                'chi2': chi2, 'p_value': p_value, 'dof': dof, 'expected': expected
            }
            
            # Display test results
            axes[1, 1].axis('tight')
            axes[1, 1].axis('off')
            test_results = [
                ['Chi-square statistic', f'{chi2:.4f}'],
                ['p-value', f'{p_value:.4f}'],
                ['Degrees of freedom', f'{dof}'],
                ['Significant?', 'Yes' if p_value < 0.05 else 'No']
            ]
            table = axes[1, 1].table(cellText=test_results,
                                   colLabels=['Statistic', 'Value'],
                                   cellLoc='center',
                                   loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            axes[1, 1].set_title('Chi-square Test Results')
            
            plt.tight_layout()
            visualizations.append(fig)
            
            if p_value < 0.05:
                insights.append(f"Significant association between {feature} and {target} (χ²={chi2:.4f}, p={p_value:.4f})")
            else:
                insights.append(f"No significant association between {feature} and {target} (χ²={chi2:.4f}, p={p_value:.4f})")
            
            # Calculate Cramér's V (effect size for categorical variables)
            n = crosstab.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))
            results['cramers_v'] = cramers_v
            
            if cramers_v > 0.5:
                insights.append(f"Strong association (Cramér's V = {cramers_v:.3f})")
            elif cramers_v > 0.3:
                insights.append(f"Moderate association (Cramér's V = {cramers_v:.3f})")
            elif cramers_v > 0.1:
                insights.append(f"Weak association (Cramér's V = {cramers_v:.3f})")
        
        return AnalysisResult(
            feature_name=feature,
            analysis_type='attrition_analysis',
            results=results,
            visualizations=visualizations,
            insights=insights
        )
    
    def create_distribution_plots(self, df: pd.DataFrame, features: List[str] = None) -> List[plt.Figure]:
        """
        Create distribution plots for specified features.
        
        Args:
            df: DataFrame to analyze
            features: List of features to plot (if None, plots all numerical features)
            
        Returns:
            List of matplotlib figures
        """
        logger.info("Creating distribution plots...")
        
        if features is None:
            # Default to key HR features
            features = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'YearsAtCompany']
            features = [f for f in features if f in df.columns]
        
        figures = []
        
        # Create subplots for multiple features
        n_features = len(features)
        if n_features == 0:
            logger.warning("No features specified for distribution plots.")
            return figures
        
        # Determine subplot layout
        n_cols = min(2, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            if feature in df.columns:
                # Create histogram with KDE
                df[feature].hist(bins=30, alpha=0.7, ax=ax, density=True)
                df[feature].plot.kde(ax=ax, color='red', linewidth=2)
                
                ax.set_title(f'{feature} Distribution')
                ax.set_xlabel(feature)
                ax.set_ylabel('Density')
                
                # Add statistics
                mean_val = df[feature].mean()
                median_val = df[feature].median()
                ax.axvline(mean_val, color='green', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_val:.2f}')
                ax.legend()
        
        # Hide empty subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        figures.append(fig)
        
        # Create department-wise analysis if Department column exists
        if 'Department' in df.columns:
            fig_dept = self._create_departmental_analysis(df)
            figures.append(fig_dept)
        
        return figures
    
    def _create_departmental_analysis(self, df: pd.DataFrame) -> plt.Figure:
        """Create comprehensive departmental analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Department distribution
        dept_counts = df['Department'].value_counts()
        dept_counts.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Employee Distribution by Department')
        axes[0, 0].set_xlabel('Department')
        axes[0, 0].set_ylabel('Count')
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Attrition by department (if Attrition column exists)
        if 'Attrition' in df.columns:
            attrition_by_dept = df.groupby('Department')['Attrition'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).sort_values(ascending=False)
            
            attrition_by_dept.plot(kind='bar', ax=axes[0, 1], color='coral')
            axes[0, 1].set_title('Attrition Rate by Department')
            axes[0, 1].set_xlabel('Department')
            axes[0, 1].set_ylabel('Attrition Rate (%)')
            plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # Average income by department (if MonthlyIncome exists)
        if 'MonthlyIncome' in df.columns:
            avg_income = df.groupby('Department')['MonthlyIncome'].mean().sort_values(ascending=False)
            avg_income.plot(kind='bar', ax=axes[1, 0], color='lightblue')
            axes[1, 0].set_title('Average Monthly Income by Department')
            axes[1, 0].set_xlabel('Department')
            axes[1, 0].set_ylabel('Average Monthly Income')
            plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Age distribution by department
        if 'Age' in df.columns:
            df.boxplot(column='Age', by='Department', ax=axes[1, 1])
            axes[1, 1].set_title('Age Distribution by Department')
            axes[1, 1].set_xlabel('Department')
            axes[1, 1].set_ylabel('Age')
        
        plt.tight_layout()
        return fig
    
    def test_hypothesis(self, df: pd.DataFrame, hypothesis: str) -> HypothesisResult:
        """
        Test specific business hypotheses about the data.
        
        Args:
            df: DataFrame to analyze
            hypothesis: Hypothesis to test
            
        Returns:
            HypothesisResult with test results
        """
        logger.info(f"Testing hypothesis: {hypothesis}")
        
        if hypothesis.lower() == "distance_attrition_correlation":
            return self._test_distance_attrition_correlation(df)
        elif hypothesis.lower() == "overtime_attrition_association":
            return self._test_overtime_attrition_association(df)
        elif hypothesis.lower() == "income_attrition_relationship":
            return self._test_income_attrition_relationship(df)
        elif hypothesis.lower() == "satisfaction_attrition_relationship":
            return self._test_satisfaction_attrition_relationship(df)
        else:
            raise ValueError(f"Unknown hypothesis: {hypothesis}")
    
    def _test_distance_attrition_correlation(self, df: pd.DataFrame) -> HypothesisResult:
        """Test correlation between distance from home and attrition."""
        if 'DistanceFromHome' not in df.columns or 'Attrition' not in df.columns:
            raise ValueError("Required columns not found for distance-attrition correlation test")
        
        # Convert attrition to binary
        attrition_binary = (df['Attrition'] == 'Yes').astype(int)
        
        # Calculate correlation
        correlation, p_value = pearsonr(df['DistanceFromHome'], attrition_binary)
        
        # Determine significance
        is_significant = p_value < 0.05
        
        # Interpretation
        if is_significant:
            if correlation > 0:
                interpretation = f"Significant positive correlation: employees living farther from work are more likely to leave (r={correlation:.3f}, p={p_value:.4f})"
            else:
                interpretation = f"Significant negative correlation: employees living closer to work are more likely to leave (r={correlation:.3f}, p={p_value:.4f})"
        else:
            interpretation = f"No significant correlation between distance from home and attrition (r={correlation:.3f}, p={p_value:.4f})"
        
        return HypothesisResult(
            hypothesis="Distance from home correlates with attrition",
            test_statistic=correlation,
            p_value=p_value,
            is_significant=is_significant,
            effect_size=abs(correlation),
            interpretation=interpretation
        )
    
    def _test_overtime_attrition_association(self, df: pd.DataFrame) -> HypothesisResult:
        """Test association between overtime and attrition."""
        if 'OverTime' not in df.columns or 'Attrition' not in df.columns:
            raise ValueError("Required columns not found for overtime-attrition association test")
        
        # Create contingency table
        crosstab = pd.crosstab(df['OverTime'], df['Attrition'])
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(crosstab)
        
        # Determine significance
        is_significant = p_value < 0.05
        
        # Calculate effect size (Cramér's V)
        n = crosstab.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))
        
        # Calculate attrition rates
        overtime_attrition_rate = crosstab.loc['Yes', 'Yes'] / crosstab.loc['Yes'].sum() * 100
        no_overtime_attrition_rate = crosstab.loc['No', 'Yes'] / crosstab.loc['No'].sum() * 100
        
        # Interpretation
        if is_significant:
            interpretation = f"Significant association: employees working overtime have {overtime_attrition_rate:.1f}% attrition rate vs {no_overtime_attrition_rate:.1f}% for non-overtime (χ²={chi2:.3f}, p={p_value:.4f})"
        else:
            interpretation = f"No significant association between overtime and attrition (χ²={chi2:.3f}, p={p_value:.4f})"
        
        return HypothesisResult(
            hypothesis="Overtime work is associated with higher attrition",
            test_statistic=chi2,
            p_value=p_value,
            is_significant=is_significant,
            effect_size=cramers_v,
            interpretation=interpretation
        )
    
    def _test_income_attrition_relationship(self, df: pd.DataFrame) -> HypothesisResult:
        """Test relationship between income and attrition."""
        if 'MonthlyIncome' not in df.columns or 'Attrition' not in df.columns:
            raise ValueError("Required columns not found for income-attrition relationship test")
        
        # Split by attrition
        stayed = df[df['Attrition'] == 'No']['MonthlyIncome']
        left = df[df['Attrition'] == 'Yes']['MonthlyIncome']
        
        # T-test
        t_stat, p_value = stats.ttest_ind(stayed, left)
        
        # Determine significance
        is_significant = p_value < 0.05
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(stayed)-1)*stayed.var() + (len(left)-1)*left.var()) / 
                           (len(stayed) + len(left) - 2))
        cohens_d = (stayed.mean() - left.mean()) / pooled_std
        
        # Interpretation
        if is_significant:
            if stayed.mean() > left.mean():
                interpretation = f"Employees who stayed earn significantly more (${stayed.mean():.0f} vs ${left.mean():.0f}, t={t_stat:.3f}, p={p_value:.4f})"
            else:
                interpretation = f"Employees who left earn significantly more (${left.mean():.0f} vs ${stayed.mean():.0f}, t={t_stat:.3f}, p={p_value:.4f})"
        else:
            interpretation = f"No significant income difference between employees who stayed and left (t={t_stat:.3f}, p={p_value:.4f})"
        
        return HypothesisResult(
            hypothesis="Income level affects attrition probability",
            test_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            effect_size=abs(cohens_d),
            interpretation=interpretation
        )
    
    def _test_satisfaction_attrition_relationship(self, df: pd.DataFrame) -> HypothesisResult:
        """Test relationship between job satisfaction and attrition."""
        if 'JobSatisfaction' not in df.columns or 'Attrition' not in df.columns:
            raise ValueError("Required columns not found for satisfaction-attrition relationship test")
        
        # Create contingency table
        crosstab = pd.crosstab(df['JobSatisfaction'], df['Attrition'])
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(crosstab)
        
        # Determine significance
        is_significant = p_value < 0.05
        
        # Calculate effect size (Cramér's V)
        n = crosstab.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))
        
        # Calculate attrition rates by satisfaction level
        attrition_rates = {}
        for satisfaction in crosstab.index:
            rate = crosstab.loc[satisfaction, 'Yes'] / crosstab.loc[satisfaction].sum() * 100
            attrition_rates[satisfaction] = rate
        
        # Interpretation
        if is_significant:
            interpretation = f"Significant relationship: Job satisfaction affects attrition (Low: {attrition_rates.get('Low', 0):.1f}%, Medium: {attrition_rates.get('Medium', 0):.1f}%, High: {attrition_rates.get('High', 0):.1f}%, χ²={chi2:.3f}, p={p_value:.4f})"
        else:
            interpretation = f"No significant relationship between job satisfaction and attrition (χ²={chi2:.3f}, p={p_value:.4f})"
        
        return HypothesisResult(
            hypothesis="Job satisfaction level affects attrition probability",
            test_statistic=chi2,
            p_value=p_value,
            is_significant=is_significant,
            effect_size=cramers_v,
            interpretation=interpretation
        )
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> StatsSummary:
        """
        Generate comprehensive statistical summary of the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            StatsSummary with comprehensive statistics
        """
        logger.info("Generating summary statistics...")
        
        # Numerical summary
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_summary = df[numerical_cols].describe()
        
        # Categorical summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_summary = {}
        for col in categorical_cols:
            categorical_summary[col] = df[col].value_counts()
        
        # Correlation matrix
        correlation_matrix = df[numerical_cols].corr() if len(numerical_cols) > 1 else pd.DataFrame()
        
        # Missing values
        missing_values = df.isnull().sum()
        
        # Data types
        data_types = df.dtypes
        
        return StatsSummary(
            numerical_summary=numerical_summary,
            categorical_summary=categorical_summary,
            correlation_matrix=correlation_matrix,
            missing_values=missing_values,
            data_types=data_types
        )

if __name__ == "__main__":
    # Test the EDAEngine with sample data
    eda = EDAEngine()
    
    # Load sample data
    df = pd.read_csv('data/hr_employee_data.csv')
    
    print("=== EDA Engine Test ===")
    
    # Generate correlation matrix
    corr_fig = eda.generate_correlation_matrix(df)
    corr_fig.savefig('reports/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("Correlation matrix saved to reports/correlation_matrix.png")
    
    # Analyze attrition by overtime
    overtime_analysis = eda.analyze_attrition_by_feature(df, 'OverTime')
    print(f"\nOvertime Analysis Insights:")
    for insight in overtime_analysis.insights:
        print(f"  - {insight}")
    
    # Test hypotheses
    distance_test = eda.test_hypothesis(df, "distance_attrition_correlation")
    print(f"\nDistance-Attrition Test: {distance_test.interpretation}")
    
    overtime_test = eda.test_hypothesis(df, "overtime_attrition_association")
    print(f"Overtime-Attrition Test: {overtime_test.interpretation}")
    
    # Generate summary statistics
    summary = eda.generate_summary_statistics(df)
    print(f"\nDataset Summary:")
    print(f"  - Numerical columns: {len(summary.numerical_summary.columns)}")
    print(f"  - Categorical columns: {len(summary.categorical_summary)}")
    print(f"  - Missing values: {summary.missing_values.sum()}")
    
    plt.close('all')  # Close all figures to free memory