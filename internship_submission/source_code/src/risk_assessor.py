"""
Risk Assessor for Employee Attrition Predictor

This module provides comprehensive risk assessment and business intelligence capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WatchList:
    """Watch list of high-risk employees."""
    employees: pd.DataFrame
    risk_threshold: float
    total_employees: int
    high_risk_count: int
    risk_distribution: Dict[str, int]
    generated_date: datetime

@dataclass
class PriorityList:
    """Prioritized list of employees for intervention."""
    employees: pd.DataFrame
    priority_factors: List[str]
    intervention_urgency: Dict[str, str]
    estimated_cost_savings: float

@dataclass
class BusinessImpact:
    """Business impact analysis of attrition predictions."""
    total_employees_at_risk: int
    estimated_turnover_cost: float
    potential_savings: float
    roi_projection: float
    department_breakdown: Dict[str, Dict[str, float]]
    cost_per_department: Dict[str, float]

class RiskAssessor:
    """
    Comprehensive risk assessor for employee attrition prediction.
    
    Provides risk scoring, watch list generation, and business impact analysis.
    """
    
    def __init__(self, risk_threshold: float = 0.7, replacement_cost_multiplier: float = 2.0):
        """
        Initialize the RiskAssessor.
        
        Args:
            risk_threshold: Threshold for high-risk classification (default: 0.7)
            replacement_cost_multiplier: Cost multiplier for employee replacement (default: 2.0 = 200%)
        """
        self.risk_threshold = risk_threshold
        self.replacement_cost_multiplier = replacement_cost_multiplier
        
    def calculate_risk_scores(self, df: pd.DataFrame, model) -> pd.DataFrame:
        """
        Calculate risk scores for all employees using trained model.
        
        Args:
            df: DataFrame with employee data
            model: Trained model for prediction
            
        Returns:
            DataFrame with risk scores added
        """
        logger.info("Calculating risk scores for all employees...")
        
        # Make a copy to avoid modifying original data
        df_with_scores = df.copy()
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            # Get probability of attrition (class 1)
            probabilities = model.predict_proba(df)
            risk_scores = probabilities[:, 1]  # Probability of 'Yes' (attrition)
        else:
            # Fallback to binary predictions
            predictions = model.predict(df)
            risk_scores = predictions.astype(float)
        
        # Ensure risk scores are bounded between 0 and 1
        risk_scores = np.clip(risk_scores, 0.0, 1.0)
        
        # Add risk scores to DataFrame
        df_with_scores['RiskScore'] = risk_scores
        
        # Add risk categories
        df_with_scores['RiskCategory'] = self._categorize_risk(risk_scores)
        
        # Add risk rank
        df_with_scores['RiskRank'] = df_with_scores['RiskScore'].rank(ascending=False, method='dense')
        
        logger.info(f"Risk scores calculated for {len(df_with_scores)} employees")
        return df_with_scores
    
    def _categorize_risk(self, risk_scores: np.ndarray) -> List[str]:
        """Categorize risk scores into Low, Medium, High categories."""
        categories = []
        for score in risk_scores:
            if score >= self.risk_threshold:
                categories.append('High')
            elif score >= 0.4:
                categories.append('Medium')
            else:
                categories.append('Low')
        return categories
    
    def generate_watch_list(self, risk_data: pd.DataFrame, 
                          performance_column: str = None) -> WatchList:
        """
        Generate watch list of high-risk employees with threshold filtering.
        
        Args:
            risk_data: DataFrame with risk scores
            performance_column: Optional column name for performance ratings
            
        Returns:
            WatchList with high-risk employees
        """
        logger.info(f"Generating watch list with threshold {self.risk_threshold}...")
        
        # Filter employees above risk threshold
        high_risk_employees = risk_data[risk_data['RiskScore'] >= self.risk_threshold].copy()
        
        # Sort by risk score (highest first)
        high_risk_employees = high_risk_employees.sort_values('RiskScore', ascending=False)
        
        # Add additional risk factors if available
        if performance_column and performance_column in risk_data.columns:
            high_risk_employees = high_risk_employees.sort_values(
                [performance_column, 'RiskScore'], 
                ascending=[False, False]
            )
        
        # Calculate risk distribution
        risk_distribution = risk_data['RiskCategory'].value_counts().to_dict()
        
        watch_list = WatchList(
            employees=high_risk_employees,
            risk_threshold=self.risk_threshold,
            total_employees=len(risk_data),
            high_risk_count=len(high_risk_employees),
            risk_distribution=risk_distribution,
            generated_date=datetime.now()
        )
        
        logger.info(f"Watch list generated with {len(high_risk_employees)} high-risk employees")
        return watch_list
    
    def segment_by_department(self, watch_list: WatchList) -> Dict[str, WatchList]:
        """
        Segment watch list by department for targeted interventions.
        
        Args:
            watch_list: WatchList to segment
            
        Returns:
            Dictionary of department-specific watch lists
        """
        logger.info("Segmenting watch list by department...")
        
        if 'Department' not in watch_list.employees.columns:
            logger.warning("Department column not found. Cannot segment by department.")
            return {'All': watch_list}
        
        department_watch_lists = {}
        
        for department in watch_list.employees['Department'].unique():
            dept_employees = watch_list.employees[
                watch_list.employees['Department'] == department
            ].copy()
            
            # Calculate department-specific risk distribution
            dept_risk_dist = dept_employees['RiskCategory'].value_counts().to_dict()
            
            dept_watch_list = WatchList(
                employees=dept_employees,
                risk_threshold=watch_list.risk_threshold,
                total_employees=len(dept_employees),
                high_risk_count=len(dept_employees),
                risk_distribution=dept_risk_dist,
                generated_date=watch_list.generated_date
            )
            
            department_watch_lists[department] = dept_watch_list
        
        logger.info(f"Watch list segmented into {len(department_watch_lists)} departments")
        return department_watch_lists
    
    def prioritize_interventions(self, watch_list: WatchList, 
                               performance_column: str = None,
                               salary_column: str = 'MonthlyIncome') -> PriorityList:
        """
        Prioritize employees for immediate intervention based on multiple factors.
        
        Args:
            watch_list: WatchList to prioritize
            performance_column: Column name for performance ratings
            salary_column: Column name for salary information
            
        Returns:
            PriorityList with prioritized employees
        """
        logger.info("Prioritizing employees for intervention...")
        
        employees = watch_list.employees.copy()
        
        # Priority factors
        priority_factors = ['RiskScore']
        
        # Add performance factor if available
        if performance_column and performance_column in employees.columns:
            priority_factors.append(performance_column)
            # Higher performance = higher priority to retain
            employees['PerformancePriority'] = employees[performance_column]
        
        # Add salary factor (higher salary = higher replacement cost)
        if salary_column in employees.columns:
            priority_factors.append(salary_column)
            employees['SalaryPriority'] = employees[salary_column]
        
        # Calculate composite priority score
        employees['PriorityScore'] = self._calculate_priority_score(
            employees, priority_factors, performance_column, salary_column
        )
        
        # Sort by priority score
        employees = employees.sort_values('PriorityScore', ascending=False)
        
        # Determine intervention urgency
        intervention_urgency = {}
        for idx, row in employees.iterrows():
            if row['RiskScore'] >= 0.8:
                urgency = 'Immediate'
            elif row['RiskScore'] >= 0.7:
                urgency = 'High'
            else:
                urgency = 'Medium'
            
            intervention_urgency[str(idx)] = urgency
        
        # Estimate cost savings from interventions
        estimated_savings = self._estimate_intervention_savings(employees, salary_column)
        
        priority_list = PriorityList(
            employees=employees,
            priority_factors=priority_factors,
            intervention_urgency=intervention_urgency,
            estimated_cost_savings=estimated_savings
        )
        
        logger.info(f"Prioritized {len(employees)} employees for intervention")
        return priority_list
    
    def _calculate_priority_score(self, employees: pd.DataFrame, 
                                priority_factors: List[str],
                                performance_column: str = None,
                                salary_column: str = 'MonthlyIncome') -> pd.Series:
        """Calculate composite priority score for employees."""
        # Base score from risk
        priority_score = employees['RiskScore'] * 0.5
        
        # Add performance component (higher performance = higher priority)
        if performance_column and performance_column in employees.columns:
            # Normalize performance to 0-1 scale
            perf_normalized = (employees[performance_column] - employees[performance_column].min()) / \
                            (employees[performance_column].max() - employees[performance_column].min())
            priority_score += perf_normalized * 0.3
        
        # Add salary component (higher salary = higher replacement cost)
        if salary_column in employees.columns:
            # Normalize salary to 0-1 scale
            salary_normalized = (employees[salary_column] - employees[salary_column].min()) / \
                              (employees[salary_column].max() - employees[salary_column].min())
            priority_score += salary_normalized * 0.2
        
        return priority_score
    
    def _estimate_intervention_savings(self, employees: pd.DataFrame, 
                                     salary_column: str = 'MonthlyIncome') -> float:
        """Estimate potential cost savings from successful interventions."""
        if salary_column not in employees.columns:
            return 0.0
        
        # Assume 50% success rate for interventions
        success_rate = 0.5
        
        # Calculate total replacement costs
        annual_salaries = employees[salary_column] * 12
        replacement_costs = annual_salaries * self.replacement_cost_multiplier
        
        # Estimate savings
        potential_savings = replacement_costs.sum() * success_rate
        
        return potential_savings
    
    def calculate_business_impact(self, predictions: pd.DataFrame, 
                                salary_column: str = 'MonthlyIncome') -> BusinessImpact:
        """
        Calculate comprehensive business impact of attrition predictions.
        
        Args:
            predictions: DataFrame with predictions and risk scores
            salary_column: Column name for salary information
            
        Returns:
            BusinessImpact with financial analysis
        """
        logger.info("Calculating business impact of attrition predictions...")
        
        # Count employees at risk
        high_risk_employees = predictions[predictions['RiskScore'] >= self.risk_threshold]
        total_at_risk = len(high_risk_employees)
        
        # Calculate replacement costs
        if salary_column in predictions.columns:
            # Annual salaries for at-risk employees
            annual_salaries = high_risk_employees[salary_column] * 12
            
            # Total replacement cost (200% of annual salary)
            total_replacement_cost = (annual_salaries * self.replacement_cost_multiplier).sum()
            
            # Potential savings (assuming 30% intervention success rate)
            intervention_success_rate = 0.3
            potential_savings = total_replacement_cost * intervention_success_rate
            
            # ROI calculation (assuming intervention cost is 10% of replacement cost)
            intervention_cost = total_replacement_cost * 0.1
            roi_projection = (potential_savings - intervention_cost) / intervention_cost if intervention_cost > 0 else 0
        else:
            total_replacement_cost = 0
            potential_savings = 0
            roi_projection = 0
        
        # Department breakdown
        department_breakdown = {}
        cost_per_department = {}
        
        if 'Department' in predictions.columns:
            for department in predictions['Department'].unique():
                dept_data = predictions[predictions['Department'] == department]
                dept_high_risk = dept_data[dept_data['RiskScore'] >= self.risk_threshold]
                
                dept_stats = {
                    'total_employees': len(dept_data),
                    'high_risk_employees': len(dept_high_risk),
                    'risk_percentage': (len(dept_high_risk) / len(dept_data)) * 100 if len(dept_data) > 0 else 0,
                    'average_risk_score': dept_high_risk['RiskScore'].mean() if len(dept_high_risk) > 0 else 0
                }
                
                department_breakdown[department] = dept_stats
                
                # Calculate cost per department
                if salary_column in dept_high_risk.columns and len(dept_high_risk) > 0:
                    dept_annual_salaries = dept_high_risk[salary_column] * 12
                    dept_replacement_cost = (dept_annual_salaries * self.replacement_cost_multiplier).sum()
                    cost_per_department[department] = dept_replacement_cost
                else:
                    cost_per_department[department] = 0
        
        business_impact = BusinessImpact(
            total_employees_at_risk=total_at_risk,
            estimated_turnover_cost=total_replacement_cost,
            potential_savings=potential_savings,
            roi_projection=roi_projection,
            department_breakdown=department_breakdown,
            cost_per_department=cost_per_department
        )
        
        logger.info(f"Business impact calculated: {total_at_risk} employees at risk, "
                   f"${total_replacement_cost:,.0f} potential cost")
        return business_impact
    
    def generate_risk_dashboard(self, risk_data: pd.DataFrame, 
                              business_impact: BusinessImpact) -> plt.Figure:
        """Generate comprehensive risk assessment dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Risk distribution
        risk_counts = risk_data['RiskCategory'].value_counts()
        axes[0, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Risk Distribution')
        
        # Risk scores histogram
        axes[0, 1].hist(risk_data['RiskScore'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(self.risk_threshold, color='red', linestyle='--', 
                          label=f'Risk Threshold ({self.risk_threshold})')
        axes[0, 1].set_xlabel('Risk Score')
        axes[0, 1].set_ylabel('Number of Employees')
        axes[0, 1].set_title('Risk Score Distribution')
        axes[0, 1].legend()
        
        # Department risk breakdown
        if 'Department' in risk_data.columns:
            dept_risk = risk_data.groupby('Department')['RiskScore'].mean().sort_values(ascending=False)
            axes[1, 0].bar(range(len(dept_risk)), dept_risk.values)
            axes[1, 0].set_xticks(range(len(dept_risk)))
            axes[1, 0].set_xticklabels(dept_risk.index, rotation=45)
            axes[1, 0].set_ylabel('Average Risk Score')
            axes[1, 0].set_title('Average Risk Score by Department')
        
        # Business impact summary
        impact_data = [
            business_impact.total_employees_at_risk,
            business_impact.estimated_turnover_cost / 1000,  # In thousands
            business_impact.potential_savings / 1000,  # In thousands
            business_impact.roi_projection * 100  # As percentage
        ]
        impact_labels = ['Employees\nat Risk', 'Est. Cost\n($K)', 'Potential\nSavings ($K)', 'ROI\n(%)']
        
        bars = axes[1, 1].bar(impact_labels, impact_data)
        axes[1, 1].set_title('Business Impact Summary')
        
        # Add value labels on bars
        for bar, value in zip(bars, impact_data):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    # Test the RiskAssessor with sample data
    from feature_encoder import FeatureEncoder
    from model_trainer import ModelTrainer
    from sklearn.model_selection import train_test_split
    
    assessor = RiskAssessor()
    
    # Load and prepare data
    df = pd.read_csv('data/hr_employee_data.csv')
    
    # Encode features
    encoder = FeatureEncoder()
    X = encoder.fit_transform(df.drop('Attrition', axis=1))
    y = df['Attrition']
    
    # Train a quick model for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    trainer = ModelTrainer()
    model = trainer.train_random_forest(X_train, y_train)
    
    print("=== Risk Assessment Test ===")
    
    # Calculate risk scores
    risk_data = assessor.calculate_risk_scores(X_test, model)
    print(f"Risk scores calculated for {len(risk_data)} employees")
    
    # Generate watch list
    watch_list = assessor.generate_watch_list(risk_data)
    print(f"Watch list generated: {watch_list.high_risk_count} high-risk employees")
    
    # Calculate business impact
    # Add salary column for testing
    risk_data['MonthlyIncome'] = np.random.normal(5000, 1000, len(risk_data))
    risk_data['Department'] = np.random.choice(['Sales', 'R&D', 'HR'], len(risk_data))
    
    business_impact = assessor.calculate_business_impact(risk_data)
    print(f"\nBusiness Impact:")
    print(f"  - Employees at risk: {business_impact.total_employees_at_risk}")
    print(f"  - Estimated turnover cost: ${business_impact.estimated_turnover_cost:,.0f}")
    print(f"  - Potential savings: ${business_impact.potential_savings:,.0f}")
    print(f"  - ROI projection: {business_impact.roi_projection:.1%}")
    
    # Generate dashboard
    dashboard = assessor.generate_risk_dashboard(risk_data, business_impact)
    dashboard.savefig('reports/risk_dashboard.png', dpi=300, bbox_inches='tight')
    print("\nRisk dashboard saved to reports/risk_dashboard.png")
    
    plt.close(dashboard)