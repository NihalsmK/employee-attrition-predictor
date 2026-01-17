"""
Employee Attrition Predictor - Main Application

This is the main entry point for the Employee Attrition Prediction system.
It demonstrates the complete workflow from data loading to business insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
import os

# Import our custom modules
from src.data_validator import DataValidator
from src.feature_encoder import FeatureEncoder
from src.eda_engine import EDAEngine
from src.model_trainer import ModelTrainer
from src.risk_assessor import RiskAssessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class EmployeeAttritionPredictor:
    """
    Main class for the Employee Attrition Prediction system.
    
    Orchestrates the complete workflow from data validation to business insights.
    """
    
    def __init__(self):
        """Initialize the prediction system."""
        self.validator = DataValidator()
        self.encoder = FeatureEncoder()
        self.eda = EDAEngine()
        self.trainer = ModelTrainer()
        self.assessor = RiskAssessor(risk_threshold=0.5)  # Lower threshold for demo
        
        self.raw_data = None
        self.processed_data = None
        self.models = {}
        self.best_model = None
        self.risk_data = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and validate HR data."""
        logger.info(f"Loading data from {filepath}...")
        
        try:
            self.raw_data = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully: {self.raw_data.shape}")
            
            # Validate data
            validation_report = self.validator.generate_validation_report(self.raw_data)
            
            logger.info("Data Validation Summary:")
            logger.info(f"  - Schema Valid: {validation_report['overall_status']['schema_valid']}")
            logger.info(f"  - Data Complete: {validation_report['overall_status']['data_complete']}")
            logger.info(f"  - Outliers Detected: {validation_report['outlier_detection'].total_outliers}")
            logger.info(f"  - Business Rule Violations: {validation_report['business_rules'].total_violations}")
            
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def perform_eda(self) -> Dict[str, Any]:
        """Perform comprehensive exploratory data analysis."""
        logger.info("Performing Exploratory Data Analysis...")
        
        if self.raw_data is None:
            raise ValueError("Data must be loaded before performing EDA")
        
        eda_results = {}
        
        # Generate correlation matrix
        corr_fig = self.eda.generate_correlation_matrix(self.raw_data)
        corr_fig.savefig('reports/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close(corr_fig)
        eda_results['correlation_matrix'] = 'reports/correlation_matrix.png'
        
        # Analyze key features
        key_features = ['OverTime', 'JobSatisfaction', 'Department']
        feature_analyses = {}
        
        for feature in key_features:
            if feature in self.raw_data.columns:
                analysis = self.eda.analyze_attrition_by_feature(self.raw_data, feature)
                feature_analyses[feature] = {
                    'insights': analysis.insights,
                    'results': analysis.results
                }
                
                # Save visualizations
                for i, fig in enumerate(analysis.visualizations):
                    fig_path = f'reports/{feature.lower()}_analysis_{i}.png'
                    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
        
        eda_results['feature_analyses'] = feature_analyses
        
        # Test key hypotheses
        hypotheses = [
            "distance_attrition_correlation",
            "overtime_attrition_association",
            "income_attrition_relationship",
            "satisfaction_attrition_relationship"
        ]
        
        hypothesis_results = {}
        for hypothesis in hypotheses:
            try:
                result = self.eda.test_hypothesis(self.raw_data, hypothesis)
                hypothesis_results[hypothesis] = {
                    'is_significant': result.is_significant,
                    'p_value': result.p_value,
                    'interpretation': result.interpretation
                }
            except Exception as e:
                logger.warning(f"Could not test hypothesis {hypothesis}: {e}")
        
        eda_results['hypothesis_tests'] = hypothesis_results
        
        # Generate summary statistics
        summary = self.eda.generate_summary_statistics(self.raw_data)
        eda_results['summary_stats'] = {
            'numerical_columns': len(summary.numerical_summary.columns),
            'categorical_columns': len(summary.categorical_summary),
            'missing_values': summary.missing_values.sum(),
            'total_records': len(self.raw_data)
        }
        
        logger.info("EDA completed successfully")
        return eda_results
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for machine learning."""
        logger.info("Preparing data for machine learning...")
        
        if self.raw_data is None:
            raise ValueError("Data must be loaded before preparation")
        
        # Separate features and target
        X = self.raw_data.drop('Attrition', axis=1)
        y = self.raw_data['Attrition']
        
        # Encode features
        X_encoded = self.encoder.fit_transform(X)
        
        # Save encoders
        self.encoder.save_encoders('models/feature_encoders.pkl')
        
        self.processed_data = (X_encoded, y)
        
        logger.info(f"Data prepared: {X_encoded.shape} features, {len(y)} samples")
        return X_encoded, y
    
    def train_models(self) -> Dict[str, Any]:
        """Train multiple machine learning models."""
        logger.info("Training machine learning models...")
        
        if self.processed_data is None:
            raise ValueError("Data must be prepared before training models")
        
        X, y = self.processed_data
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        logger.info("Training Logistic Regression...")
        lr_model = self.trainer.train_logistic_regression(X_train, y_train)
        
        logger.info("Training Random Forest...")
        rf_model = self.trainer.train_random_forest(X_train, y_train)
        
        logger.info("Training Decision Tree...")
        dt_model = self.trainer.train_decision_tree(X_train, y_train)
        
        self.models = {
            'logistic_regression': lr_model,
            'random_forest': rf_model,
            'decision_tree': dt_model
        }
        
        # Compare models
        models_list = list(self.models.values())
        comparison = self.trainer.compare_models(models_list, X_test, y_test)
        
        # Select best model
        best_model_name = comparison.best_model.lower().replace(' ', '_')
        self.best_model = self.models[best_model_name]
        
        # Save best model
        self.trainer.save_model(self.best_model, f'models/best_model.pkl')
        
        # Extract feature importance
        importance = self.trainer.extract_feature_importance(self.best_model)
        if importance.visualization:
            importance.visualization.savefig('reports/feature_importance.png', 
                                           dpi=300, bbox_inches='tight')
            plt.close(importance.visualization)
        
        training_results = {
            'comparison': comparison,
            'best_model': comparison.best_model,
            'feature_importance': importance.importance_ranking[:10],
            'recommendations': comparison.recommendations
        }
        
        logger.info(f"Model training completed. Best model: {comparison.best_model}")
        return training_results
    
    def assess_risk(self) -> Dict[str, Any]:
        """Perform comprehensive risk assessment."""
        logger.info("Performing risk assessment...")
        
        if self.best_model is None:
            raise ValueError("Models must be trained before risk assessment")
        
        X, y = self.processed_data
        
        # Calculate risk scores
        self.risk_data = self.assessor.calculate_risk_scores(X, self.best_model)
        
        # Add original data for context
        original_columns = ['Age', 'Department', 'MonthlyIncome', 'JobSatisfaction', 'OverTime']
        for col in original_columns:
            if col in self.raw_data.columns:
                self.risk_data[col] = self.raw_data[col].values
        
        # Generate watch list
        watch_list = self.assessor.generate_watch_list(self.risk_data)
        
        # Segment by department
        dept_watch_lists = self.assessor.segment_by_department(watch_list)
        
        # Calculate business impact
        business_impact = self.assessor.calculate_business_impact(self.risk_data)
        
        # Generate risk dashboard
        dashboard = self.assessor.generate_risk_dashboard(self.risk_data, business_impact)
        dashboard.savefig('reports/risk_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close(dashboard)
        
        risk_results = {
            'watch_list': watch_list,
            'department_segments': dept_watch_lists,
            'business_impact': business_impact,
            'risk_summary': {
                'total_employees': len(self.risk_data),
                'high_risk_count': watch_list.high_risk_count,
                'risk_percentage': (watch_list.high_risk_count / len(self.risk_data)) * 100,
                'estimated_cost': business_impact.estimated_turnover_cost,
                'potential_savings': business_impact.potential_savings
            }
        }
        
        logger.info(f"Risk assessment completed. {watch_list.high_risk_count} high-risk employees identified")
        return risk_results
    
    def generate_executive_summary(self, eda_results: Dict, training_results: Dict, 
                                 risk_results: Dict) -> str:
        """Generate executive summary report."""
        logger.info("Generating executive summary...")
        
        summary = f"""
# Employee Attrition Prediction - Executive Summary

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Findings

### Data Overview
- **Total Employees Analyzed:** {eda_results['summary_stats']['total_records']:,}
- **Data Quality:** {eda_results['summary_stats']['missing_values']} missing values detected
- **Features Analyzed:** {eda_results['summary_stats']['numerical_columns']} numerical, {eda_results['summary_stats']['categorical_columns']} categorical

### Model Performance
- **Best Performing Model:** {training_results['best_model']}
- **Model Accuracy:** {training_results['comparison'].metrics_comparison.loc[0, 'Accuracy']:.1%}
- **Key Predictive Features:**
"""
        
        # Add top features
        for i, (feature, importance) in enumerate(training_results['feature_importance'][:5]):
            summary += f"  {i+1}. {feature} (importance: {importance:.3f})\n"
        
        summary += f"""
### Risk Assessment
- **Employees at High Risk:** {risk_results['risk_summary']['high_risk_count']} ({risk_results['risk_summary']['risk_percentage']:.1f}%)
- **Estimated Turnover Cost:** ${risk_results['risk_summary']['estimated_cost']:,.0f}
- **Potential Savings:** ${risk_results['risk_summary']['potential_savings']:,.0f}

### Business Insights
"""
        
        # Add hypothesis test results
        for hypothesis, result in eda_results['hypothesis_tests'].items():
            if result['is_significant']:
                summary += f"- **{hypothesis.replace('_', ' ').title()}:** {result['interpretation']}\n"
        
        summary += f"""
### Recommendations
"""
        
        # Add model recommendations
        for rec in training_results['recommendations']:
            summary += f"- {rec}\n"
        
        summary += f"""
### Department Breakdown
"""
        
        # Add department risk breakdown
        for dept, watch_list in risk_results['department_segments'].items():
            if len(watch_list.employees) > 0:
                summary += f"- **{dept}:** {len(watch_list.employees)} high-risk employees\n"
        
        # Save summary
        with open('reports/executive_summary.md', 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info("Executive summary generated")
        return summary
    
    def run_complete_analysis(self, data_filepath: str) -> Dict[str, Any]:
        """Run the complete end-to-end analysis."""
        logger.info("Starting complete Employee Attrition Analysis...")
        
        try:
            # Create reports directory
            os.makedirs('reports', exist_ok=True)
            os.makedirs('models', exist_ok=True)
            
            # Step 1: Load and validate data
            self.load_data(data_filepath)
            
            # Step 2: Exploratory Data Analysis
            eda_results = self.perform_eda()
            
            # Step 3: Prepare data for ML
            self.prepare_data()
            
            # Step 4: Train models
            training_results = self.train_models()
            
            # Step 5: Risk assessment
            risk_results = self.assess_risk()
            
            # Step 6: Generate executive summary
            executive_summary = self.generate_executive_summary(
                eda_results, training_results, risk_results
            )
            
            # Compile final results
            final_results = {
                'eda': eda_results,
                'training': training_results,
                'risk_assessment': risk_results,
                'executive_summary': executive_summary,
                'artifacts': {
                    'models': 'models/',
                    'reports': 'reports/',
                    'encoders': 'models/feature_encoders.pkl'
                }
            }
            
            logger.info("Complete analysis finished successfully!")
            return final_results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

def main():
    """Main function to run the Employee Attrition Predictor."""
    print("=" * 60)
    print("Employee Attrition Predictor")
    print("HR Analytics & Machine Learning System")
    print("=" * 60)
    
    # Initialize the system
    predictor = EmployeeAttritionPredictor()
    
    # Run complete analysis
    try:
        results = predictor.run_complete_analysis('data/hr_employee_data.csv')
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        
        print(f"\nKey Results:")
        print(f"- Best Model: {results['training']['best_model']}")
        print(f"- High-Risk Employees: {results['risk_assessment']['risk_summary']['high_risk_count']}")
        print(f"- Potential Cost Savings: ${results['risk_assessment']['risk_summary']['potential_savings']:,.0f}")
        
        print(f"\nGenerated Artifacts:")
        print(f"- Executive Summary: reports/executive_summary.md")
        print(f"- Risk Dashboard: reports/risk_dashboard.png")
        print(f"- Feature Importance: reports/feature_importance.png")
        print(f"- Correlation Matrix: reports/correlation_matrix.png")
        print(f"- Best Model: models/best_model.pkl")
        
        print(f"\nTop 3 Attrition Drivers:")
        for i, (feature, importance) in enumerate(results['training']['feature_importance'][:3]):
            print(f"  {i+1}. {feature} (importance: {importance:.3f})")
        
        print("\n" + "=" * 60)
        print("Ready for business presentation and deployment!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        logger.error(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()