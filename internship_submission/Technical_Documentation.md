# Employee Attrition Predictor - Technical Documentation

**Version:** 1.0  
**Date:** January 17, 2026  
**Author:** Nihal Sunil Khare  

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Design](#architecture-design)
3. [Data Pipeline](#data-pipeline)
4. [Machine Learning Models](#machine-learning-models)
5. [API Documentation](#api-documentation)
6. [Deployment Guide](#deployment-guide)
7. [Testing Strategy](#testing-strategy)
8. [Performance Metrics](#performance-metrics)

---

## 1. System Overview

### 1.1 Purpose
The Employee Attrition Predictor is an enterprise-grade HR analytics system that uses machine learning to predict employee turnover and provide actionable business insights.

### 1.2 Key Components
- **Data Validator:** Ensures data quality and integrity
- **Feature Encoder:** Handles categorical and numerical feature processing
- **EDA Engine:** Generates exploratory data analysis and visualizations
- **Model Trainer:** Trains and evaluates multiple ML algorithms
- **Risk Assessor:** Calculates business impact and employee risk scores
- **Web Interface:** Interactive Streamlit dashboard

### 1.3 Technology Stack
```
Backend:     Python 3.8+, scikit-learn, pandas, numpy
Frontend:    Streamlit, plotly, matplotlib, seaborn
Testing:     pytest, Hypothesis (property-based testing)
Deployment:  Docker, Streamlit Cloud
Storage:     CSV files, pickle models
```

---

## 2. Architecture Design

### 2.1 System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Input    │───▶│  Data Pipeline  │───▶│   ML Pipeline   │
│   (CSV Files)   │    │  (Validation &  │    │  (Training &    │
│                 │    │   Encoding)     │    │   Evaluation)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Web Interface  │◀───│  Risk Assessor  │◀───│  Trained Models │
│  (Streamlit)    │    │  (Business BI)  │    │  (Pickle Files) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2.2 Data Flow
1. **Input:** Raw HR data (CSV format)
2. **Validation:** Data quality checks and validation
3. **Preprocessing:** Feature encoding and normalization
4. **Training:** Multiple ML model training and evaluation
5. **Assessment:** Risk scoring and business impact analysis
6. **Visualization:** Interactive dashboard and reporting

---

## 3. Data Pipeline

### 3.1 Data Validator (`src/data_validator.py`)
```python
class DataValidator:
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, int]
    def validate_data_types(self, df: pd.DataFrame) -> Dict[str, bool]
    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, List[int]]
```

**Key Features:**
- Missing value detection and reporting
- Data type validation
- Outlier detection using IQR method
- Comprehensive quality scoring

### 3.2 Feature Encoder (`src/feature_encoder.py`)
```python
class FeatureEncoder:
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame
    def normalize_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame
    def prepare_features_for_training(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]
```

**Encoding Strategies:**
- **Label Encoding:** Ordinal variables (JobSatisfaction: Low=1, Medium=2, High=3)
- **One-Hot Encoding:** Nominal variables (Department, JobRole)
- **Normalization:** StandardScaler for numerical features

---

## 4. Machine Learning Models

### 4.1 Model Trainer (`src/model_trainer.py`)
```python
class ModelTrainer:
    def train_logistic_regression(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression
    def train_random_forest(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier
    def train_decision_tree(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]
```

### 4.2 Model Configurations
```yaml
# config/model_config.yaml
models:
  logistic_regression:
    max_iter: 1000
    random_state: 42
  
  random_forest:
    n_estimators: 100
    max_depth: 10
    random_state: 42
  
  decision_tree:
    max_depth: 10
    min_samples_split: 20
    random_state: 42
```

### 4.3 Performance Metrics
- **Accuracy:** Overall prediction correctness
- **Precision:** True positive rate (minimize false alarms)
- **Recall:** Sensitivity (catch actual attrition cases)
- **F1-Score:** Harmonic mean of precision and recall
- **AUC-ROC:** Area under the receiver operating characteristic curve

---

## 5. API Documentation

### 5.1 Risk Assessor (`src/risk_assessor.py`)
```python
class RiskAssessor:
    def calculate_employee_risk_scores(self, df: pd.DataFrame, model) -> pd.DataFrame
    def identify_high_risk_employees(self, risk_scores: pd.DataFrame) -> List[Dict]
    def calculate_business_impact(self, predictions: np.ndarray) -> Dict[str, float]
    def generate_retention_recommendations(self, high_risk_employees: List[Dict]) -> List[str]
```

### 5.2 Business Intelligence Functions
```python
def calculate_turnover_cost(num_employees: int, avg_salary: float = 50000) -> float:
    """Calculate estimated turnover cost (200% of salary per employee)"""
    return num_employees * avg_salary * 2.0

def calculate_intervention_roi(potential_savings: float, intervention_cost: float) -> float:
    """Calculate return on investment for retention interventions"""
    return ((potential_savings - intervention_cost) / intervention_cost) * 100
```

---

## 6. Deployment Guide

### 6.1 Local Development
```bash
# Clone repository
git clone https://github.com/NihalsmK/employee-attrition-predictor.git
cd employee-attrition-predictor

# Install dependencies
pip install -r requirements_deploy.txt

# Run training pipeline
python main.py

# Launch web application
streamlit run app.py
```

### 6.2 Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_deploy.txt .
RUN pip install -r requirements_deploy.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 6.3 Cloud Deployment
- **Streamlit Cloud:** Direct GitHub integration
- **AWS:** ECS, App Runner, or Lambda
- **Google Cloud:** Cloud Run or App Engine
- **Azure:** Container Instances or App Service

---

## 7. Testing Strategy

### 7.1 Property-Based Testing
```python
# tests/test_data_validator_properties.py
@given(st.data())
def test_data_quality_score_bounds(data):
    """Property: Data quality scores should always be between 0 and 1"""
    df = data.draw(generate_hr_dataframe())
    validator = DataValidator()
    quality_report = validator.validate_data_quality(df)
    assert 0 <= quality_report['overall_quality_score'] <= 1
```

### 7.2 Unit Testing
```python
def test_feature_encoder_categorical():
    """Test categorical feature encoding produces expected output"""
    encoder = FeatureEncoder()
    test_df = pd.DataFrame({'Department': ['Sales', 'HR', 'Sales']})
    encoded_df = encoder.encode_categorical_features(test_df)
    assert 'Department_Sales' in encoded_df.columns
    assert 'Department_HR' in encoded_df.columns
```

### 7.3 Integration Testing
- End-to-end pipeline testing
- Model training and prediction workflows
- Web application functionality testing

---

## 8. Performance Metrics

### 8.1 System Performance
- **Training Time:** ~30 seconds for 1,470 records
- **Prediction Time:** <1 second for single employee
- **Memory Usage:** ~50MB for loaded models
- **Disk Space:** ~10MB for all models and data

### 8.2 Model Performance
```
Decision Tree (Best Model):
├── Accuracy: 67.0%
├── Precision: 61.6%
├── Recall: 62.6%
├── F1-Score: 62.1%
└── AUC: 54.3%

Feature Importance:
├── MonthlyIncome: 19.5%
├── Age: 13.5%
├── DistanceFromHome: 9.0%
├── YearsAtCompany: 8.5%
└── NumCompaniesWorked: 8.5%
```

### 8.3 Business Metrics
- **High-Risk Employees:** 446 (30.3% of workforce)
- **Estimated Cost Avoidance:** $121,102,870
- **ROI on Interventions:** 1,012%
- **Processing Capacity:** 10,000+ employees per minute

---

## 9. Configuration Management

### 9.1 Environment Variables
```bash
# .env (optional)
MODEL_PATH=models/
DATA_PATH=data/
REPORTS_PATH=reports/
LOG_LEVEL=INFO
```

### 9.2 Model Configuration
```yaml
# config/model_config.yaml
training:
  test_size: 0.2
  random_state: 42
  cross_validation_folds: 5

business_rules:
  high_risk_threshold: 0.7
  avg_salary: 50000
  turnover_cost_multiplier: 2.0
  intervention_cost_per_employee: 90
```

---

## 10. Monitoring and Maintenance

### 10.1 Model Monitoring
- **Performance Drift:** Monitor accuracy degradation over time
- **Data Drift:** Detect changes in input data distribution
- **Prediction Distribution:** Track prediction score distributions

### 10.2 Maintenance Schedule
- **Weekly:** Data quality reports
- **Monthly:** Model performance evaluation
- **Quarterly:** Model retraining with new data
- **Annually:** Architecture review and updates

---

**Repository:** https://github.com/NihalsmK/employee-attrition-predictor  
**Documentation:** Complete technical specifications available in `/docs/specs/`  
**Support:** nihalsmkhare@gmail.com  

---

*This technical documentation provides comprehensive implementation details for the Employee Attrition Predictor system, suitable for development teams and technical stakeholders.*