"""
Employee Attrition Predictor - Streamlit Web Application

A user-friendly web interface for HR analytics and employee attrition prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime

# Import our modules
from src.data_validator import DataValidator
from src.feature_encoder import FeatureEncoder
from src.eda_engine import EDAEngine
from src.model_trainer import ModelTrainer
from src.risk_assessor import RiskAssessor

# Configure page
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high { color: #d62728; font-weight: bold; }
    .risk-medium { color: #ff7f0e; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample HR data."""
    return pd.read_csv('data/hr_employee_data.csv')

@st.cache_resource
def load_trained_model():
    """Load pre-trained model and encoders."""
    try:
        # Load model
        with open('models/best_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Load encoder
        encoder = FeatureEncoder()
        encoder.load_encoders('models/feature_encoders.pkl')
        
        return model_data, encoder
    except FileNotFoundError:
        return None, None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">👥 Employee Attrition Predictor</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered HR Analytics for Employee Retention**")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "📊 Data Analysis", "🤖 Predictions", "📈 Risk Assessment", "📋 Individual Prediction"]
    )
    
    # Load data and models
    df = load_sample_data()
    model_data, encoder = load_trained_model()
    
    if page == "🏠 Dashboard":
        show_dashboard(df)
    elif page == "📊 Data Analysis":
        show_data_analysis(df)
    elif page == "🤖 Predictions":
        show_predictions(df, model_data, encoder)
    elif page == "📈 Risk Assessment":
        show_risk_assessment(df, model_data, encoder)
    elif page == "📋 Individual Prediction":
        show_individual_prediction(model_data, encoder)

def show_dashboard(df):
    """Show main dashboard with key metrics."""
    st.header("📊 HR Analytics Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", f"{len(df):,}")
    
    with col2:
        attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
        st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    
    with col3:
        avg_tenure = df['YearsAtCompany'].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} years")
    
    with col4:
        avg_income = df['MonthlyIncome'].mean()
        st.metric("Avg Monthly Income", f"${avg_income:,.0f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attrition by Department")
        dept_attrition = df.groupby('Department')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).sort_values(ascending=False)
        
        fig = px.bar(
            x=dept_attrition.index,
            y=dept_attrition.values,
            labels={'x': 'Department', 'y': 'Attrition Rate (%)'},
            title="Attrition Rate by Department"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Overtime vs Attrition")
        overtime_attrition = pd.crosstab(df['OverTime'], df['Attrition'], normalize='index') * 100
        
        fig = px.bar(
            overtime_attrition,
            title="Attrition Rate by Overtime Status",
            labels={'value': 'Percentage', 'index': 'Overtime Status'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Age distribution
    st.subheader("Age Distribution by Attrition")
    fig = px.histogram(
        df, x='Age', color='Attrition', 
        title="Age Distribution by Attrition Status",
        nbins=20, opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

def show_data_analysis(df):
    """Show detailed data analysis."""
    st.header("📊 Exploratory Data Analysis")
    
    # Data overview
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Missing Values:**", df.isnull().sum().sum())
        st.write("**Numerical Columns:**", len(df.select_dtypes(include=[np.number]).columns))
        st.write("**Categorical Columns:**", len(df.select_dtypes(include=['object']).columns))
    
    with col2:
        st.write("**Sample Data:**")
        st.dataframe(df.head())
    
    # Correlation matrix
    st.subheader("Correlation Matrix")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Correlation Matrix of Numerical Features",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    # Select feature to analyze
    feature = st.selectbox("Select feature to analyze:", df.columns[:-1])  # Exclude target
    
    if df[feature].dtype in ['object']:
        # Categorical feature
        fig = px.countplot(data=df, x=feature, color='Attrition')
        st.plotly_chart(fig, use_container_width=True)
        
        # Cross-tabulation
        crosstab = pd.crosstab(df[feature], df['Attrition'])
        st.write("**Cross-tabulation:**")
        st.dataframe(crosstab)
        
    else:
        # Numerical feature
        fig = px.histogram(
            df, x=feature, color='Attrition',
            title=f"{feature} Distribution by Attrition",
            opacity=0.7, nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        summary = df.groupby('Attrition')[feature].describe()
        st.write("**Summary Statistics:**")
        st.dataframe(summary)

def show_predictions(df, model_data, encoder):
    """Show model predictions and performance."""
    st.header("🤖 Model Predictions & Performance")
    
    if model_data is None or encoder is None:
        st.error("⚠️ Pre-trained model not found. Please run the training pipeline first.")
        st.code("python main.py")
        return
    
    # Model info
    st.subheader("Model Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", model_data['model_name'])
    
    with col2:
        st.metric("Features", len(model_data['feature_names']))
    
    with col3:
        training_time = model_data.get('training_time', 0)
        st.metric("Training Time", f"{training_time:.2f}s")
    
    # Feature importance
    st.subheader("Feature Importance")
    
    if hasattr(model_data['model'], 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': model_data['feature_names'],
            'Importance': model_data['model'].feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig = px.bar(
            importance_df, x='Importance', y='Feature',
            orientation='h', title="Top 10 Most Important Features"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Predictions on sample data
    st.subheader("Sample Predictions")
    
    # Prepare sample data
    X_sample = df.drop('Attrition', axis=1).head(10)
    X_encoded = encoder.transform(X_sample)
    
    # Make predictions
    predictions = model_data['model'].predict(X_encoded)
    if hasattr(model_data['model'], 'predict_proba'):
        probabilities = model_data['model'].predict_proba(X_encoded)[:, 1]
    else:
        probabilities = predictions.astype(float)
    
    # Create results dataframe
    results_df = X_sample.copy()
    results_df['Actual_Attrition'] = df['Attrition'].head(10)
    results_df['Predicted_Attrition'] = ['Yes' if p == 1 else 'No' for p in predictions]
    results_df['Risk_Score'] = probabilities
    results_df['Risk_Category'] = ['High' if p >= 0.7 else 'Medium' if p >= 0.4 else 'Low' for p in probabilities]
    
    # Display results
    st.dataframe(results_df[['Age', 'Department', 'MonthlyIncome', 'Actual_Attrition', 
                            'Predicted_Attrition', 'Risk_Score', 'Risk_Category']])

def show_risk_assessment(df, model_data, encoder):
    """Show comprehensive risk assessment."""
    st.header("📈 Risk Assessment Dashboard")
    
    if model_data is None or encoder is None:
        st.error("⚠️ Pre-trained model not found. Please run the training pipeline first.")
        return
    
    # Initialize risk assessor
    assessor = RiskAssessor(risk_threshold=0.5)
    
    # Prepare data
    X = df.drop('Attrition', axis=1)
    X_encoded = encoder.transform(X)
    
    # Calculate risk scores
    risk_data = assessor.calculate_risk_scores(X_encoded, model_data['model'])
    
    # Add original data for context
    for col in ['Age', 'Department', 'MonthlyIncome', 'JobSatisfaction', 'OverTime']:
        if col in df.columns:
            risk_data[col] = df[col].values
    
    # Generate watch list
    watch_list = assessor.generate_watch_list(risk_data)
    
    # Calculate business impact
    business_impact = assessor.calculate_business_impact(risk_data)
    
    # Display metrics
    st.subheader("Risk Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("High-Risk Employees", watch_list.high_risk_count)
    
    with col2:
        risk_percentage = (watch_list.high_risk_count / len(risk_data)) * 100
        st.metric("Risk Percentage", f"{risk_percentage:.1f}%")
    
    with col3:
        st.metric("Estimated Cost", f"${business_impact.estimated_turnover_cost:,.0f}")
    
    with col4:
        st.metric("Potential Savings", f"${business_impact.potential_savings:,.0f}")
    
    # Risk distribution
    st.subheader("Risk Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        risk_counts = risk_data['RiskCategory'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Employee Risk Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            risk_data, x='RiskScore', nbins=20,
            title="Risk Score Distribution"
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                     annotation_text="Risk Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    # Department breakdown
    st.subheader("Risk by Department")
    dept_risk = risk_data.groupby('Department').agg({
        'RiskScore': 'mean',
        'RiskCategory': lambda x: (x == 'High').sum()
    }).round(3)
    dept_risk.columns = ['Avg_Risk_Score', 'High_Risk_Count']
    
    fig = px.bar(
        dept_risk.reset_index(),
        x='Department', y='Avg_Risk_Score',
        title="Average Risk Score by Department"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # High-risk employees table
    st.subheader("High-Risk Employees")
    high_risk_employees = watch_list.employees.head(20)
    
    if len(high_risk_employees) > 0:
        display_cols = ['Age', 'Department', 'MonthlyIncome', 'JobSatisfaction', 
                       'OverTime', 'RiskScore', 'RiskCategory']
        available_cols = [col for col in display_cols if col in high_risk_employees.columns]
        st.dataframe(high_risk_employees[available_cols])
    else:
        st.info("No high-risk employees found with current threshold.")

def show_individual_prediction(model_data, encoder):
    """Show individual employee prediction interface."""
    st.header("📋 Individual Employee Risk Prediction")
    
    if model_data is None or encoder is None:
        st.error("⚠️ Pre-trained model not found. Please run the training pipeline first.")
        return
    
    st.write("Enter employee information to predict attrition risk:")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 65, 35)
        distance = st.slider("Distance from Home (miles)", 1, 50, 10)
        monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000)
        years_company = st.slider("Years at Company", 0, 40, 5)
        num_companies = st.slider("Number of Companies Worked", 0, 10, 2)
    
    with col2:
        department = st.selectbox("Department", 
                                ['Sales', 'Research & Development', 'Human Resources', 'Finance', 'Marketing'])
        education = st.selectbox("Education Field", 
                               ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'])
        job_role = st.selectbox("Job Role", 
                              ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 
                               'Manufacturing Director', 'Healthcare Representative', 'Manager', 
                               'Sales Representative', 'Research Director', 'Human Resources'])
        job_satisfaction = st.selectbox("Job Satisfaction", ['Low', 'Medium', 'High'])
        overtime = st.selectbox("Overtime", ['Yes', 'No'])
    
    if st.button("Predict Risk", type="primary"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'EmployeeID': [1],
            'Age': [age],
            'Department': [department],
            'DistanceFromHome': [distance],
            'EducationField': [education],
            'JobRole': [job_role],
            'JobSatisfaction': [job_satisfaction],
            'MonthlyIncome': [monthly_income],
            'NumCompaniesWorked': [num_companies],
            'YearsAtCompany': [years_company],
            'OverTime': [overtime]
        })
        
        try:
            # Encode features
            X_encoded = encoder.transform(input_data)
            
            # Make prediction
            prediction = model_data['model'].predict(X_encoded)[0]
            if hasattr(model_data['model'], 'predict_proba'):
                probability = model_data['model'].predict_proba(X_encoded)[0, 1]
            else:
                probability = float(prediction)
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pred_text = "Will Leave" if prediction == 1 else "Will Stay"
                st.metric("Prediction", pred_text)
            
            with col2:
                st.metric("Risk Score", f"{probability:.3f}")
            
            with col3:
                if probability >= 0.7:
                    risk_level = "🔴 High Risk"
                elif probability >= 0.4:
                    risk_level = "🟡 Medium Risk"
                else:
                    risk_level = "🟢 Low Risk"
                st.metric("Risk Level", risk_level)
            
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Attrition Risk Score"},
                delta = {'reference': 0.5},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.4], 'color': "lightgreen"},
                        {'range': [0.4, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.7
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("Recommendations")
            if probability >= 0.7:
                st.error("🚨 **Immediate Action Required**")
                st.write("- Schedule retention meeting within 1 week")
                st.write("- Review compensation and benefits")
                st.write("- Discuss career development opportunities")
                st.write("- Consider flexible work arrangements")
            elif probability >= 0.4:
                st.warning("⚠️ **Monitor Closely**")
                st.write("- Regular check-ins with manager")
                st.write("- Provide additional training opportunities")
                st.write("- Ensure workload balance")
            else:
                st.success("✅ **Low Risk - Continue Current Approach**")
                st.write("- Maintain current engagement level")
                st.write("- Continue regular performance reviews")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()