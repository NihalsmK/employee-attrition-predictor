"""
Sample HR Data Generator for Employee Attrition Predictor

This module generates realistic HR data for development and testing purposes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

def generate_sample_hr_data(n_employees: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic HR data for testing and development.
    
    Args:
        n_employees: Number of employee records to generate
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with HR data including all required columns
    """
    np.random.seed(random_state)
    
    # Define realistic value ranges and distributions
    departments = ['Sales', 'Research & Development', 'Human Resources', 'Finance', 'Marketing']
    education_fields = ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other']
    job_roles = ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
                'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director',
                'Human Resources']
    job_satisfaction_levels = ['Low', 'Medium', 'High']
    
    # Generate employee data
    data = {
        'EmployeeID': range(1, n_employees + 1),
        'Age': np.random.normal(37, 10, n_employees).astype(int).clip(18, 65),
        'Department': np.random.choice(departments, n_employees),
        'DistanceFromHome': np.random.exponential(10, n_employees).astype(int).clip(1, 50),
        'EducationField': np.random.choice(education_fields, n_employees),
        'JobRole': np.random.choice(job_roles, n_employees),
        'JobSatisfaction': np.random.choice(job_satisfaction_levels, n_employees, 
                                          p=[0.2, 0.5, 0.3]),  # More medium satisfaction
        'MonthlyIncome': np.random.lognormal(10.5, 0.5, n_employees).astype(int),
        'NumCompaniesWorked': np.random.poisson(2.5, n_employees).clip(0, 10),
        'YearsAtCompany': np.random.exponential(5, n_employees).astype(int).clip(0, 40),
        'OverTime': np.random.choice(['Yes', 'No'], n_employees, p=[0.3, 0.7])
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate attrition based on realistic factors
    attrition_probability = calculate_attrition_probability(df)
    df['Attrition'] = np.random.binomial(1, attrition_probability, n_employees)
    df['Attrition'] = df['Attrition'].map({0: 'No', 1: 'Yes'})
    
    return df

def calculate_attrition_probability(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate realistic attrition probabilities based on employee characteristics.
    
    Args:
        df: DataFrame with employee data
    
    Returns:
        Array of attrition probabilities
    """
    base_prob = 0.16  # Base attrition rate of 16%
    
    # Factors that increase attrition risk
    prob = np.full(len(df), base_prob)
    
    # Distance from home effect
    prob += (df['DistanceFromHome'] > 20) * 0.1
    
    # Overtime effect
    prob += (df['OverTime'] == 'Yes') * 0.15
    
    # Job satisfaction effect
    satisfaction_effect = df['JobSatisfaction'].map({
        'Low': 0.2, 'Medium': 0.0, 'High': -0.1
    })
    prob += satisfaction_effect
    
    # Income effect (lower income = higher attrition)
    income_percentile = df['MonthlyIncome'].rank(pct=True)
    prob += (income_percentile < 0.25) * 0.1  # Bottom quartile
    
    # Years at company effect (new employees more likely to leave)
    prob += (df['YearsAtCompany'] < 2) * 0.15
    
    # Age effect (younger employees more likely to leave)
    prob += (df['Age'] < 30) * 0.1
    
    # Number of companies worked (job hoppers)
    prob += (df['NumCompaniesWorked'] > 4) * 0.1
    
    # Ensure probabilities are between 0 and 1
    return np.clip(prob, 0.05, 0.8)

if __name__ == "__main__":
    # Generate sample data
    sample_data = generate_sample_hr_data(1470)  # IBM HR dataset size
    
    # Save to CSV
    sample_data.to_csv('data/hr_employee_data.csv', index=False)
    
    print(f"Generated {len(sample_data)} employee records")
    print(f"Attrition rate: {(sample_data['Attrition'] == 'Yes').mean():.2%}")
    print("\nDataset shape:", sample_data.shape)
    print("\nColumn info:")
    print(sample_data.info())