
# Employee Attrition Prediction - Executive Summary

**Generated on:** 2026-01-17 17:42:30

## Key Findings

### Data Overview
- **Total Employees Analyzed:** 1,470
- **Data Quality:** 0 missing values detected
- **Features Analyzed:** 6 numerical, 6 categorical

### Model Performance
- **Best Performing Model:** Decision Tree
- **Model Accuracy:** 67.0%
- **Key Predictive Features:**
  1. MonthlyIncome (importance: 0.195)
  2. EmployeeID (importance: 0.146)
  3. Age (importance: 0.135)
  4. DistanceFromHome (importance: 0.090)
  5. NumCompaniesWorked (importance: 0.085)

### Risk Assessment
- **Employees at High Risk:** 446 (30.3%)
- **Estimated Turnover Cost:** $403,676,232
- **Potential Savings:** $121,102,870

### Business Insights
- **Distance Attrition Correlation:** Significant positive correlation: employees living farther from work are more likely to leave (r=0.065, p=0.0129)
- **Overtime Attrition Association:** Significant association: employees working overtime have 43.4% attrition rate vs 29.6% for non-overtime (χ²=24.267, p=0.0000)
- **Satisfaction Attrition Relationship:** Significant relationship: Job satisfaction affects attrition (Low: 47.1%, Medium: 32.9%, High: 23.4%, χ²=44.284, p=0.0000)

### Recommendations
- Best overall model: Decision Tree
- Logistic Regression recommended for interpretability with minimal performance loss

### Department Breakdown
- **Research & Development:** 74 high-risk employees
- **Sales:** 116 high-risk employees
- **Marketing:** 78 high-risk employees
- **Finance:** 95 high-risk employees
- **Human Resources:** 83 high-risk employees
