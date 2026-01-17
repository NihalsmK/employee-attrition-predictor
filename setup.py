from setuptools import setup, find_packages

setup(
    name="employee-attrition-predictor",
    version="1.0.0",
    description="HR Analytics system for predicting employee attrition",
    author="Data Science Intern",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "jupyter>=1.0.0",
        "hypothesis>=6.0.0",
        "fpdf2>=2.7.0",
        "numpy>=1.24.0",
        "plotly>=5.15.0",
        "kaleido>=0.2.1"
    ],
    python_requires=">=3.8",
)