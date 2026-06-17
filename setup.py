"""
TrendSense Package Setup
"""

from setuptools import setup, find_packages

setup(
    name="trendsense",
    version="1.0.0",
    description="AI-driven Decision Support System integrating social media trend velocity with historical sales data",
    author="A S Nemitha, Chaitanya G, Inika Ranganath Prasad",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "statsmodels>=0.14.0",
        "pytrends>=4.9.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "streamlit>=1.28.0",
        "joblib>=1.3.0",
        "tqdm>=4.65.0",
    ],
)
