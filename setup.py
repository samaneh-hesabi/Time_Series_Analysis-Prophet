#!/usr/bin/env python3
"""
Setup script for Time Series Analysis with Prophet project.
This script installs the required packages for the project.
"""

from setuptools import setup, find_packages

setup(
    name="time_series_prophet",
    version="0.1.0",
    description="Time Series Analysis with Prophet",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "prophet>=1.1.0",
        "pandas>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",
        "jupyter>=1.0.0",
        "notebook>=6.4.0",
        "ipykernel>=6.0.0",
        "requests>=2.27.0",
    ],
    python_requires=">=3.8",
) 