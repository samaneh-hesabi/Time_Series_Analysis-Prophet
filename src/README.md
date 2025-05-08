<div style="font-size:1.8em; font-weight:bold; text-align:center; margin-top:20px;">Source Code Directory</div>

This directory contains the source code modules for the time series analysis project.

## 1. Directory Structure

- `data/`: Scripts for data acquisition and preprocessing
  - `download_data.py`: Downloads and processes the Airline Passengers dataset

- `models/`: Scripts for model creation and evaluation
  - `prophet_model.py`: Implements the Prophet time series model for forecasting

- `visualization/`: Scripts for creating visualizations
  - `visualize.py`: Generates various plots and visualizations of the model results

## 2. Module Descriptions

### 2.1 data/download_data.py

This module handles downloading the Airline Passengers dataset from the internet and preprocessing it for use with Prophet. It performs the following tasks:
- Downloads data from a GitHub repository
- Processes the raw data into the format expected by Prophet (ds/y columns)
- Saves both raw and processed datasets to appropriate directories

### 2.2 models/prophet_model.py

This module implements the Prophet time series forecasting model. It includes:
- Loading the processed dataset
- Configuring and training a Prophet model with appropriate parameters
- Making future predictions
- Evaluating model performance with common metrics (MAE, RMSE, MAPE, R²)
- Saving the trained model, forecasts, and performance metrics

### 2.3 visualization/visualize.py

This module creates various visualizations from the model results:
- Forecast plots with prediction intervals
- Component plots showing trend and seasonality
- Forecast vs actual comparisons with residuals
- Formatted forecast tables for future periods

Each module is designed to be run independently or as part of the pipeline controlled by the main `run_pipeline.py` script in the project root. 