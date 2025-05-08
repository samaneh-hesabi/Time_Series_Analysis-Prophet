#!/usr/bin/env python3
"""
Visualization utilities for Prophet model results.
This script generates various plots for the Prophet model forecasts.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import pickle
from datetime import datetime

def load_model_and_forecast(model_path, forecast_path):
    """Load the saved model and forecast data."""
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load forecast
    if not os.path.exists(forecast_path):
        raise FileNotFoundError(f"Forecast file not found: {forecast_path}")
    
    forecast = pd.read_csv(forecast_path, parse_dates=['ds'])
    
    return model, forecast

def plot_forecast(forecast, original_data=None, figsize=(12, 8), output_path=None):
    """Plot the forecast results."""
    plt.figure(figsize=figsize)
    
    # Plot forecast
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                     color='lightblue', alpha=0.3, label='Prediction Interval')
    plt.plot(forecast['ds'], forecast['yhat'], color='blue', linestyle='-', label='Forecast')
    
    # Plot original data if provided
    if original_data is not None:
        plt.plot(original_data['ds'], original_data['y'], color='black', marker='o', 
                 linestyle='', markersize=4, label='Actual')
    
    # Formatting
    plt.title('Prophet Forecast: Airline Passengers (1949-1960)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Passengers', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Forecast plot saved to {output_path}")
    
    plt.close()

def plot_components(model, forecast, figsize=(18, 12), output_path=None):
    """Plot the components of the forecast (trend, seasonality)."""
    fig = model.plot_components(forecast, figsize=figsize)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Components plot saved to {output_path}")
    
    plt.close(fig)

def plot_forecast_vs_actual(evaluation_df, figsize=(12, 8), output_path=None):
    """Plot forecast vs actual values with residuals."""
    # Create a figure with two subplots (main plot and residuals)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
    
    # Main plot - Forecast vs Actual
    ax1.fill_between(evaluation_df['ds'], evaluation_df['yhat_lower'], evaluation_df['yhat_upper'], 
                     color='lightblue', alpha=0.3, label='Prediction Interval')
    ax1.plot(evaluation_df['ds'], evaluation_df['yhat'], 'b-', label='Forecast')
    ax1.plot(evaluation_df['ds'], evaluation_df['y'], 'ko', markersize=4, label='Actual')
    
    ax1.set_title('Forecast vs Actual: Airline Passengers', fontsize=16)
    ax1.set_ylabel('Passengers', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = evaluation_df['y'] - evaluation_df['yhat']
    ax2.plot(evaluation_df['ds'], residuals, 'r-', label='Residuals')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.fill_between(evaluation_df['ds'], 0, residuals, alpha=0.3, color='red' if any(residuals < 0) else 'green')
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Forecast vs Actual plot saved to {output_path}")
    
    plt.close()

def create_monthly_forecast_table(forecast, start_date=None, end_date=None, output_path=None):
    """Create a formatted table of monthly forecasts."""
    # Filter forecast by date range if specified
    filtered_forecast = forecast.copy()
    if start_date:
        filtered_forecast = filtered_forecast[filtered_forecast['ds'] >= start_date]
    if end_date:
        filtered_forecast = filtered_forecast[filtered_forecast['ds'] <= end_date]
    
    # Select relevant columns and round values
    table_df = filtered_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    table_df = table_df.rename(columns={
        'ds': 'Date',
        'yhat': 'Forecast',
        'yhat_lower': 'Lower Bound (95%)',
        'yhat_upper': 'Upper Bound (95%)'
    })
    
    # Format numeric columns
    for col in ['Forecast', 'Lower Bound (95%)', 'Upper Bound (95%)']:
        table_df[col] = table_df[col].round(0).astype(int)
    
    # Format date column
    table_df['Date'] = table_df['Date'].dt.strftime('%b %Y')
    
    if output_path:
        table_df.to_csv(output_path, index=False)
        print(f"Forecast table saved to {output_path}")
    
    return table_df

def main():
    """Main function to generate all visualization outputs."""
    # Define file paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(project_root, 'models', 'prophet_model.pkl')
    forecast_path = os.path.join(project_root, 'models', 'forecast.csv')
    data_path = os.path.join(project_root, 'data', 'processed', 'airline-passengers-prophet.csv')
    
    # Create output directory
    output_dir = os.path.join(project_root, 'results', 'visualizations')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Load model and forecast
    model, forecast = load_model_and_forecast(model_path, forecast_path)
    
    # Load original data
    original_data = pd.read_csv(data_path, parse_dates=['ds'])
    
    # Create evaluation dataframe
    evaluation_df = pd.merge(
        original_data,
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        on='ds',
        how='left'
    )
    
    # Generate visualizations
    plot_forecast(
        forecast, 
        original_data=original_data,
        output_path=os.path.join(output_dir, 'forecast_plot.png')
    )
    
    plot_components(
        model, 
        forecast,
        output_path=os.path.join(output_dir, 'components_plot.png')
    )
    
    plot_forecast_vs_actual(
        evaluation_df,
        output_path=os.path.join(output_dir, 'forecast_vs_actual.png')
    )
    
    # Create forecast table for future months
    today = pd.Timestamp.today()
    future_forecast = create_monthly_forecast_table(
        forecast,
        start_date=original_data['ds'].max(),
        output_path=os.path.join(output_dir, 'future_forecast_table.csv')
    )
    
    print("Visualization generation complete.")

if __name__ == "__main__":
    main() 