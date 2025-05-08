#!/usr/bin/env python3
"""
Train a Prophet model on the Airline Passengers dataset.
This script loads the processed data, trains a Prophet model,
and saves the trained model and forecasts.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import pickle
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(file_path):
    """Load the processed dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path, parse_dates=['ds'])
    print(f"Loaded data from {file_path} with shape {df.shape}")
    return df

def train_prophet_model(df, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False):
    """Train a Prophet model with specified parameters."""
    print("Training Prophet model...")
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        seasonality_mode='multiplicative',  # Airline data has multiplicative seasonality
        changepoint_prior_scale=0.05,       # Controls flexibility of trend
        seasonality_prior_scale=10.0        # Controls flexibility of seasonality
    )
    
    model.fit(df)
    print("Model training complete")
    return model

def make_future_predictions(model, periods=12, freq='MS'):
    """Generate future predictions."""
    print(f"Generating predictions for {periods} {freq} periods...")
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

def evaluate_model(df, forecast):
    """Evaluate the model performance on the training data."""
    print("Evaluating model performance...")
    
    # Merge actual and predicted values
    evaluation_df = pd.merge(
        df, 
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        on='ds', 
        how='left'
    )
    
    # Calculate metrics
    mae = mean_absolute_error(evaluation_df['y'], evaluation_df['yhat'])
    rmse = np.sqrt(mean_squared_error(evaluation_df['y'], evaluation_df['yhat']))
    r2 = r2_score(evaluation_df['y'], evaluation_df['yhat'])
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((evaluation_df['y'] - evaluation_df['yhat']) / evaluation_df['y'])) * 100
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }
    
    return metrics, evaluation_df

def save_model(model, forecast, metrics, output_dir):
    """Save the model, forecasts, and metrics."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Save model
    model_path = os.path.join(output_dir, 'prophet_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    # Save forecast
    forecast_path = os.path.join(output_dir, 'forecast.csv')
    forecast.to_csv(forecast_path, index=False)
    print(f"Forecast saved to {forecast_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

def main():
    """Main function to train and evaluate the Prophet model."""
    # Define file paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(project_root, 'data', 'processed', 'airline-passengers-prophet.csv')
    output_dir = os.path.join(project_root, 'models')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    df = load_data(data_path)
    
    # Train model
    model = train_prophet_model(df)
    
    # Make future predictions (12 months ahead)
    forecast = make_future_predictions(model, periods=24, freq='MS')
    
    # Evaluate model
    metrics, evaluation_df = evaluate_model(df, forecast)
    
    # Save outputs
    save_model(model, forecast, metrics, output_dir)
    
    print("Prophet model training and evaluation complete.")

if __name__ == "__main__":
    main() 