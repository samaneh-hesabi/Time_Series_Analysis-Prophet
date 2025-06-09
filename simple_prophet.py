#!/usr/bin/env python3
"""
A Simple Stock Price Forecasting Tool

This script helps you:
1. Download stock price data (using Apple as an example)
2. Create future price predictions
3. Save visualization graphs

We use Facebook's Prophet library which is great for beginners in time series forecasting.
"""

# Import required libraries
import pandas as pd  # For handling data
import matplotlib.pyplot as plt  # For creating graphs
from prophet import Prophet  # Facebook's forecasting tool
import yfinance as yf  # For downloading stock data
from pathlib import Path  # For handling file paths
import logging  # For showing helpful messages
import sys  # For showing detailed error messages
import traceback  # For showing detailed error traces

# Set up logging to show what the program is doing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'  # More detailed format
)
logger = logging.getLogger(__name__)

# Create a folder to save our graphs
IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True)

# How many days into the future we want to predict
FORECAST_DAYS = 30  # We'll predict one month ahead


def get_stock_data(ticker='AAPL', start_date='2020-01-01'):
    """
    Downloads stock price data from Yahoo Finance.
    
    For example:
    - ticker='AAPL' for Apple stock
    - start_date='2020-01-01' means get data from January 1st, 2020
    
    Returns:
    - A table (DataFrame) with dates and stock prices
    """
    logger.info(f"Downloading {ticker} stock data...")
    
    try:
        # Download the data
        stock_data = yf.download(ticker, start=start_date, progress=False)
        
        if stock_data.empty:
            logger.error(f"No data received for {ticker}")
            return None
            
        logger.info(f"Downloaded data shape: {stock_data.shape}")
        
        # Prepare data for Prophet
        stock_df = stock_data.reset_index()
        
        # Convert to proper format for Prophet
        stock_df['ds'] = stock_df['Date']
        stock_df['y'] = stock_df['Close'].values  # Convert to numpy array
        stock_df = stock_df[['ds', 'y']].copy()
        
        # Convert to proper types
        stock_df['ds'] = pd.to_datetime(stock_df['ds'])
        stock_df['y'] = stock_df['y'].astype('float64')
        
        # Verify data
        logger.info(f"Data types - ds: {stock_df['ds'].dtype}, y: {stock_df['y'].dtype}")
        logger.info(f"Successfully downloaded {len(stock_df)} days of stock data")
        logger.info(f"Data range: from {stock_df['ds'].min()} to {stock_df['ds'].max()}")
        logger.info(f"Sample of data:\n{stock_df.head()}")
        
        return stock_df
    
    except Exception as e:
        logger.error(f"Error in get_stock_data: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def make_forecast(data, days_to_predict=30, stock_name="Stock Price", save_as="forecast"):
    """
    Creates a forecast of future stock prices.
    
    Args:
        data: Table with historical stock prices
        days_to_predict: How many days into the future to predict
        stock_name: Name to show on the graphs
        save_as: Base name for saving the graph files
    """
    logger.info(f"Creating forecast for {stock_name}...")
    
    if data is None or data.empty:
        logger.error("No data provided. Can't make forecast.")
        return None
    
    try:
        # Verify data format
        if not isinstance(data, pd.DataFrame):
            logger.error("Input must be a pandas DataFrame")
            return None
            
        if 'ds' not in data.columns or 'y' not in data.columns:
            logger.error("DataFrame must have 'ds' and 'y' columns")
            return None
        
        # Create a copy to avoid modifying original data
        forecast_data = data.copy()
        
        # Ensure data types are correct
        forecast_data['ds'] = pd.to_datetime(forecast_data['ds'])
        forecast_data['y'] = forecast_data['y'].astype('float64')
        
        # Create and train Prophet model
        logger.info("Creating and training Prophet model...")
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95
        )
        
        logger.info("Fitting model...")
        model.fit(forecast_data)
        
        logger.info("Creating future dates for prediction...")
        future_dates = model.make_future_dataframe(periods=days_to_predict)
        
        logger.info("Making predictions...")
        forecast = model.predict(future_dates)
        
        # Create visualization directory if it doesn't exist
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("Creating forecast graph...")
        plt.figure(figsize=(12, 6))
        
        # Plot actual data
        plt.plot(forecast_data['ds'], forecast_data['y'], 
                label='Actual Price', 
                color='blue', 
                linewidth=2)
        
        # Plot predictions
        future_data = forecast[forecast['ds'] > forecast_data['ds'].max()]
        plt.plot(future_data['ds'], 
                future_data['yhat'], 
                label='Predicted Price', 
                color='red', 
                linewidth=2)
        
        plt.fill_between(
            future_data['ds'],
            future_data['yhat_lower'],
            future_data['yhat_upper'],
            color='red',
            alpha=0.2,
            label='Prediction Uncertainty'
        )
        
        plt.title(f"{stock_name} Forecast", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price (USD)", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        logger.info(f"Saving forecast graph to {IMAGE_DIR}/{save_as}_forecast.png")
        plt.savefig(f"{IMAGE_DIR}/{save_as}_forecast.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info("Creating components graph...")
        fig = model.plot_components(forecast)
        plt.tight_layout()
        logger.info(f"Saving components graph to {IMAGE_DIR}/{save_as}_components.png")
        plt.savefig(f"{IMAGE_DIR}/{save_as}_components.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info("Graphs have been saved successfully")
        return forecast
    
    except Exception as e:
        logger.error(f"Error in make_forecast: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def main():
    """
    Main function that runs our stock price forecasting program.
    """
    try:
        logger.info("Starting our stock price forecasting program...")
        
        # Get Apple stock data
        stock_data = get_stock_data(ticker='AAPL', start_date='2020-01-01')
        
        if stock_data is not None:
            logger.info("Making predictions...")
            forecast = make_forecast(
                data=stock_data,
                days_to_predict=FORECAST_DAYS,
                stock_name="Apple Stock Price",
                save_as="apple_stock"
            )
            
            if forecast is not None:
                # Show predictions for the next few days
                future_predictions = forecast[forecast['ds'] > stock_data['ds'].max()]
                print("\nPredicted prices for the next few days:")
                predictions = future_predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                predictions = predictions.rename(columns={
                    'ds': 'Date',
                    'yhat': 'Predicted Price',
                    'yhat_lower': 'Lower Bound',
                    'yhat_upper': 'Upper Bound'
                })
                print(predictions.head().to_string(index=False))
        
        logger.info("Program completed! Check the 'images' folder for graphs.")
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")


# This is where the program starts
if __name__ == "__main__":
    main() 