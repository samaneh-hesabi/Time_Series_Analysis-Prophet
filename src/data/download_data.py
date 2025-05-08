#!/usr/bin/env python3
"""
Download and prepare the Airline Passengers dataset.
This dataset contains monthly airline passenger counts from 1949 to 1960.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from io import StringIO

def create_directory(directory_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def download_airline_passengers():
    """Download the Airline Passengers dataset."""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    
    print(f"Downloading data from {url}...")
    response = requests.get(url)
    response.raise_for_status()  # Raise exception if download fails
    
    # Create StringIO object from response content
    data = StringIO(response.text)
    
    # Parse CSV data
    df = pd.read_csv(data, parse_dates=['Month'])
    
    return df

def process_data(df):
    """Process the dataset to prepare it for Prophet."""
    # Rename columns to match Prophet requirements
    prophet_df = df.rename(columns={'Month': 'ds', 'Passengers': 'y'})
    
    # Ensure datetime format
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    return prophet_df

def main():
    """Main function to download and process data."""
    # Create necessary directories
    raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')
    processed_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed')
    
    create_directory(raw_data_dir)
    create_directory(processed_data_dir)
    
    # Download data
    df = download_airline_passengers()
    
    # Save raw data
    raw_data_path = os.path.join(raw_data_dir, 'airline-passengers.csv')
    df.to_csv(raw_data_path, index=False)
    print(f"Raw data saved to {raw_data_path}")
    
    # Process data for Prophet
    prophet_df = process_data(df)
    
    # Save processed data
    processed_data_path = os.path.join(processed_data_dir, 'airline-passengers-prophet.csv')
    prophet_df.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")

if __name__ == "__main__":
    main() 