<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Time Series Analysis with Prophet</div>

# 1. Overview
This project uses Facebook's Prophet library to perform time series analysis and forecasting on stock price data. The implementation is designed to be simple and beginner-friendly while maintaining best practices.

# 2. Features
- Downloads historical stock price data using yfinance
- Creates future price predictions using Prophet
- Generates visualizations of forecasts and trends
- Handles data validation and error logging
- Saves results as high-quality PNG images

# 3. Project Structure
```
Time_Series_Analysis-Prophet/
├── images/              # Directory for generated visualizations
├── simple_prophet.py    # Main script for forecasting
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # Project documentation
```

# 4. Installation
```bash
# Install required packages
pip install -r requirements.txt
```

# 5. Usage
Run the script with default settings (Apple stock data):
```bash
python simple_prophet.py
```

The script will:
1. Download Apple (AAPL) stock data from 2020 onwards
2. Create a forecast for the next 30 days
3. Generate two visualization files in the `images` directory:
   - `apple_stock_forecast.png`: Shows actual prices and predictions
   - `apple_stock_components.png`: Shows trend components

# 6. Dependencies
- pandas
- matplotlib
- prophet
- yfinance
- numpy

# 7. Output
The script generates two types of visualizations:
1. Forecast Plot
   - Blue line: Historical stock prices
   - Red line: Predicted prices
   - Red shaded area: Prediction uncertainty range

2. Components Plot
   - Trend: Overall price movement
   - Yearly: Yearly seasonal patterns
   - Weekly: Weekly seasonal patterns

## 8. Customization

You can easily modify the scripts to:

- Forecast different stocks by changing the ticker symbol
- Adjust forecast periods (default is 90 days)
- Modify model parameters like seasonality and changepoint flexibility
- Create your own custom scenarios

## 9. Resources and Documentation

- **Prophet Documentation**:
  - [Quick Start Guide](https://facebook.github.io/prophet/docs/quick_start.html)
  - [Installation Instructions](https://facebook.github.io/prophet/docs/installation.html)
  - [Diagnostics and Validation](https://facebook.github.io/prophet/docs/diagnostics.html)

- **Prophet Research Paper**:
  - [Prophet: Forecasting at Scale](https://peerj.com/preprints/3190/)

## 10. License

This project is licensed under the MIT License. 