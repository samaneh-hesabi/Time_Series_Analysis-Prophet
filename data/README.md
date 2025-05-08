<div style="font-size:1.8em; font-weight:bold; text-align:center; margin-top:20px;">Data Directory</div>

This directory contains all datasets used in the time series analysis project.

## 1. Directory Structure

- `raw/`: Contains the original unmodified datasets
  - `airline-passengers.csv`: Monthly airline passenger counts from 1949 to 1960

- `processed/`: Contains cleaned and preprocessed datasets ready for model training
  - `airline-passengers-prophet.csv`: Preprocessed data with columns renamed to match Prophet's requirements (ds, y)

## 2. Data Description

### 2.1 Airline Passengers Dataset

The Airline Passengers dataset is a classic time series dataset that records the monthly total number of international airline passengers from 1949 to 1960. This dataset exhibits both trend and seasonal components, making it ideal for time series forecasting demonstrations.

**Features:**
- Monthly data points (144 observations)
- Strong upward trend
- Multiplicative seasonality (seasonal fluctuations increase with the rising trend)
- No missing values

**Source:** Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (1976) Time Series Analysis, Forecasting and Control. Third Edition. Holden-Day. Series G. 