# Time_Series_Analysis-Prophet
Time_Series_Analysis-Prophet

<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Time Series Analysis with Prophet</div>

This repository contains a complete time series analysis project using Facebook Prophet. The project demonstrates forecasting techniques on the classic Airline Passengers dataset with proper data preprocessing, model training, evaluation, and visualization.

## 1. Project Structure

- `data/`: Contains all datasets
  - `raw/`: Original unmodified datasets
  - `processed/`: Cleaned and preprocessed datasets
- `notebooks/`: Jupyter notebooks for exploratory analysis and modeling
- `src/`: Source code modules
  - `data/`: Scripts for data acquisition and preprocessing
  - `models/`: Scripts for model creation and evaluation 
  - `visualization/`: Scripts for creating visualizations
- `models/`: Saved model artifacts and forecasts
- `results/`: Visualizations and evaluation metrics
- `requirements.txt`: Required Python packages
- `environment.yml`: Conda environment configuration
- `run_pipeline.py`: Script to execute the entire analysis pipeline
- `install.sh`: Installation script

## 2. Setup

```bash
# Option 1: Quick setup with install script
chmod +x install.sh
./install.sh

# Option 2: Manual setup with pip
pip install -r requirements.txt

# Option 3: Setup with conda
conda env create -f environment.yml
conda activate TSAP
```

## 3. Usage

### Running the Complete Pipeline

To execute the entire analysis pipeline (data download, model training, and visualization):

```bash
python run_pipeline.py
```

### Exploring the Analysis in Jupyter Notebook

```bash
jupyter notebook notebooks/time_series_analysis_with_prophet.ipynb
```

## 4. Dataset

This project uses the Airline Passengers dataset, which contains monthly totals of international airline passengers from 1949 to 1960. The dataset exhibits both trend and seasonality components, making it ideal for time series forecasting demonstrations.

## 5. Results

The Prophet model achieved excellent performance on this dataset:

- Mean Absolute Error (MAE): 8.00
- Root Mean Squared Error (RMSE): 10.33
- Mean Absolute Percentage Error (MAPE): 3.34%
- R²: 0.9925

Key visualizations from the analysis:

1. **Forecast Plot**: Shows the model's predictions with confidence intervals
2. **Components Plot**: Breaks down the forecast into trend and seasonal components
3. **Forecast vs Actual**: Compares predicted values against actual values with residuals
4. **Future Forecast Table**: Detailed monthly predictions for future periods

All visualizations can be found in the `results/visualizations/` directory after running the pipeline.

## 6. License

This project is licensed under the terms included in the LICENSE file.
