#!/usr/bin/env python3
"""
Run the entire time series analysis pipeline.
This script executes all components of the time series analysis:
1. Data download and processing
2. Model training
3. Visualization generation
"""

import os
import subprocess
import sys
import time

def run_command(command, description):
    """Run a command with a description and capture output."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        process = subprocess.run(
            command,
            check=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(process.stdout)
        if process.stderr:
            print(f"STDERR: {process.stderr}")
        
        elapsed_time = time.time() - start_time
        print(f"COMPLETED in {elapsed_time:.2f} seconds")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        elapsed_time = time.time() - start_time
        print(f"FAILED in {elapsed_time:.2f} seconds")
        return False

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'results/visualizations'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def main():
    """Run the complete pipeline."""
    print(f"\n{'*'*80}")
    print(f"STARTING TIME SERIES ANALYSIS PIPELINE")
    print(f"{'*'*80}")
    
    # Record start time
    overall_start_time = time.time()
    
    # Create necessary directories
    create_directories()
    
    # Step 1: Download and process data
    if not run_command(
        "python src/data/download_data.py", 
        "Download and process the Airline Passengers dataset"
    ):
        print("Data download failed. Exiting pipeline.")
        return False
    
    # Step 2: Train the Prophet model
    if not run_command(
        "python src/models/prophet_model.py",
        "Train the Prophet model"
    ):
        print("Model training failed. Exiting pipeline.")
        return False
    
    # Step 3: Generate visualizations
    if not run_command(
        "python src/visualization/visualize.py",
        "Generate visualizations"
    ):
        print("Visualization generation failed. Exiting pipeline.")
        return False
    
    # Record end time and print summary
    overall_elapsed_time = time.time() - overall_start_time
    
    print(f"\n{'*'*80}")
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {overall_elapsed_time:.2f} seconds")
    print(f"{'*'*80}")
    
    print("\nResults can be found in:")
    print("- Processed data: data/processed/")
    print("- Model outputs: models/")
    print("- Visualizations: results/visualizations/")
    print("\nTo explore the analysis in detail, open the Jupyter notebook:")
    print("notebooks/time_series_analysis_with_prophet.ipynb")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 