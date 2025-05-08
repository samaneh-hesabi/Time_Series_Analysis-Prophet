#!/bin/bash
# Installation script for Time Series Analysis with Prophet project

echo "Setting up Time Series Analysis with Prophet project..."

# Check if conda is installed
if command -v conda >/dev/null 2>&1; then
    echo "Setting up conda environment..."
    
    # Create conda environment if it doesn't exist
    if conda info --envs | grep -q "TSAP"; then
        echo "Conda environment 'TSAP' already exists."
    else
        echo "Creating conda environment 'TSAP'..."
        conda env create -f environment.yml
    fi
    
    echo "Activating conda environment 'TSAP'..."
    echo "Please run 'conda activate TSAP' manually after this script completes."
    
else
    echo "Conda not detected. Using pip for installation..."
    
    # Check if virtual environment exists
    if [ -d "venv" ]; then
        echo "Virtual environment already exists."
    else
        echo "Creating virtual environment..."
        python -m venv venv
    fi
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate || source venv/Scripts/activate
    
    # Install requirements
    echo "Installing required packages..."
    pip install -r requirements.txt
    
    # Install the project in development mode
    echo "Installing project in development mode..."
    pip install -e .
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/raw data/processed notebooks src/data src/models src/visualization models results/visualizations

echo "Installation complete! You can now run the pipeline with:"
echo "python run_pipeline.py"
echo ""
echo "Or explore the Jupyter notebook with:"
echo "jupyter notebook notebooks/time_series_analysis_with_prophet.ipynb" 