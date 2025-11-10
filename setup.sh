#!/bin/bash
# Startup script for Weather Emergency Prediction System

echo "Weather Emergency Prediction System - Startup"
echo "============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/raw data/processed data/external models logs

# Generate initial data if needed
echo "Checking for data files..."
if [ ! -f "data/raw/weather_rostov.csv" ]; then
    echo "Generating initial data..."
    python -c "from src.data import WeatherDataLoader, EmergencyDataLoader; w=WeatherDataLoader(); d=w.generate_synthetic_data(); e=EmergencyDataLoader(); e.generate_synthetic_emergencies(d)"
fi

# Train initial model if needed
if [ ! -f "models/emergency_model_random_forest.pkl" ]; then
    echo "Training initial model..."
    python main.py train
fi

echo ""
echo "Setup complete!"
echo ""
echo "To start the system:"
echo "  - API only:    python main.py api"
echo "  - Web only:    python main.py web"
echo "  - Train model: python main.py train"
echo "  - Everything:  python main.py all"
echo ""
echo "Or use Docker:"
echo "  docker-compose up"
