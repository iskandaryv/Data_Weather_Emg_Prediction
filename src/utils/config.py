"""
Configuration management for the Weather Emergency Prediction system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model paths
MODEL_DIR = PROJECT_ROOT / "models"

# Logs
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# API Configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")  # localhost only
API_PORT = int(os.getenv("API_PORT", 8000))
API_WORKERS = int(os.getenv("API_WORKERS", 1))

# Google Maps API
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# CORS for localhost
CORS_ORIGINS = ["http://localhost:3000", "http://localhost:7860", "http://127.0.0.1:8000"]

# Default Geographic Coordinates (configurable via environment variables)
DEFAULT_GEO_LOCATION = {
    "lat": float(os.getenv("GEO_LAT", "47.2357")),
    "lon": float(os.getenv("GEO_LON", "39.7015")),
    "name": os.getenv("GEO_NAME", "Region"),
    "country": os.getenv("GEO_COUNTRY", "Russia")
}

# Time Configuration
YEARS_OF_DATA = 30
START_YEAR = 2015

# Model Configuration
MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "validation_size": 0.1,
    "cv_folds": 5
}

# Emergency Types (based on climatic criteria)
EMERGENCY_TYPES = [
    "drought",
    "flood",
    "heatwave",
    "frost"
]

# Weather Features
WEATHER_FEATURES = [
    "temperature",
    "precipitation",
    "humidity",
    "wind_speed",
    "pressure"
]

# Geopandas Configuration
GEO_CONFIG = {
    "crs": "EPSG:4326",  # WGS84 coordinate system
    "buffer_km": 50,  # Buffer around point in kilometers
    "geometry_type": "Point"
}
