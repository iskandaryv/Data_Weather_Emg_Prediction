"""
FastAPI backend for Weather Emergency Prediction System.
Provides 5 main endpoints for the application.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, date
import pandas as pd
import numpy as np
from pathlib import Path
import io

from ..data import WeatherDataLoader, EmergencyDataLoader, DataPreprocessor
from ..models import EmergencyPredictor, MultiEmergencyPredictor
from ..utils.config import CORS_ORIGINS, EMERGENCY_TYPES, MODEL_DIR, RAW_DATA_DIR
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Weather Emergency Prediction API",
    description="API for predicting weather-related emergencies in Rostov region",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models and data
weather_loader = WeatherDataLoader()
emergency_loader = EmergencyDataLoader()
preprocessor = DataPreprocessor()
predictor: Optional[EmergencyPredictor] = None
weather_data: Optional[pd.DataFrame] = None
emergency_data: Optional[pd.DataFrame] = None


# Pydantic models
class PredictionRequest(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    temperature: float = Field(..., description="Temperature in Celsius")
    precipitation: float = Field(..., description="Precipitation in mm")
    humidity: float = Field(..., description="Humidity percentage")
    wind_speed: float = Field(..., description="Wind speed in m/s")
    pressure: float = Field(..., description="Atmospheric pressure in hPa")

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2024-07-15",
                "temperature": 35.5,
                "precipitation": 0.0,
                "humidity": 25.0,
                "wind_speed": 5.2,
                "pressure": 1013.0
            }
        }


class PredictionResponse(BaseModel):
    date: str
    emergency_probability: float
    will_occur: bool
    emergency_type: Optional[str] = None
    confidence: float


class DateRangeRequest(BaseModel):
    start_date: str
    end_date: str


class StatsResponse(BaseModel):
    total_days: int
    total_emergencies: int
    emergency_rate: float
    emergency_types: Dict[str, int]
    avg_temperature: float
    avg_precipitation: float


class EmergencyEvent(BaseModel):
    date: str
    type: str
    severity: float
    latitude: float
    longitude: float


# Endpoint 1: Predict Emergency
@app.post("/api/predict", response_model=PredictionResponse)
async def predict_emergency(request: PredictionRequest):
    """
    Predict if an emergency will occur based on weather conditions.
    """
    try:
        global predictor, preprocessor

        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")

        # Create DataFrame from request
        input_data = pd.DataFrame([{
            'date': pd.to_datetime(request.date),
            'temperature': request.temperature,
            'precipitation': request.precipitation,
            'humidity': request.humidity,
            'wind_speed': request.wind_speed,
            'pressure': request.pressure,
            'latitude': 47.2357,
            'longitude': 39.7015
        }])

        # Feature engineering
        input_data = preprocessor.create_time_features(input_data)

        # Get feature columns that model expects
        feature_cols = predictor.feature_names
        available_cols = [col for col in feature_cols if col in input_data.columns]

        # Fill missing features with 0
        for col in feature_cols:
            if col not in input_data.columns:
                input_data[col] = 0.0

        X = input_data[feature_cols]

        # Make prediction
        proba = predictor.predict_proba(X)[0]
        prediction = predictor.predict(X)[0]

        response = PredictionResponse(
            date=request.date,
            emergency_probability=float(proba[1]),
            will_occur=bool(prediction),
            confidence=float(max(proba)),
            emergency_type="potential" if prediction else None
        )

        logger.info(f"Prediction made for {request.date}: {prediction}")
        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 2: Get Historical Data
@app.get("/api/data/historical")
async def get_historical_data(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(1000, description="Maximum number of records")
):
    """
    Get historical weather and emergency data for Rostov region.
    """
    try:
        global weather_data, emergency_data

        if weather_data is None:
            weather_data = weather_loader.load_weather_data()

        df = weather_data.copy()

        # Filter by date range
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]

        # Merge with emergency data if available
        if emergency_data is not None and len(emergency_data) > 0:
            df = preprocessor.merge_weather_emergency(df, emergency_data)

        # Limit results
        df = df.head(limit)

        # Convert to JSON-serializable format
        result = df.to_dict(orient='records')
        for record in result:
            if 'date' in record and isinstance(record['date'], pd.Timestamp):
                record['date'] = record['date'].strftime('%Y-%m-%d')

        logger.info(f"Returned {len(result)} historical records")
        return {"data": result, "count": len(result)}

    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 3: Get Statistics
@app.get("/api/stats", response_model=StatsResponse)
async def get_statistics(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """
    Get statistical summary of weather and emergency data.
    """
    try:
        global weather_data, emergency_data

        if weather_data is None:
            weather_data = weather_loader.load_weather_data()

        df = weather_data.copy()

        # Filter by date range
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]

        # Calculate statistics
        total_days = len(df)
        avg_temp = float(df['temperature'].mean())
        avg_precip = float(df['precipitation'].mean())

        # Emergency statistics
        if emergency_data is not None and len(emergency_data) > 0:
            emg_df = emergency_data.copy()
            if start_date:
                emg_df = emg_df[emg_df['date'] >= pd.to_datetime(start_date)]
            if end_date:
                emg_df = emg_df[emg_df['date'] <= pd.to_datetime(end_date)]

            total_emergencies = len(emg_df)
            emergency_types_count = emg_df['type'].value_counts().to_dict()
        else:
            total_emergencies = 0
            emergency_types_count = {}

        emergency_rate = total_emergencies / total_days if total_days > 0 else 0.0

        response = StatsResponse(
            total_days=total_days,
            total_emergencies=total_emergencies,
            emergency_rate=emergency_rate,
            emergency_types=emergency_types_count,
            avg_temperature=avg_temp,
            avg_precipitation=avg_precip
        )

        logger.info("Statistics calculated successfully")
        return response

    except Exception as e:
        logger.error(f"Error calculating statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 4: Upload Data
@app.post("/api/data/upload")
async def upload_data(
    file: UploadFile = File(...),
    data_type: str = Query(..., description="Type of data: 'weather' or 'emergency'")
):
    """
    Upload weather or emergency data from Excel/CSV file.
    """
    try:
        global weather_data, emergency_data

        # Read file content
        contents = await file.read()

        # Try to read as CSV or Excel
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="File must be CSV or Excel format")

        # Validate and store data
        if data_type == 'weather':
            required_cols = ['date', 'temperature', 'precipitation', 'humidity', 'wind_speed', 'pressure']
            if not all(col in df.columns for col in required_cols):
                raise HTTPException(
                    status_code=400,
                    detail=f"Weather data must contain columns: {required_cols}"
                )

            df['date'] = pd.to_datetime(df['date'])
            weather_data = df

            # Save to disk
            output_path = RAW_DATA_DIR / 'weather_rostov.csv'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)

            logger.info(f"Uploaded {len(df)} weather records")
            return {"message": f"Successfully uploaded {len(df)} weather records", "type": "weather"}

        elif data_type == 'emergency':
            required_cols = ['date', 'type', 'severity']
            if not all(col in df.columns for col in required_cols):
                raise HTTPException(
                    status_code=400,
                    detail=f"Emergency data must contain columns: {required_cols}"
                )

            df['date'] = pd.to_datetime(df['date'])
            emergency_data = df

            # Save to disk
            output_path = RAW_DATA_DIR / 'emergencies_rostov.csv'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)

            logger.info(f"Uploaded {len(df)} emergency records")
            return {"message": f"Successfully uploaded {len(df)} emergency records", "type": "emergency"}

        else:
            raise HTTPException(status_code=400, detail="data_type must be 'weather' or 'emergency'")

    except Exception as e:
        logger.error(f"Error uploading data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 5: Train Model
@app.post("/api/model/train")
async def train_model(
    model_type: str = Query("random_forest", description="Model type: random_forest, gradient_boosting, logistic")
):
    """
    Train emergency prediction model on available data.
    """
    try:
        global predictor, weather_data, emergency_data, preprocessor

        # Load data
        if weather_data is None:
            weather_data = weather_loader.load_weather_data()

        if emergency_data is None or len(emergency_data) == 0:
            # Generate synthetic emergency data
            emergency_data = emergency_loader.generate_synthetic_emergencies(weather_data)

        # Preprocess data
        logger.info("Preprocessing data for training...")

        # Remove outliers
        weather_clean = preprocessor.remove_outliers(weather_data, ['temperature', 'precipitation'])

        # Create time features
        weather_clean = preprocessor.create_time_features(weather_clean)

        # Create rolling features
        weather_clean = preprocessor.create_rolling_features(weather_clean, windows=[7, 14])

        # Create lag features
        weather_clean = preprocessor.create_lag_features(weather_clean, lags=[1, 3, 7])

        # Merge with emergency data
        merged_df = preprocessor.merge_weather_emergency(weather_clean, emergency_data, window_days=3)

        # Prepare training data
        X, y = preprocessor.prepare_training_data(merged_df, target_col='has_emergency')

        # Scale features
        X_scaled = preprocessor.scale_features(X, fit=True)

        # Train model
        logger.info(f"Training {model_type} model...")
        predictor = EmergencyPredictor(model_type=model_type)
        metrics = predictor.train(X_scaled, y, validation=True)

        # Save model
        model_filename = f"emergency_model_{model_type}.pkl"
        predictor.save(model_filename)

        logger.info("Model trained successfully")
        return {
            "message": "Model trained successfully",
            "model_type": model_type,
            "metrics": metrics,
            "training_samples": len(X),
            "features": len(X.columns)
        }

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check API health status."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "weather_data_loaded": weather_data is not None,
        "emergency_data_loaded": emergency_data is not None
    }


@app.on_event("startup")
async def startup_event():
    """Initialize API on startup."""
    global weather_data, emergency_data, predictor

    logger.info("Starting Weather Emergency Prediction API...")

    # Try to load existing data
    try:
        weather_data = weather_loader.load_weather_data()
        logger.info("Weather data loaded")
    except Exception as e:
        logger.warning(f"Could not load weather data: {e}")

    try:
        emergency_data = emergency_loader.load_emergency_data()
        logger.info("Emergency data loaded")
    except Exception as e:
        logger.warning(f"Could not load emergency data: {e}")

    # Try to load existing model
    try:
        model_files = list(MODEL_DIR.glob("emergency_model_*.pkl"))
        if model_files:
            latest_model = sorted(model_files)[-1]
            predictor = EmergencyPredictor.load(latest_model.name)
            logger.info(f"Model loaded: {latest_model.name}")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")

    logger.info("API startup complete")


if __name__ == "__main__":
    import uvicorn
    from ..utils.config import API_HOST, API_PORT

    uvicorn.run(app, host=API_HOST, port=API_PORT)
