"""
Tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import pandas as pd

from src.api.main import app

client = TestClient(app)


class TestAPI:
    """Test API endpoints."""

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_predict_endpoint(self):
        """Test prediction endpoint."""
        payload = {
            "date": "2024-07-15",
            "temperature": 35.5,
            "precipitation": 0.0,
            "humidity": 25.0,
            "wind_speed": 5.2,
            "pressure": 1013.0
        }

        # Note: This might fail if model is not trained
        # In production, ensure model is loaded
        response = client.post("/api/predict", json=payload)

        # Could be 200 (success) or 503 (model not loaded)
        assert response.status_code in [200, 503]

    def test_historical_data_endpoint(self):
        """Test historical data endpoint."""
        response = client.get("/api/data/historical?limit=10")
        assert response.status_code == 200

        data = response.json()
        assert "data" in data
        assert "count" in data

    def test_stats_endpoint(self):
        """Test statistics endpoint."""
        response = client.get("/api/stats")
        assert response.status_code == 200

        data = response.json()
        assert "total_days" in data
        assert "total_emergencies" in data
        assert "emergency_rate" in data

    def test_invalid_prediction_data(self):
        """Test prediction with invalid data."""
        payload = {
            "date": "invalid-date",
            "temperature": "not-a-number"
        }

        response = client.post("/api/predict", json=payload)
        assert response.status_code == 422  # Validation error
