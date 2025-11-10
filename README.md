

For easy access and execution, use these Google Colab notebooks:

### Part A: Data Processing
- Load/generate weather and emergency data
- Data cleaning and outlier removal
- Feature engineering (time features, rolling statistics, lag features)
- Data visualization

**File**: `notebooks/Part_A_Data_Processing.ipynb`

### Part B: Model Training
- Train multiple ML models
- Model evaluation and comparison
- Feature importance analysis
- Model saving and testing

**File**: `notebooks/Part_B_Model_Training.ipynb`

### Part C: API Development
- FastAPI backend with 5 endpoints
- Prediction API
- Data management
- Model training endpoint

**File**: `notebooks/Part_C_API_Development.ipynb`

### Part D: Web Interface
- Gradio interactive dashboard
- Real-time predictions
- Data visualization
- Statistics display

**File**: `notebooks/Part_D_Web_Interface.ipynb`

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Open `notebooks/Part_A_Data_Processing.ipynb` in Google Colab
2. Run all cells to process data
3. Open `notebooks/Part_B_Model_Training.ipynb` to train models
4. Open `notebooks/Part_C_API_Development.ipynb` for API
5. Open `notebooks/Part_D_Web_Interface.ipynb` for web interface

### Option 2: Local Installation

```bash
# Clone repository
git clone <repo-url>
cd Data_Weather_Emg_Prediction

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Run application
python main.py all
```

### Option 3: Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access services:
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Web Interface: http://localhost:7860
```

## 📁 Project Structure

```
Data_Weather_Emg_Prediction/
├── src/
│   ├── api/              # FastAPI backend
│   ├── data/             # Data loading and processing
│   ├── models/           # ML models
│   ├── utils/            # Utilities and configuration
│   └── web/              # Gradio web interface
├── notebooks/            # Google Colab notebooks (Parts A-D)
├── data/
│   ├── raw/             # Raw data files (Excel/CSV)
│   ├── processed/       # Processed data
│   └── external/        # External data sources
├── models/              # Trained models
├── tests/               # Test suite
├── logs/                # Application logs
├── main.py              # Main entry point
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose configuration
└── README.md           # This file
```

## 🔌 API Endpoints

1. **POST /api/predict** - Predict emergency occurrence
2. **GET /api/data/historical** - Get historical data
3. **GET /api/stats** - Get statistics
4. **POST /api/data/upload** - Upload data files (Excel/CSV)
5. **POST /api/model/train** - Train new model

API Documentation: http://localhost:8000/docs

## 📊 Data Format

### Weather Data (CSV/Excel)
```csv
date,latitude,longitude,temperature,precipitation,humidity,wind_speed,pressure
2015-01-01,47.2357,39.7015,5.2,2.3,65.0,4.5,1013.2
```

### Emergency Data (CSV/Excel)
```csv
date,type,severity,latitude,longitude
2015-07-15,heatwave,8.5,47.2357,39.7015
```

## 🎯 Emergency Types

- **Heatwave**: Temperature > 35°C
- **Drought**: Low precipitation + low humidity
- **Flood**: Precipitation > 50mm
- **Frost**: Temperature < -20°C

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py
```

## 🛠️ Commands

```bash
# Train model only
python main.py train

# Run API only
python main.py api

# Run web interface only
python main.py web

# Run everything (train + API + web)
python main.py all
```

## 📈 Model Performance

Models are evaluated on:
- **Accuracy**: Overall correctness
- **Precision**: Emergency prediction accuracy
- **Recall**: Emergency detection rate
- **F1-Score**: Balanced metric
- **ROC-AUC**: Model discrimination ability

## 🌍 Location

- Latitude: 47.2357°N
- Longitude: 39.7015°E
- Region: Southern Federal District

## 📦 Dependencies

Main dependencies:
- Python 3.10+
- pandas, numpy - Data processing
- scikit-learn - Machine learning
- fastapi, uvicorn - API backend
- gradio - Web interface
- plotly - Visualizations

See `requirements.txt` for complete list.

## 🐳 Docker Deployment

```bash
# Build image
docker build -t weather-emergency-prediction .

# Run API
docker run -p 8000:8000 weather-emergency-prediction python main.py api

# Run web interface
docker run -p 7860:7860 weather-emergency-prediction python main.py web
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is for educational purposes.

## 👥 Authors

Iskandar - Weather Emergency Prediction System

## 🙏 Acknowledgments

- Weather data sources
- Machine learning community

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Check documentation at `/docs`

---

**Note**: This system is designed for decision support. Always combine predictions with expert judgment for critical decisions.
