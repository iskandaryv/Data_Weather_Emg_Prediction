# 🌦️ Weather Emergency Prediction System

**Production-ready weather emergency prediction system with GeoPandas integration, Gradio dashboard, and FastAPI backend.**

## ✨ Features

- 📊 **Interactive Gradio Dashboard** - Real-time visualization and predictions
- 🗺️ **GeoPandas Integration** - Advanced geospatial analysis
- 🤖 **Machine Learning Models** - Multiple models with ensemble predictions
- 🔌 **FastAPI Backend** - RESTful API with 5 endpoints
- 📈 **Advanced Analytics** - Clustering, correlation analysis, outlier detection
- 🌡️ **Climate Norms** - 30-year historical averages
- 📓 **Jupyter Notebooks** - Complete workflow from data to deployment

## 📚 Notebooks

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

### Block A: Complete Data Preprocessing
- Universal file parser (CSV, Excel, TXT, XML, JSON, PDF)
- GeoPandas integration
- Competition-ready preprocessing

**File**: `notebooks/Block_A_Complete_Preprocessing.ipynb`

### Enhanced Features
- 3 clustering models (KMeans, DBSCAN, Hierarchical)
- Correlation analysis (weather ↔ emergencies)
- Climate norms and anomaly detection
- Imbalanced data handling (SMOTE)

**File**: `notebooks/Enhanced_Features_Clustering_Climate.ipynb`

### Outlier Detection
- 5 detection methods (Z-Score, IQR, Isolation Forest, Elliptic Envelope, MAD)
- Ensemble detection with voting
- Multiple handling strategies

**File**: `notebooks/Outlier_Detection_Demo.ipynb`

## 🚀 Quick Start

### Option 1: Run Dashboard (Recommended)

```bash
# Clone repository
git clone https://github.com/iskandaryv/Data_Weather_Emg_Prediction.git
cd Data_Weather_Emg_Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Gradio dashboard
python rostov_dashboard.py
```

Dashboard will open at: **http://localhost:7860**

### Option 2: Run Full Application

```bash
# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Run complete application (API + Dashboard)
python main.py all
```

Access:
- **Dashboard**: http://localhost:7860
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Option 3: Google Colab (No Installation)

1. Upload notebooks to Google Colab
2. Run cells sequentially
3. All dependencies install automatically

## 📁 Project Structure

```
Data_Weather_Emg_Prediction/
├── notebooks/                        # Complete workflow notebooks
│   ├── Part_A_Data_Processing.ipynb
│   ├── Part_B_Model_Training.ipynb
│   ├── Part_C_API_Development.ipynb
│   ├── Part_D_Web_Interface.ipynb
│   ├── Block_A_Complete_Preprocessing.ipynb
│   ├── Enhanced_Features_Clustering_Climate.ipynb
│   └── Outlier_Detection_Demo.ipynb
├── src/
│   ├── api/                         # FastAPI backend
│   ├── data/                        # Data loading and processing
│   ├── models/                      # ML models
│   ├── utils/                       # Utilities and configuration
│   │   ├── outlier_detection.py    # Outlier detection module
│   │   ├── geo_utils.py            # GeoPandas utilities
│   │   ├── rostov_data.py          # Region data
│   │   └── config.py               # Configuration
│   └── web/                         # Gradio web interface
├── data/
│   ├── raw/                         # Raw data files (Excel/CSV)
│   ├── processed/                   # Processed data
│   └── external/                    # External data sources
├── models/                          # Trained models (.pkl)
├── tests/                           # Test suite
├── rostov_dashboard.py              # Standalone Gradio dashboard
├── generate_rostov_excel.py         # Sample data generator
├── main.py                          # Main entry point
├── requirements.txt                 # Python dependencies
├── setup.sh                         # Setup script
└── README.md                        # This file
```

## 🎯 Dashboard Features

The Gradio dashboard includes 6 interactive tabs:

1. **📍 Interactive Map** - Geodata visualization with districts
2. **🔥 Heat Map** - Temperature/precipitation heatmaps
3. **📊 Statistics** - Data summaries and insights
4. **🏘️ District Comparison** - Compare metrics across districts
5. **🚨 Emergency Prediction** - Real-time predictions
6. **📈 Time Series Analysis** - Trend analysis and forecasting

## 🔌 API Endpoints

1. **POST /api/predict** - Predict emergency occurrence
   ```json
   {
     "temperature": 25.5,
     "precipitation": 10.2,
     "humidity": 65.0,
     "wind_speed": 12.3,
     "pressure": 1013.2
   }
   ```

2. **GET /api/data/historical** - Get historical data
   ```
   ?start_date=2020-01-01&end_date=2020-12-31
   ```

3. **GET /api/stats** - Get statistics
   ```
   ?metric=temperature&aggregation=monthly
   ```

4. **POST /api/data/upload** - Upload data files (Excel/CSV)
   ```bash
   curl -X POST -F "file=@weather_data.xlsx" http://localhost:8000/api/data/upload
   ```

5. **POST /api/model/train** - Train new model
   ```json
   {
     "model_type": "random_forest",
     "parameters": {"n_estimators": 100}
   }
   ```

**Interactive API Documentation**: http://localhost:8000/docs

## 📊 Data Format

### Weather Data (CSV/Excel with GeoPandas)
```csv
date,latitude,longitude,district,temperature,precipitation,humidity,wind_speed,pressure
2015-01-01,47.2357,39.7015,District_1,5.2,2.3,65.0,4.5,1013.2
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

# Run dashboard
python rostov_dashboard.py

# Generate sample data
python generate_rostov_excel.py

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

## 🌍 Geographic Configuration

Configure location via environment variables:

```bash
export GEO_LAT=47.2357
export GEO_LON=39.7015
export GEO_NAME="Region"
export GEO_COUNTRY="Russia"
```

Or use `.env` file:
```
GEO_LAT=47.2357
GEO_LON=39.7015
GEO_NAME=Region
GEO_COUNTRY=Russia
```

## 📦 Dependencies

Main dependencies:
- **Python 3.10+**
- **pandas, numpy** - Data processing
- **scikit-learn** - Machine learning
- **geopandas** - Geospatial analysis
- **fastapi, uvicorn** - API backend
- **gradio** - Interactive dashboard
- **plotly, folium** - Visualizations
- **imbalanced-learn** - Imbalanced data handling

See `requirements.txt` for complete list.

## 🎨 Advanced Features

### Outlier Detection (выбросы)
5 detection methods with ensemble voting:
- Z-Score (statistical)
- IQR/Tukey's Fences (robust)
- Isolation Forest (ML multivariate)
- Elliptic Envelope (Gaussian)
- MAD - Median Absolute Deviation

### Clustering Analysis
3 clustering models for pattern detection:
- KMeans (district grouping)
- DBSCAN (anomaly detection)
- Hierarchical (regional hierarchy)

### Climate Norms
- 30-year historical averages
- Anomaly detection
- Regional aggregation
- Trend analysis

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
- GeoPandas contributors

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Check API documentation at http://localhost:8000/docs

---

**Note**: This system is designed for decision support. Always combine predictions with expert judgment for critical decisions.
