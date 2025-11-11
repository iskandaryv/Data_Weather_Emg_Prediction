# Quick Start Guide - Rostov-on-Don Dashboard

## 🚀 Launch the Dashboard

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Rostov dashboard
python rostov_dashboard.py
```

The dashboard will:
- Generate sample Excel file with geodata (lat/lon columns)
- Load 8 Rostov-on-Don districts
- Load landmarks and places
- Launch interactive dashboard at http://127.0.0.1:7860

## 📊 Dashboard Features

### 1. **Rostov Districts Map** 🗺️
- Interactive map with 8 district boundaries
- District centers marked with red circles
- 8 landmarks with custom icons:
  - Rostov Arena (stadium)
  - Gorky Park
  - Musical Theater
  - Bolshaya Sadovaya Street
  - Rostov Zoo
  - Don River Embankment
  - Regional Museum
  - Central Market

### 2. **Heat Map** 🔥
- Temperature/precipitation/humidity heat maps
- Visualize patterns across districts
- Color-coded intensity

### 3. **District Comparison** 📊
- Compare metrics across 8 districts
- Bar charts with statistics
- Average/min/max values

### 4. **Time Series** 📈
- Multi-line charts by district
- Date range filtering
- Trend analysis

### 5. **District Details** 🏘️
- Detailed statistics per district
- Population data
- Weather metrics

### 6. **Landmarks & Places** 📍
- List of key locations
- District mapping
- Russian and English names

## 📁 Excel File Format

The dashboard uses Excel files with geodata:

```
date       | district  | district_ru      | latitude | longitude | temperature | precipitation | humidity | wind_speed | pressure
-----------|-----------|------------------|----------|-----------|-------------|---------------|----------|------------|----------
2024-01-01 | Leninsky  | Ленинский район  | 47.2220  | 39.7180   | 5.2         | 0.0           | 65.0     | 4.5        | 1013.2
2024-01-01 | Kirovsky  | Кировский район  | 47.2580  | 39.7850   | 5.5         | 0.0           | 63.0     | 5.1        | 1012.8
```

**Key columns:**
- `latitude` - Geographic latitude
- `longitude` - Geographic longitude
- `district` - District name (English)
- `district_ru` - District name (Russian)

## 🏘️ Rostov-on-Don Districts

1. **Leninsky** (Ленинский) - Central, historical center
2. **Kirovsky** (Кировский) - Industrial & residential
3. **Oktyabrsky** (Октябрьский) - Northern residential
4. **Pervomaisky** (Первомайский) - Western with parks
5. **Proletarsky** (Пролетарский) - Southern residential
6. **Sovetsky** (Советский) - Western residential
7. **Zheleznodorozhny** (Железнодорожный) - Railway area
8. **Voroshilovsky** (Ворошиловский) - Central residential

## 📍 Use Your Own Excel File

To use your own data:

```python
# Your Excel file should have these columns:
# - date
# - latitude (required!)
# - longitude (required!)
# - temperature, precipitation, humidity, etc.
# - Optional: district, district_ru

# Load your data:
weather_df = pd.read_excel('your_file.xlsx')
```

## 🗺️ Google Maps Integration

To enable Google Maps features:

1. Get API key from Google Cloud Console
2. Create `.env` file:
```
GOOGLE_MAPS_API_KEY=your_key_here
```

3. Use Google Maps API:
```python
from src.utils.geo_utils import GoogleMapsAPI

gmaps = GoogleMapsAPI()
address = gmaps.reverse_geocode(47.2357, 39.7015)
# Returns: "Rostov-on-Don, Russia"
```

## 🎯 Generated Files

When you run the dashboard:
- `sample_rostov_weather.xlsx` - Sample data with geodata
- Maps are embedded in the web interface
- Can export maps to HTML

## 💡 Tips

- Dashboard runs on **localhost only** (127.0.0.1:7860)
- No Docker needed
- Excel files must have lat/lon columns
- Interactive maps use Folium (HTML/JavaScript)
- All 8 districts have approximate boundaries
- Landmarks include real Rostov places
