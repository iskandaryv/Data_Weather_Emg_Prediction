"""
Generate sample Excel file with Russian region weather data and geodata.
Standalone script - no dependencies on other modules.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Russian Region Districts
ROSTOV_DISTRICTS = {
    "Leninsky": {"name": "Leninsky District", "name_ru": "Ленинский район", "lat": 47.2220, "lon": 39.7180},
    "Kirovsky": {"name": "Kirovsky District", "name_ru": "Кировский район", "lat": 47.2580, "lon": 39.7850},
    "Oktyabrsky": {"name": "Oktyabrsky District", "name_ru": "Октябрьский район", "lat": 47.2750, "lon": 39.7320},
    "Pervomaisky": {"name": "Pervomaisky District", "name_ru": "Первомайский район", "lat": 47.2180, "lon": 39.6420},
    "Proletarsky": {"name": "Proletarsky District", "name_ru": "Пролетарский район", "lat": 47.1980, "lon": 39.7680},
    "Sovetsky": {"name": "Sovetsky District", "name_ru": "Советский район", "lat": 47.2420, "lon": 39.6850},
    "Zheleznodorozhny": {"name": "Zheleznodorozhny District", "name_ru": "Железнодорожный район", "lat": 47.2640, "lon": 39.7180},
    "Voroshilovsky": {"name": "Voroshilovsky District", "name_ru": "Ворошиловский район", "lat": 47.2380, "lon": 39.7420}
}

print("=" * 70)
print("GENERATING SAMPLE EXCEL FILE WITH REGION GEODATA")
print("=" * 70)

# Generate weather data
data = []
start_date = datetime(2024, 1, 1)
np.random.seed(42)

print("\nGenerating data for 365 days across 8 districts...")

for i in range(365):  # One year
    current_date = start_date + timedelta(days=i)

    # Generate data for each district
    for district_id, district in ROSTOV_DISTRICTS.items():
        # Seasonal temperature
        temp = 10 + 15 * np.sin(2 * np.pi * i / 365) + np.random.normal(0, 3)

        # Random precipitation
        precip = np.random.gamma(2, 3) if np.random.random() < 0.3 else 0

        # Seasonal humidity
        humidity = np.clip(50 + 20 * np.sin(2 * np.pi * i / 365) + np.random.normal(0, 10), 0, 100)

        # Wind speed
        wind = np.abs(np.random.gamma(3, 2))

        # Pressure
        pressure = 1013 + np.random.normal(0, 8)

        data.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "district": district["name"],
            "district_ru": district["name_ru"],
            "latitude": district["lat"],
            "longitude": district["lon"],
            "temperature": round(temp, 1),
            "precipitation": round(precip, 1),
            "humidity": round(humidity, 1),
            "wind_speed": round(wind, 1),
            "pressure": round(pressure, 1)
        })

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
filename = "sample_weather.xlsx"
df.to_excel(filename, index=False, sheet_name="Weather Data")

print(f"\n✅ SUCCESS! Generated Excel file: {filename}")
print(f"   📊 Total records: {len(df):,}")
print(f"   🏘️  Districts: {len(ROSTOV_DISTRICTS)}")
print(f"   📅 Date range: {df['date'].min()} to {df['date'].max()}")
print(f"   📍 Geodata columns: latitude, longitude")
print(f"\n📋 Columns: {list(df.columns)}")

print("\n" + "=" * 70)
print("FILE PREVIEW")
print("=" * 70)
print(df.head(10).to_string())

print("\n" + "=" * 70)
print("DISTRICT SUMMARY")
print("=" * 70)
summary = df.groupby(['district', 'district_ru']).agg({
    'temperature': 'mean',
    'precipitation': 'sum',
    'latitude': 'first',
    'longitude': 'first'
}).round(2)
print(summary)

print("\n✅ You can now use this Excel file in the dashboard!")
print("   Run: python dashboard.py")
print("=" * 70)
