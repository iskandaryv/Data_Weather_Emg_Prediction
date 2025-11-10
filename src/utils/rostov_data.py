"""
Rostov-on-Don specific geographic data and districts.
"""
from typing import Dict, List
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

# Rostov-on-Don city center
ROSTOV_CENTER = {
    "lat": 47.2357,
    "lon": 39.7015,
    "name": "Rostov-on-Don City Center"
}

# Rostov-on-Don Districts with approximate boundaries
ROSTOV_DISTRICTS = {
    "Leninsky": {
        "name": "Leninsky District",
        "name_ru": "Ленинский район",
        "center_lat": 47.2220,
        "center_lon": 39.7180,
        "population": 150000,
        "area_km2": 43.5,
        "description": "Central district with historical center"
    },
    "Kirovsky": {
        "name": "Kirovsky District",
        "name_ru": "Кировский район",
        "center_lat": 47.2580,
        "center_lon": 39.7850,
        "population": 97000,
        "area_km2": 65.3,
        "description": "Industrial and residential district"
    },
    "Oktyabrsky": {
        "name": "Oktyabrsky District",
        "name_ru": "Октябрьский район",
        "center_lat": 47.2750,
        "center_lon": 39.7320,
        "population": 207000,
        "area_km2": 45.2,
        "description": "Northern residential district"
    },
    "Pervomaisky": {
        "name": "Pervomaisky District",
        "name_ru": "Первомайский район",
        "center_lat": 47.2180,
        "center_lon": 39.6420,
        "population": 180000,
        "area_km2": 53.7,
        "description": "Western district with parks"
    },
    "Proletarsky": {
        "name": "Proletarsky District",
        "name_ru": "Пролетарский район",
        "center_lat": 47.1980,
        "center_lon": 39.7680,
        "population": 156000,
        "area_km2": 42.8,
        "description": "Southern residential area"
    },
    "Sovetsky": {
        "name": "Sovetsky District",
        "name_ru": "Советский район",
        "center_lat": 47.2420,
        "center_lon": 39.6850,
        "population": 93000,
        "area_km2": 38.4,
        "description": "Western residential district"
    },
    "Zheleznodorozhny": {
        "name": "Zheleznodorozhny District",
        "name_ru": "Железнодорожный район",
        "center_lat": 47.2640,
        "center_lon": 39.7180,
        "population": 178000,
        "area_km2": 48.6,
        "description": "Railway and industrial area"
    },
    "Voroshilovsky": {
        "name": "Voroshilovsky District",
        "name_ru": "Ворошиловский район",
        "center_lat": 47.2380,
        "center_lon": 39.7420,
        "population": 134000,
        "area_km2": 36.9,
        "description": "Central residential district"
    }
}

# Key places/landmarks in Rostov-on-Don
ROSTOV_LANDMARKS = [
    {
        "name": "Rostov Arena",
        "name_ru": "Ростов Арена",
        "type": "stadium",
        "lat": 47.2085,
        "lon": 39.7378,
        "district": "Leninsky"
    },
    {
        "name": "Gorky Park",
        "name_ru": "Парк Горького",
        "type": "park",
        "lat": 47.2233,
        "lon": 39.7488,
        "district": "Leninsky"
    },
    {
        "name": "Rostov State Musical Theater",
        "name_ru": "Ростовский музыкальный театр",
        "type": "theater",
        "lat": 47.2346,
        "lon": 39.7177,
        "district": "Leninsky"
    },
    {
        "name": "Bolshaya Sadovaya Street",
        "name_ru": "Большая Садовая улица",
        "type": "street",
        "lat": 47.2236,
        "lon": 39.7197,
        "district": "Leninsky"
    },
    {
        "name": "Rostov Zoo",
        "name_ru": "Ростовский зоопарк",
        "type": "zoo",
        "lat": 47.2647,
        "lon": 39.7061,
        "district": "Oktyabrsky"
    },
    {
        "name": "Don River Embankment",
        "name_ru": "Набережная реки Дон",
        "type": "waterfront",
        "lat": 47.2153,
        "lon": 39.7089,
        "district": "Leninsky"
    },
    {
        "name": "Rostov Regional Museum",
        "name_ru": "Ростовский областной музей",
        "type": "museum",
        "lat": 47.2239,
        "lon": 39.7188,
        "district": "Leninsky"
    },
    {
        "name": "Central Market",
        "name_ru": "Центральный рынок",
        "type": "market",
        "lat": 47.2298,
        "lon": 39.7142,
        "district": "Leninsky"
    }
]


def get_rostov_districts_dataframe() -> pd.DataFrame:
    """
    Get Rostov-on-Don districts as DataFrame.

    Returns:
        DataFrame with district information
    """
    districts_list = []
    for key, district in ROSTOV_DISTRICTS.items():
        districts_list.append({
            "district_id": key,
            "name": district["name"],
            "name_ru": district["name_ru"],
            "latitude": district["center_lat"],
            "longitude": district["center_lon"],
            "population": district["population"],
            "area_km2": district["area_km2"],
            "description": district["description"]
        })

    return pd.DataFrame(districts_list)


def get_rostov_landmarks_dataframe() -> pd.DataFrame:
    """
    Get Rostov-on-Don landmarks as DataFrame.

    Returns:
        DataFrame with landmark information
    """
    return pd.DataFrame(ROSTOV_LANDMARKS)


def get_rostov_districts_geodataframe() -> gpd.GeoDataFrame:
    """
    Get Rostov-on-Don districts as GeoDataFrame with Point geometries.

    Returns:
        GeoDataFrame with district geometries
    """
    df = get_rostov_districts_dataframe()
    geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    return gdf


def get_rostov_landmarks_geodataframe() -> gpd.GeoDataFrame:
    """
    Get Rostov-on-Don landmarks as GeoDataFrame.

    Returns:
        GeoDataFrame with landmark geometries
    """
    df = get_rostov_landmarks_dataframe()
    geometry = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    return gdf


def create_district_polygons() -> gpd.GeoDataFrame:
    """
    Create approximate polygon boundaries for Rostov districts.

    Returns:
        GeoDataFrame with district polygons
    """
    # Approximate polygons for each district
    # These are simplified boundaries based on district centers

    polygons = {}

    # Leninsky (Central)
    polygons["Leninsky"] = Polygon([
        (39.68, 47.21),
        (39.74, 47.21),
        (39.74, 47.24),
        (39.68, 47.24)
    ])

    # Kirovsky (Northeast)
    polygons["Kirovsky"] = Polygon([
        (39.74, 47.24),
        (39.82, 47.24),
        (39.82, 47.28),
        (39.74, 47.28)
    ])

    # Oktyabrsky (North)
    polygons["Oktyabrsky"] = Polygon([
        (39.68, 47.26),
        (39.76, 47.26),
        (39.76, 47.30),
        (39.68, 47.30)
    ])

    # Pervomaisky (West)
    polygons["Pervomaisky"] = Polygon([
        (39.60, 47.20),
        (39.68, 47.20),
        (39.68, 47.24),
        (39.60, 47.24)
    ])

    # Proletarsky (South)
    polygons["Proletarsky"] = Polygon([
        (39.73, 47.18),
        (39.80, 47.18),
        (39.80, 47.21),
        (39.73, 47.21)
    ])

    # Sovetsky (Southwest)
    polygons["Sovetsky"] = Polygon([
        (39.65, 47.23),
        (39.71, 47.23),
        (39.71, 47.26),
        (39.65, 47.26)
    ])

    # Zheleznodorozhny (Central-North)
    polygons["Zheleznodorozhny"] = Polygon([
        (39.69, 47.25),
        (39.75, 47.25),
        (39.75, 47.28),
        (39.69, 47.28)
    ])

    # Voroshilovsky (Central-East)
    polygons["Voroshilovsky"] = Polygon([
        (39.71, 47.22),
        (39.77, 47.22),
        (39.77, 47.25),
        (39.71, 47.25)
    ])

    # Create GeoDataFrame
    data = []
    for district_id, polygon in polygons.items():
        district_info = ROSTOV_DISTRICTS[district_id]
        data.append({
            "district_id": district_id,
            "name": district_info["name"],
            "name_ru": district_info["name_ru"],
            "population": district_info["population"],
            "geometry": polygon
        })

    gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')
    return gdf


def generate_sample_excel_with_geodata(filename: str = "sample_rostov_weather.xlsx"):
    """
    Generate sample Excel file with geodata (lat/lon columns).

    Args:
        filename: Output filename
    """
    import openpyxl
    from datetime import datetime, timedelta
    import numpy as np

    # Generate sample data for different districts
    data = []

    start_date = datetime(2024, 1, 1)

    for i in range(365):  # One year of data
        current_date = start_date + timedelta(days=i)

        # Generate data for each district
        for district_id, district in ROSTOV_DISTRICTS.items():
            temp = 10 + 15 * np.sin(2 * np.pi * i / 365) + np.random.normal(0, 3)
            precip = np.random.gamma(2, 3) if np.random.random() < 0.3 else 0
            humidity = np.clip(50 + 20 * np.sin(2 * np.pi * i / 365) + np.random.normal(0, 10), 0, 100)
            wind = np.abs(np.random.gamma(3, 2))
            pressure = 1013 + np.random.normal(0, 8)

            data.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "district": district["name"],
                "district_ru": district["name_ru"],
                "latitude": district["center_lat"],
                "longitude": district["center_lon"],
                "temperature": round(temp, 1),
                "precipitation": round(precip, 1),
                "humidity": round(humidity, 1),
                "wind_speed": round(wind, 1),
                "pressure": round(pressure, 1)
            })

    df = pd.DataFrame(data)
    df.to_excel(filename, index=False, sheet_name="Weather Data")

    print(f"✅ Generated sample Excel file: {filename}")
    print(f"   Records: {len(df)}")
    print(f"   Districts: {len(ROSTOV_DISTRICTS)}")
    print(f"   Columns: {list(df.columns)}")

    return df
