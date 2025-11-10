"""
Example usage of GeoPandas and Google Maps API integration.
"""
from src.data import WeatherDataLoader, EmergencyDataLoader
from src.utils.geo_utils import GeoDataHandler, GoogleMapsAPI
from src.utils.config import DEFAULT_GEO_LOCATION
import pandas as pd

print("=" * 70)
print("GEODATA & MAPPING EXAMPLE")
print("=" * 70)

# 1. Load or generate data
print("\n1. Loading data...")
weather_loader = WeatherDataLoader()
emergency_loader = EmergencyDataLoader()

weather_df = weather_loader.load_weather_data()
emergency_df = emergency_loader.load_emergency_data()

if len(emergency_df) == 0:
    print("Generating emergency data...")
    emergency_df = emergency_loader.generate_synthetic_emergencies(weather_df)

print(f"✅ Loaded {len(weather_df)} weather records")
print(f"✅ Loaded {len(emergency_df)} emergency records")

# 2. Create GeoDataFrame
print("\n2. Creating GeoDataFrame...")
geo_handler = GeoDataHandler()

# Convert emergency data to GeoDataFrame
emergency_gdf = geo_handler.create_geodataframe(emergency_df)
print(f"✅ Created GeoDataFrame with {len(emergency_gdf)} geometries")
print(f"   CRS: {emergency_gdf.crs}")

# 3. Add buffer around emergency points
print("\n3. Adding 10km buffer around emergency points...")
buffered_gdf = geo_handler.add_buffer(emergency_gdf.head(10), buffer_km=10)
print(f"✅ Created buffer zones")

# 4. Calculate distances
print("\n4. Calculating distances from center...")
emergency_gdf = geo_handler.calculate_distance(
    emergency_gdf,
    target_lat=DEFAULT_GEO_LOCATION['lat'],
    target_lon=DEFAULT_GEO_LOCATION['lon']
)
print(f"✅ Calculated distances")
print(f"   Min distance: {emergency_gdf['distance_km'].min():.2f} km")
print(f"   Max distance: {emergency_gdf['distance_km'].max():.2f} km")

# 5. Create Folium map
print("\n5. Creating interactive map...")
folium_map = geo_handler.create_folium_map(emergency_gdf.head(50), zoom_start=10)
geo_handler.export_to_html(folium_map, 'emergency_map.html')
print("✅ Map saved to: emergency_map.html")

# 6. Create heat map
print("\n6. Creating heat map...")
heatmap = geo_handler.create_heatmap(emergency_gdf.head(100), intensity_col='severity')
geo_handler.export_to_html(heatmap, 'emergency_heatmap.html')
print("✅ Heat map saved to: emergency_heatmap.html")

# 7. Export to GeoJSON
print("\n7. Exporting to GeoJSON...")
geo_handler.export_to_geojson(emergency_gdf, 'emergencies.geojson')
print("✅ GeoJSON saved to: emergencies.geojson")

# 8. Google Maps API (optional - requires API key)
print("\n8. Google Maps API integration...")
gmaps = GoogleMapsAPI()

if gmaps.api_key:
    print("Testing Google Maps API...")

    # Reverse geocode center point
    address = gmaps.reverse_geocode(
        DEFAULT_GEO_LOCATION['lat'],
        DEFAULT_GEO_LOCATION['lon']
    )
    if address:
        print(f"   Location: {address}")

    # Get elevation
    elevation = gmaps.get_elevation(
        DEFAULT_GEO_LOCATION['lat'],
        DEFAULT_GEO_LOCATION['lon']
    )
    if elevation:
        print(f"   Elevation: {elevation:.1f} meters")
else:
    print("⚠️  Google Maps API key not configured")
    print("   Set GOOGLE_MAPS_API_KEY in .env file")

# 9. Spatial analysis example
print("\n9. Spatial analysis...")
# Get emergencies within 50km of center
center_gdf = geo_handler.get_region_boundary(buffer_km=50)
nearby_emergencies = geo_handler.spatial_join(
    emergency_gdf,
    center_gdf,
    predicate='within'
)
print(f"✅ Found {len(nearby_emergencies)} emergencies within 50km")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"✅ Weather data: {len(weather_df)} records")
print(f"✅ Emergency data: {len(emergency_df)} records")
print(f"✅ GeoDataFrame created with CRS: {emergency_gdf.crs}")
print(f"✅ Maps exported: emergency_map.html, emergency_heatmap.html")
print(f"✅ GeoJSON exported: emergencies.geojson")
print("\nOpen the HTML files in your browser to view the maps!")
print("=" * 70)
