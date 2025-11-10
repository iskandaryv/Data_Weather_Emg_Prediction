"""
Geographic utilities using GeoPandas and Google Maps API.
"""
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from typing import Tuple, Optional, List
import folium
from folium.plugins import HeatMap
import requests

from .config import DEFAULT_GEO_LOCATION, GOOGLE_MAPS_API_KEY, GEO_CONFIG
from .logger import setup_logger

logger = setup_logger(__name__)


class GeoDataHandler:
    """Handle geographic data operations."""

    def __init__(self, center_lat: Optional[float] = None, center_lon: Optional[float] = None):
        """
        Initialize geo data handler.

        Args:
            center_lat: Center latitude (defaults to config)
            center_lon: Center longitude (defaults to config)
        """
        self.center_lat = center_lat or DEFAULT_GEO_LOCATION['lat']
        self.center_lon = center_lon or DEFAULT_GEO_LOCATION['lon']
        self.crs = GEO_CONFIG['crs']

    def create_geodataframe(
        self,
        df: pd.DataFrame,
        lat_col: str = 'latitude',
        lon_col: str = 'longitude'
    ) -> gpd.GeoDataFrame:
        """
        Convert DataFrame to GeoDataFrame.

        Args:
            df: Input DataFrame
            lat_col: Latitude column name
            lon_col: Longitude column name

        Returns:
            GeoDataFrame with Point geometries
        """
        logger.info(f"Creating GeoDataFrame from {len(df)} records")

        # Create Point geometries
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=self.crs)

        logger.info(f"Created GeoDataFrame with CRS: {gdf.crs}")
        return gdf

    def add_buffer(
        self,
        gdf: gpd.GeoDataFrame,
        buffer_km: float = None
    ) -> gpd.GeoDataFrame:
        """
        Add buffer around points.

        Args:
            gdf: Input GeoDataFrame
            buffer_km: Buffer distance in kilometers

        Returns:
            GeoDataFrame with buffer geometries
        """
        if buffer_km is None:
            buffer_km = GEO_CONFIG['buffer_km']

        logger.info(f"Adding {buffer_km}km buffer to geometries")

        # Convert to projected CRS for accurate distance calculations
        gdf_proj = gdf.to_crs('EPSG:3857')  # Web Mercator

        # Buffer in meters
        buffer_m = buffer_km * 1000
        gdf_proj['geometry'] = gdf_proj.geometry.buffer(buffer_m)

        # Convert back to WGS84
        gdf_buffer = gdf_proj.to_crs(self.crs)

        return gdf_buffer

    def spatial_join(
        self,
        left_gdf: gpd.GeoDataFrame,
        right_gdf: gpd.GeoDataFrame,
        how: str = 'inner',
        predicate: str = 'intersects'
    ) -> gpd.GeoDataFrame:
        """
        Perform spatial join between two GeoDataFrames.

        Args:
            left_gdf: Left GeoDataFrame
            right_gdf: Right GeoDataFrame
            how: Join type ('inner', 'left', 'right')
            predicate: Spatial predicate ('intersects', 'contains', 'within')

        Returns:
            Joined GeoDataFrame
        """
        logger.info(f"Performing spatial join with predicate: {predicate}")

        result = gpd.sjoin(left_gdf, right_gdf, how=how, predicate=predicate)

        logger.info(f"Spatial join resulted in {len(result)} records")
        return result

    def create_folium_map(
        self,
        gdf: gpd.GeoDataFrame,
        zoom_start: int = 10,
        tiles: str = 'OpenStreetMap'
    ) -> folium.Map:
        """
        Create Folium map from GeoDataFrame.

        Args:
            gdf: Input GeoDataFrame
            zoom_start: Initial zoom level
            tiles: Map tiles ('OpenStreetMap', 'CartoDB positron', etc.)

        Returns:
            Folium Map object
        """
        logger.info("Creating Folium map")

        # Create base map
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=zoom_start,
            tiles=tiles
        )

        # Add markers for each point
        for idx, row in gdf.iterrows():
            if row.geometry.geom_type == 'Point':
                popup_text = f"Date: {row.get('date', 'N/A')}<br>"
                popup_text += f"Type: {row.get('type', 'N/A')}<br>"
                popup_text += f"Severity: {row.get('severity', 'N/A')}"

                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    popup=popup_text,
                    color='red',
                    fill=True,
                    fillColor='red'
                ).add_to(m)

        return m

    def create_heatmap(
        self,
        gdf: gpd.GeoDataFrame,
        intensity_col: Optional[str] = None,
        zoom_start: int = 10
    ) -> folium.Map:
        """
        Create heat map from GeoDataFrame.

        Args:
            gdf: Input GeoDataFrame
            intensity_col: Column for intensity values
            zoom_start: Initial zoom level

        Returns:
            Folium Map with HeatMap layer
        """
        logger.info("Creating heat map")

        # Create base map
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=zoom_start,
            tiles='CartoDB positron'
        )

        # Prepare heat data
        heat_data = []
        for idx, row in gdf.iterrows():
            if row.geometry.geom_type == 'Point':
                intensity = row[intensity_col] if intensity_col else 1.0
                heat_data.append([row.geometry.y, row.geometry.x, intensity])

        # Add heat map layer
        HeatMap(heat_data).add_to(m)

        return m

    def get_region_boundary(
        self,
        buffer_km: float = 100
    ) -> gpd.GeoDataFrame:
        """
        Create boundary polygon around center point.

        Args:
            buffer_km: Buffer distance in kilometers

        Returns:
            GeoDataFrame with boundary polygon
        """
        logger.info(f"Creating region boundary with {buffer_km}km buffer")

        # Create center point
        center_point = Point(self.center_lon, self.center_lat)
        gdf = gpd.GeoDataFrame({'geometry': [center_point]}, crs=self.crs)

        # Add buffer
        gdf_buffer = self.add_buffer(gdf, buffer_km=buffer_km)

        return gdf_buffer

    def calculate_distance(
        self,
        gdf: gpd.GeoDataFrame,
        target_lat: float,
        target_lon: float
    ) -> gpd.GeoDataFrame:
        """
        Calculate distance from each point to target location.

        Args:
            gdf: Input GeoDataFrame
            target_lat: Target latitude
            target_lon: Target longitude

        Returns:
            GeoDataFrame with distance column (in km)
        """
        logger.info("Calculating distances")

        # Create target point
        target_point = Point(target_lon, target_lat)
        target_gdf = gpd.GeoDataFrame({'geometry': [target_point]}, crs=self.crs)

        # Convert to projected CRS for accurate distance
        gdf_proj = gdf.to_crs('EPSG:3857')
        target_proj = target_gdf.to_crs('EPSG:3857')

        # Calculate distance in meters, convert to km
        gdf['distance_km'] = gdf_proj.geometry.distance(target_proj.geometry[0]) / 1000

        return gdf

    def export_to_geojson(
        self,
        gdf: gpd.GeoDataFrame,
        filename: str
    ):
        """
        Export GeoDataFrame to GeoJSON file.

        Args:
            gdf: Input GeoDataFrame
            filename: Output filename
        """
        logger.info(f"Exporting to GeoJSON: {filename}")

        gdf.to_file(filename, driver='GeoJSON')

        logger.info("Export completed")

    def export_to_html(
        self,
        folium_map: folium.Map,
        filename: str
    ):
        """
        Export Folium map to HTML file.

        Args:
            folium_map: Folium Map object
            filename: Output filename
        """
        logger.info(f"Exporting map to HTML: {filename}")

        folium_map.save(filename)

        logger.info("Export completed")


class GoogleMapsAPI:
    """Google Maps API integration for Russia."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Google Maps API client.

        Args:
            api_key: Google Maps API key
        """
        self.api_key = api_key or GOOGLE_MAPS_API_KEY
        if not self.api_key:
            logger.warning("Google Maps API key not configured")

    def geocode_location(self, address: str) -> Optional[Tuple[float, float]]:
        """
        Geocode address to coordinates.

        Args:
            address: Address string

        Returns:
            Tuple of (latitude, longitude) or None
        """
        if not self.api_key:
            logger.error("API key not configured")
            return None

        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': address,
            'key': self.api_key,
            'region': 'ru'  # Bias results to Russia
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data['status'] == 'OK':
                location = data['results'][0]['geometry']['location']
                return (location['lat'], location['lng'])
            else:
                logger.error(f"Geocoding failed: {data['status']}")
                return None

        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            return None

    def reverse_geocode(
        self,
        lat: float,
        lon: float
    ) -> Optional[str]:
        """
        Reverse geocode coordinates to address.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Address string or None
        """
        if not self.api_key:
            logger.error("API key not configured")
            return None

        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'latlng': f"{lat},{lon}",
            'key': self.api_key,
            'language': 'ru'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data['status'] == 'OK':
                return data['results'][0]['formatted_address']
            else:
                logger.error(f"Reverse geocoding failed: {data['status']}")
                return None

        except Exception as e:
            logger.error(f"Reverse geocoding error: {e}")
            return None

    def get_elevation(
        self,
        lat: float,
        lon: float
    ) -> Optional[float]:
        """
        Get elevation for coordinates.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Elevation in meters or None
        """
        if not self.api_key:
            logger.error("API key not configured")
            return None

        url = "https://maps.googleapis.com/maps/api/elevation/json"
        params = {
            'locations': f"{lat},{lon}",
            'key': self.api_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data['status'] == 'OK':
                return data['results'][0]['elevation']
            else:
                logger.error(f"Elevation API failed: {data['status']}")
                return None

        except Exception as e:
            logger.error(f"Elevation API error: {e}")
            return None
