"""
Rostov-on-Don Interactive Dashboard with Districts and Places.
"""
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium import plugins
import geopandas as gpd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.rostov_data import (
    ROSTOV_CENTER,
    ROSTOV_DISTRICTS,
    get_rostov_districts_dataframe,
    get_rostov_landmarks_dataframe,
    get_rostov_districts_geodataframe,
    get_rostov_landmarks_geodataframe,
    create_district_polygons,
    generate_sample_excel_with_geodata
)
from src.utils.geo_utils import GeoDataHandler

print("=" * 70)
print("ROSTOV-ON-DON WEATHER EMERGENCY DASHBOARD")
print("=" * 70)

# Generate sample data if not exists
try:
    weather_df = pd.read_excel('sample_rostov_weather.xlsx')
    print(f"✅ Loaded existing data: {len(weather_df)} records")
except:
    print("📊 Generating sample weather data with geodata...")
    weather_df = generate_sample_excel_with_geodata('sample_rostov_weather.xlsx')

# Load geodata
districts_df = get_rostov_districts_dataframe()
landmarks_df = get_rostov_landmarks_dataframe()
districts_gdf = get_rostov_districts_geodataframe()
landmarks_gdf = get_rostov_landmarks_geodataframe()
districts_polygons = create_district_polygons()

print(f"✅ Loaded {len(districts_df)} districts")
print(f"✅ Loaded {len(landmarks_df)} landmarks")


def create_rostov_map_with_districts():
    """Create Folium map of Rostov with district boundaries."""
    m = folium.Map(
        location=[ROSTOV_CENTER["lat"], ROSTOV_CENTER["lon"]],
        zoom_start=11,
        tiles='CartoDB positron'
    )

    # Add district polygons
    folium.GeoJson(
        districts_polygons,
        name='Districts',
        style_function=lambda x: {
            'fillColor': '#3388ff',
            'color': '#000000',
            'weight': 2,
            'fillOpacity': 0.2
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['name', 'name_ru', 'population'],
            aliases=['District:', 'Район:', 'Population:'],
            localize=True
        )
    ).add_to(m)

    # Add district centers
    for idx, row in districts_gdf.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=f"<b>{row['name']}</b><br>{row['name_ru']}<br>Population: {row['population']:,}",
            color='red',
            fill=True,
            fillColor='red'
        ).add_to(m)

    # Add landmarks
    for idx, row in landmarks_gdf.iterrows():
        icon_map = {
            'stadium': 'futbol',
            'park': 'tree',
            'theater': 'music',
            'street': 'road',
            'zoo': 'paw',
            'waterfront': 'water',
            'museum': 'landmark',
            'market': 'shopping-cart'
        }

        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"<b>{row['name']}</b><br>{row['name_ru']}<br>Type: {row['type']}",
            icon=folium.Icon(
                color='green',
                icon=icon_map.get(row['type'], 'info-sign'),
                prefix='fa'
            )
        ).add_to(m)

    folium.LayerControl().add_to(m)

    return m


def create_district_heatmap(df, value_col='temperature'):
    """Create heatmap by district."""
    # Aggregate by district
    district_stats = df.groupby('district')[value_col].mean().reset_index()
    district_stats = district_stats.merge(
        districts_df[['name', 'latitude', 'longitude']],
        left_on='district',
        right_on='name'
    )

    m = folium.Map(
        location=[ROSTOV_CENTER["lat"], ROSTOV_CENTER["lon"]],
        zoom_start=11,
        tiles='CartoDB positron'
    )

    # Create heat data
    heat_data = [[row['latitude'], row['longitude'], row[value_col]]
                 for idx, row in district_stats.iterrows()]

    plugins.HeatMap(heat_data, radius=30, blur=25).add_to(m)

    return m


def plot_district_comparison(df, metric='temperature'):
    """Plot comparison across districts."""
    district_stats = df.groupby('district')[metric].agg(['mean', 'min', 'max']).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=district_stats['district'],
        y=district_stats['mean'],
        name='Average',
        marker_color='lightblue'
    ))

    fig.update_layout(
        title=f'{metric.capitalize()} by District - Rostov-on-Don',
        xaxis_title='District',
        yaxis_title=metric.capitalize(),
        xaxis_tickangle=-45,
        height=500,
        showlegend=True
    )

    return fig


def plot_time_series_by_district(df, start_date, end_date, metric='temperature'):
    """Plot time series for selected metric by district."""
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    fig = go.Figure()

    for district in df_filtered['district'].unique():
        district_data = df_filtered[df_filtered['district'] == district]
        district_daily = district_data.groupby('date')[metric].mean().reset_index()

        fig.add_trace(go.Scatter(
            x=pd.to_datetime(district_daily['date']),
            y=district_daily[metric],
            name=district,
            mode='lines'
        ))

    fig.update_layout(
        title=f'{metric.capitalize()} Over Time - Rostov Districts',
        xaxis_title='Date',
        yaxis_title=metric.capitalize(),
        height=500,
        hovermode='x unified'
    )

    return fig


def get_district_statistics(district_name):
    """Get statistics for a specific district."""
    if district_name == "All Districts":
        df_filtered = weather_df
        district_info = f"All Districts - Rostov-on-Don"
        population = sum([d['population'] for d in ROSTOV_DISTRICTS.values()])
    else:
        df_filtered = weather_df[weather_df['district'] == district_name]
        district_key = [k for k, v in ROSTOV_DISTRICTS.items() if v['name'] == district_name][0]
        district_info = ROSTOV_DISTRICTS[district_key]
        population = district_info['population']

    stats = {
        'records': len(df_filtered),
        'avg_temp': df_filtered['temperature'].mean(),
        'max_temp': df_filtered['temperature'].max(),
        'min_temp': df_filtered['temperature'].min(),
        'total_precip': df_filtered['precipitation'].sum(),
        'avg_humidity': df_filtered['humidity'].mean(),
        'population': population
    }

    return (
        f"{stats['records']} records",
        f"{stats['avg_temp']:.1f}°C",
        f"{stats['max_temp']:.1f}°C",
        f"{stats['min_temp']:.1f}°C",
        f"{stats['total_precip']:.0f}mm",
        f"{stats['avg_humidity']:.0f}%",
        f"{stats['population']:,}"
    )


# Create Gradio Dashboard
with gr.Blocks(title="Rostov-on-Don Weather Dashboard", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🌡️ Rostov-on-Don Weather Emergency Dashboard
        ## Interactive Dashboard with 8 Districts and Landmarks
        Real-time weather data visualization for Rostov-on-Don, Russia
        """
    )

    with gr.Tabs():

        # Tab 1: Map View
        with gr.Tab("🗺️ Rostov Districts Map"):
            gr.Markdown("### Interactive map showing all 8 districts and key landmarks")

            map_btn = gr.Button("🔄 Generate Interactive Map", variant="primary")
            map_output = gr.HTML()

            def generate_map_html():
                m = create_rostov_map_with_districts()
                return m._repr_html_()

            map_btn.click(fn=generate_map_html, outputs=map_output)

            # Show districts info
            with gr.Row():
                gr.Dataframe(
                    value=districts_df[['name', 'name_ru', 'population', 'area_km2', 'description']],
                    label="Rostov-on-Don Districts",
                    height=300
                )

        # Tab 2: Heat Map
        with gr.Tab("🔥 Heat Map"):
            gr.Markdown("### Temperature heat map by district")

            with gr.Row():
                heatmap_metric = gr.Dropdown(
                    choices=['temperature', 'precipitation', 'humidity'],
                    value='temperature',
                    label="Select Metric"
                )
                heatmap_btn = gr.Button("Generate Heat Map", variant="primary")

            heatmap_output = gr.HTML()

            def generate_heatmap_html(metric):
                m = create_district_heatmap(weather_df, metric)
                return m._repr_html_()

            heatmap_btn.click(
                fn=generate_heatmap_html,
                inputs=[heatmap_metric],
                outputs=heatmap_output
            )

        # Tab 3: District Comparison
        with gr.Tab("📊 District Comparison"):
            gr.Markdown("### Compare weather metrics across districts")

            with gr.Row():
                comparison_metric = gr.Dropdown(
                    choices=['temperature', 'precipitation', 'humidity', 'wind_speed', 'pressure'],
                    value='temperature',
                    label="Select Metric"
                )
                compare_btn = gr.Button("Compare Districts", variant="primary")

            comparison_plot = gr.Plot()

            compare_btn.click(
                fn=plot_district_comparison,
                inputs=[gr.State(weather_df), comparison_metric],
                outputs=comparison_plot
            )

        # Tab 4: Time Series
        with gr.Tab("📈 Time Series Analysis"):
            gr.Markdown("### Analyze weather trends over time")

            with gr.Row():
                ts_start = gr.Textbox(label="Start Date", value="2024-01-01")
                ts_end = gr.Textbox(label="End Date", value="2024-12-31")
                ts_metric = gr.Dropdown(
                    choices=['temperature', 'precipitation', 'humidity'],
                    value='temperature',
                    label="Metric"
                )

            ts_btn = gr.Button("Generate Time Series", variant="primary")
            ts_plot = gr.Plot()

            ts_btn.click(
                fn=plot_time_series_by_district,
                inputs=[gr.State(weather_df), ts_start, ts_end, ts_metric],
                outputs=ts_plot
            )

        # Tab 5: District Details
        with gr.Tab("🏘️ District Details"):
            gr.Markdown("### Detailed statistics for each district")

            district_select = gr.Dropdown(
                choices=["All Districts"] + [d['name'] for d in ROSTOV_DISTRICTS.values()],
                value="All Districts",
                label="Select District"
            )

            stats_btn = gr.Button("Get Statistics", variant="primary")

            with gr.Row():
                stat_records = gr.Textbox(label="Total Records")
                stat_avg_temp = gr.Textbox(label="Avg Temperature")
                stat_max_temp = gr.Textbox(label="Max Temperature")
                stat_min_temp = gr.Textbox(label="Min Temperature")

            with gr.Row():
                stat_precip = gr.Textbox(label="Total Precipitation")
                stat_humidity = gr.Textbox(label="Avg Humidity")
                stat_population = gr.Textbox(label="Population")

            stats_btn.click(
                fn=get_district_statistics,
                inputs=[district_select],
                outputs=[stat_records, stat_avg_temp, stat_max_temp, stat_min_temp,
                        stat_precip, stat_humidity, stat_population]
            )

        # Tab 6: Landmarks
        with gr.Tab("📍 Landmarks & Places"):
            gr.Markdown("### Key places and landmarks in Rostov-on-Don")

            gr.Dataframe(
                value=landmarks_df[['name', 'name_ru', 'type', 'district']],
                label="Rostov-on-Don Landmarks",
                height=400
            )

    gr.Markdown(
        """
        ---
        **Data Source:** Sample weather data with geodata (latitude/longitude) from Excel file
        **Districts:** 8 administrative districts of Rostov-on-Don
        **Landmarks:** Major places and points of interest
        """
    )

print("\n" + "=" * 70)
print("LAUNCHING DASHBOARD")
print("=" * 70)
print("Dashboard includes:")
print("  ✅ 8 Rostov-on-Don districts with boundaries")
print("  ✅ Weather data with geodata (lat/lon)")
print("  ✅ Interactive maps with Folium")
print("  ✅ Heat maps by district")
print("  ✅ District comparisons")
print("  ✅ Time series analysis")
print("  ✅ Landmarks and places")
print("=" * 70 + "\n")

demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
