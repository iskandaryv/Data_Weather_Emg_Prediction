"""
Gradio web interface for Weather Emergency Prediction System.
Provides interactive dashboard for Rostov region emergency prediction.
"""
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from typing import Optional

from ..utils.config import ROSTOV_COORDS, API_HOST, API_PORT, EMERGENCY_TYPES
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

# API base URL
API_BASE_URL = f"http://{API_HOST}:{API_PORT}/api"


def predict_emergency(date_input, temp, precip, humidity, wind, pressure):
    """Make emergency prediction via API."""
    try:
        payload = {
            "date": date_input,
            "temperature": float(temp),
            "precipitation": float(precip),
            "humidity": float(humidity),
            "wind_speed": float(wind),
            "pressure": float(pressure)
        }

        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()

        result = response.json()

        status = "⚠️ EMERGENCY LIKELY" if result['will_occur'] else "✅ NO EMERGENCY"
        probability = result['emergency_probability'] * 100
        confidence = result['confidence'] * 100

        return (
            status,
            f"{probability:.1f}%",
            f"{confidence:.1f}%",
            result.get('emergency_type', 'N/A')
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Error", "N/A", "N/A", str(e)


def get_historical_data(start_date, end_date, limit=500):
    """Fetch historical data from API."""
    try:
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit
        }

        response = requests.get(f"{API_BASE_URL}/data/historical", params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        df = pd.DataFrame(data['data'])

        if len(df) > 0 and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        return df

    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()


def plot_temperature_trends(start_date, end_date):
    """Plot temperature trends over time."""
    df = get_historical_data(start_date, end_date)

    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    # Aggregate by month
    df_monthly = df.groupby(df['date'].dt.to_period('M')).agg({
        'temperature': ['mean', 'min', 'max']
    }).reset_index()

    df_monthly.columns = ['date', 'mean', 'min', 'max']
    df_monthly['date'] = df_monthly['date'].dt.to_timestamp()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_monthly['date'],
        y=df_monthly['mean'],
        name='Average',
        line=dict(color='orange', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df_monthly['date'],
        y=df_monthly['max'],
        name='Maximum',
        line=dict(color='red', width=1, dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=df_monthly['date'],
        y=df_monthly['min'],
        name='Minimum',
        line=dict(color='blue', width=1, dash='dot')
    ))

    fig.update_layout(
        title="Temperature Trends - Rostov-on-Don",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def plot_precipitation(start_date, end_date):
    """Plot precipitation over time."""
    df = get_historical_data(start_date, end_date)

    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)

    # Aggregate by month
    df_monthly = df.groupby(df['date'].dt.to_period('M')).agg({
        'precipitation': 'sum'
    }).reset_index()

    df_monthly.columns = ['date', 'precipitation']
    df_monthly['date'] = df_monthly['date'].dt.to_timestamp()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_monthly['date'],
        y=df_monthly['precipitation'],
        name='Precipitation',
        marker_color='lightblue'
    ))

    fig.update_layout(
        title="Monthly Precipitation - Rostov-on-Don",
        xaxis_title="Date",
        yaxis_title="Precipitation (mm)",
        template='plotly_white'
    )

    return fig


def plot_emergency_distribution():
    """Plot emergency type distribution."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        response.raise_for_status()

        stats = response.json()
        emergency_types = stats.get('emergency_types', {})

        if not emergency_types:
            return go.Figure().add_annotation(text="No emergency data available", showarrow=False)

        fig = go.Figure(data=[go.Pie(
            labels=list(emergency_types.keys()),
            values=list(emergency_types.values()),
            hole=0.3
        )])

        fig.update_layout(
            title="Emergency Types Distribution",
            template='plotly_white'
        )

        return fig

    except Exception as e:
        logger.error(f"Error plotting emergency distribution: {e}")
        return go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)


def get_statistics(start_date, end_date):
    """Get statistics for date range."""
    try:
        params = {
            "start_date": start_date,
            "end_date": end_date
        }

        response = requests.get(f"{API_BASE_URL}/stats", params=params, timeout=10)
        response.raise_for_status()

        stats = response.json()

        return (
            f"{stats['total_days']} days",
            f"{stats['total_emergencies']} events",
            f"{stats['emergency_rate']*100:.2f}%",
            f"{stats['avg_temperature']:.1f}°C",
            f"{stats['avg_precipitation']:.1f}mm"
        )

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return "Error", "Error", "Error", "Error", "Error"


def upload_file(file, data_type):
    """Upload data file to API."""
    try:
        if file is None:
            return "No file selected"

        with open(file.name, 'rb') as f:
            files = {'file': (file.name.split('/')[-1], f, 'application/octet-stream')}
            params = {'data_type': data_type}

            response = requests.post(
                f"{API_BASE_URL}/data/upload",
                files=files,
                params=params,
                timeout=30
            )
            response.raise_for_status()

        result = response.json()
        return f"✅ {result['message']}"

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return f"❌ Error: {str(e)}"


def train_model(model_type):
    """Train model via API."""
    try:
        params = {'model_type': model_type}

        response = requests.post(
            f"{API_BASE_URL}/model/train",
            params=params,
            timeout=300
        )
        response.raise_for_status()

        result = response.json()

        return (
            f"✅ {result['message']}",
            f"Model: {result['model_type']}",
            f"Samples: {result['training_samples']}",
            f"Features: {result['features']}",
            f"Metrics: {result['metrics']}"
        )

    except Exception as e:
        logger.error(f"Error training model: {e}")
        return f"❌ Error: {str(e)}", "", "", "", ""


# Create Gradio Interface
def create_interface():
    """Create Gradio web interface."""

    with gr.Blocks(title="Weather Emergency Prediction - Rostov", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            """
            # 🌡️ Weather Emergency Prediction System
            ## Rostov-on-Don Region
            Predict weather-related emergencies using machine learning
            """
        )

        with gr.Tabs():

            # Tab 1: Prediction
            with gr.Tab("🔮 Emergency Prediction"):
                gr.Markdown("### Enter weather conditions to predict emergency occurrence")

                with gr.Row():
                    with gr.Column():
                        date_input = gr.Textbox(
                            label="Date (YYYY-MM-DD)",
                            value=datetime.now().strftime("%Y-%m-%d")
                        )
                        temp_input = gr.Slider(
                            label="Temperature (°C)",
                            minimum=-40, maximum=45, value=20, step=0.1
                        )
                        precip_input = gr.Slider(
                            label="Precipitation (mm)",
                            minimum=0, maximum=100, value=0, step=0.1
                        )

                    with gr.Column():
                        humidity_input = gr.Slider(
                            label="Humidity (%)",
                            minimum=0, maximum=100, value=50, step=1
                        )
                        wind_input = gr.Slider(
                            label="Wind Speed (m/s)",
                            minimum=0, maximum=30, value=5, step=0.1
                        )
                        pressure_input = gr.Slider(
                            label="Pressure (hPa)",
                            minimum=950, maximum=1050, value=1013, step=1
                        )

                predict_btn = gr.Button("🔍 Predict Emergency", variant="primary")

                with gr.Row():
                    status_output = gr.Textbox(label="Status")
                    prob_output = gr.Textbox(label="Emergency Probability")
                    conf_output = gr.Textbox(label="Confidence")
                    type_output = gr.Textbox(label="Emergency Type")

                predict_btn.click(
                    fn=predict_emergency,
                    inputs=[date_input, temp_input, precip_input, humidity_input, wind_input, pressure_input],
                    outputs=[status_output, prob_output, conf_output, type_output]
                )

            # Tab 2: Dashboard
            with gr.Tab("📊 Dashboard"):
                gr.Markdown("### Historical Data Analysis")

                with gr.Row():
                    start_date = gr.Textbox(
                        label="Start Date",
                        value="2015-01-01"
                    )
                    end_date = gr.Textbox(
                        label="End Date",
                        value=datetime.now().strftime("%Y-%m-%d")
                    )

                refresh_btn = gr.Button("🔄 Refresh Dashboard")

                with gr.Row():
                    total_days = gr.Textbox(label="Total Days")
                    total_emerg = gr.Textbox(label="Total Emergencies")
                    emerg_rate = gr.Textbox(label="Emergency Rate")
                    avg_temp = gr.Textbox(label="Avg Temperature")
                    avg_precip = gr.Textbox(label="Avg Precipitation")

                with gr.Row():
                    temp_plot = gr.Plot(label="Temperature Trends")

                with gr.Row():
                    precip_plot = gr.Plot(label="Precipitation")
                    emerg_plot = gr.Plot(label="Emergency Distribution")

                refresh_btn.click(
                    fn=get_statistics,
                    inputs=[start_date, end_date],
                    outputs=[total_days, total_emerg, emerg_rate, avg_temp, avg_precip]
                ).then(
                    fn=plot_temperature_trends,
                    inputs=[start_date, end_date],
                    outputs=temp_plot
                ).then(
                    fn=plot_precipitation,
                    inputs=[start_date, end_date],
                    outputs=precip_plot
                ).then(
                    fn=plot_emergency_distribution,
                    inputs=[],
                    outputs=emerg_plot
                )

            # Tab 3: Data Management
            with gr.Tab("📁 Data Management"):
                gr.Markdown("### Upload Weather or Emergency Data")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Upload Data File (CSV or Excel)**")
                        file_upload = gr.File(label="Select File")
                        data_type_radio = gr.Radio(
                            choices=["weather", "emergency"],
                            label="Data Type",
                            value="weather"
                        )
                        upload_btn = gr.Button("📤 Upload Data")
                        upload_status = gr.Textbox(label="Upload Status")

                        upload_btn.click(
                            fn=upload_file,
                            inputs=[file_upload, data_type_radio],
                            outputs=upload_status
                        )

            # Tab 4: Model Training
            with gr.Tab("🤖 Model Training"):
                gr.Markdown("### Train Emergency Prediction Model")

                model_type_dropdown = gr.Dropdown(
                    choices=["random_forest", "gradient_boosting", "logistic"],
                    label="Model Type",
                    value="random_forest"
                )

                train_btn = gr.Button("🚀 Train Model", variant="primary")

                with gr.Column():
                    train_status = gr.Textbox(label="Training Status")
                    model_info = gr.Textbox(label="Model Info")
                    samples_info = gr.Textbox(label="Training Samples")
                    features_info = gr.Textbox(label="Features")
                    metrics_info = gr.Textbox(label="Performance Metrics")

                train_btn.click(
                    fn=train_model,
                    inputs=[model_type_dropdown],
                    outputs=[train_status, model_info, samples_info, features_info, metrics_info]
                )

        gr.Markdown(
            """
            ---
            **Note:** This system uses machine learning to predict weather-related emergencies.
            Predictions are based on historical data and should be used as a decision support tool.
            """
        )

    return demo


def launch_interface(share=False):
    """Launch Gradio interface."""
    logger.info("Launching Gradio interface...")

    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share
    )


if __name__ == "__main__":
    launch_interface()
