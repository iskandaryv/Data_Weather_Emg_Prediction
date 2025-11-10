"""
Main entry point for Weather Emergency Prediction System.
"""
import argparse
from pathlib import Path

from src.utils.logger import setup_logger
from src.utils.config import LOG_DIR

logger = setup_logger(__name__, log_file=LOG_DIR / "app.log")


def run_api():
    """Run FastAPI backend."""
    import uvicorn
    from src.utils.config import API_HOST, API_PORT, API_WORKERS

    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        workers=API_WORKERS,
        reload=False
    )


def run_web():
    """Run Gradio web interface."""
    from src.web.app import launch_interface

    logger.info("Starting Gradio web interface...")
    launch_interface(share=False)


def run_training():
    """Run model training pipeline."""
    from src.data import WeatherDataLoader, EmergencyDataLoader, DataPreprocessor
    from src.models import EmergencyPredictor

    logger.info("Starting training pipeline...")

    # Load data
    weather_loader = WeatherDataLoader()
    emergency_loader = EmergencyDataLoader()

    logger.info("Loading weather data...")
    weather_df = weather_loader.load_weather_data()

    logger.info("Loading emergency data...")
    emergency_df = emergency_loader.load_emergency_data()

    if len(emergency_df) == 0:
        logger.info("No emergency data found, generating synthetic data...")
        emergency_df = emergency_loader.generate_synthetic_emergencies(weather_df)

    # Preprocess
    preprocessor = DataPreprocessor()

    logger.info("Preprocessing data...")
    weather_clean = preprocessor.remove_outliers(weather_df, ['temperature', 'precipitation'])
    weather_clean = preprocessor.create_time_features(weather_clean)
    weather_clean = preprocessor.create_rolling_features(weather_clean, windows=[7, 14])
    weather_clean = preprocessor.create_lag_features(weather_clean, lags=[1, 3, 7])

    # Merge with emergency data
    merged_df = preprocessor.merge_weather_emergency(weather_clean, emergency_df, window_days=3)

    # Prepare training data
    X, y = preprocessor.prepare_training_data(merged_df, target_col='has_emergency')
    X_scaled = preprocessor.scale_features(X, fit=True)

    # Train model
    logger.info("Training model...")
    predictor = EmergencyPredictor(model_type='random_forest')
    metrics = predictor.train(X_scaled, y, validation=True)

    logger.info(f"Training complete. Metrics: {metrics}")

    # Save model
    predictor.save("emergency_model_random_forest.pkl")
    logger.info("Model saved successfully")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Weather Emergency Prediction System")

    parser.add_argument(
        "command",
        choices=["api", "web", "train", "all"],
        help="Command to run"
    )

    args = parser.parse_args()

    if args.command == "api":
        run_api()
    elif args.command == "web":
        run_web()
    elif args.command == "train":
        run_training()
    elif args.command == "all":
        # Run training first, then start services
        import multiprocessing

        logger.info("Running training first...")
        run_training()

        logger.info("Starting all services...")
        api_process = multiprocessing.Process(target=run_api)
        web_process = multiprocessing.Process(target=run_web)

        api_process.start()
        web_process.start()

        try:
            api_process.join()
            web_process.join()
        except KeyboardInterrupt:
            logger.info("Shutting down services...")
            api_process.terminate()
            web_process.terminate()


if __name__ == "__main__":
    main()
