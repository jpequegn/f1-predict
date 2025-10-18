"""Prediction utilities for web interface.

This module provides helpers for:
- Loading and caching ML models
- Preparing feature data for predictions
- Generating and formatting predictions
- Managing prediction state
"""

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import structlog

from f1_predict.data.collector import F1DataCollector
from f1_predict.features.engineering import DriverFormCalculator
from f1_predict.models import (
    EnsemblePredictor,
    LightGBMRacePredictor,
    RandomForestRacePredictor,
    XGBoostRacePredictor,
)

logger = structlog.get_logger(__name__)


class PredictionManager:
    """Manages ML model loading, prediction generation, and result formatting."""

    def __init__(self):
        """Initialize prediction manager."""
        self.logger = logger.bind(component="prediction_manager")
        self.collector = F1DataCollector()
        self.form_calculator = DriverFormCalculator(window_size=5)

    def get_upcoming_races(self) -> list[dict]:
        """Get list of upcoming races with metadata.

        Returns:
            List of race dictionaries with name, circuit, date, location
        """
        try:
            # Load race schedules
            schedule_file = Path("data/raw/race_schedules_2020_2024.csv")
            if not schedule_file.exists():
                self.logger.warning("Schedule file not found, returning empty list")
                return []

            schedules_df = pd.read_csv(schedule_file)

            # Filter for future races (simplified - just return last season races)
            races = []
            for _, row in schedules_df.iterrows():
                races.append(
                    {
                        "race_id": row.get("race_id", f"race_{row.get('round')}"),
                        "name": row.get("name", "Unknown"),
                        "circuit": row.get("circuit", "Unknown"),
                        "date": row.get("date", "TBD"),
                        "location": row.get("location", "Unknown"),
                        "round": row.get("round", 0),
                        "season": row.get("season", 2024),
                    }
                )

            return sorted(races, key=lambda x: x.get("round", 0))
        except Exception as e:
            self.logger.error(f"Error loading upcoming races: {e}")
            return []

    def load_model(self, model_type: str) -> tuple[Any, dict]:
        """Load and cache ML model.

        Args:
            model_type: Type of model ('ensemble', 'xgboost', 'lightgbm', 'random_forest')

        Returns:
            Tuple of (model, metadata_dict)

        Raises:
            ValueError: If model type is invalid or model cannot be loaded
        """
        model_map = {
            "ensemble": EnsemblePredictor,
            "xgboost": XGBoostRacePredictor,
            "lightgbm": LightGBMRacePredictor,
            "random_forest": RandomForestRacePredictor,
        }

        if model_type not in model_map:
            msg = f"Invalid model type: {model_type}. Must be one of {list(model_map.keys())}"
            raise ValueError(msg)

        try:
            # For now, create a new instance - in production would load from saved weights
            if model_type == "ensemble":
                # Create ensemble from individual models
                xgb = XGBoostRacePredictor()
                lgb = LightGBMRacePredictor()
                rf = RandomForestRacePredictor()
                model = EnsemblePredictor(
                    models=[xgb, lgb, rf], weights=[0.4, 0.3, 0.3], voting="soft"
                )
            else:
                model = model_map[model_type]()

            metadata = {
                "type": model_type,
                "accuracy": 0.72,  # Placeholder
                "training_date": "2024-10-01",
                "version": "1.0.0",
            }

            self.logger.info(f"Loaded model: {model_type}")
            return model, metadata
        except Exception as e:
            self.logger.error(f"Error loading model {model_type}: {e}")
            raise

    def prepare_race_features(
        self, race_id: str, season: int = 2024  # noqa: ARG002
    ) -> pd.DataFrame:
        """Prepare feature data for a specific race.

        Args:
            race_id: Identifier for the race (reserved for future use)
            season: Racing season

        Returns:
            DataFrame with prepared features for all drivers

        Raises:
            ValueError: If race data cannot be prepared
        """
        try:
            # Load historical race results
            results_file = Path(f"data/raw/race_results_{season}.csv")
            if not results_file.exists():
                raise FileNotFoundError(f"Race results file not found: {results_file}")

            results_df = pd.read_csv(results_file)

            # Prepare features for each driver
            drivers = results_df["driver_id"].unique()
            features_list = []

            for driver_id in drivers:
                # Calculate driver form score
                form_score = self.form_calculator.calculate_form_score(
                    results_df, driver_id
                )

                # Basic features (placeholder - expand with more features)
                features_list.append(
                    {
                        "driver_id": driver_id,
                        "qualifying_position": 10,  # Would come from qualifying data
                        "driver_form_score": form_score,
                        "team_reliability_score": 85.0,  # Would calculate from team results
                        "circuit_performance_score": 80.0,  # Would come from circuit history
                    }
                )

            features_df = pd.DataFrame(features_list)
            self.logger.info(f"Prepared features for {len(features_df)} drivers")
            return features_df
        except Exception as e:
            self.logger.error(f"Error preparing race features: {e}")
            raise

    def generate_prediction(
        self,
        model: Any,
        features: pd.DataFrame,
        race_name: str,
    ) -> dict:
        """Generate prediction using model and features.

        Args:
            model: Trained/initialized ML model
            features: DataFrame with prepared features
            race_name: Name of the race for reporting

        Returns:
            Dictionary with prediction results
        """
        try:
            # Generate predictions
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)
            else:
                probabilities = model.predict(features)

            # Sort by confidence descending
            if isinstance(probabilities, dict):
                sorted_preds = sorted(
                    probabilities.items(), key=lambda x: x[1], reverse=True
                )
            else:
                # Handle numpy array output
                sorted_indices = probabilities.argsort()[::-1]
                sorted_preds = [
                    (features.iloc[idx]["driver_id"], float(probabilities[idx]))
                    for idx in sorted_indices
                ]

            # Format results
            results = {
                "race": race_name,
                "predictions": [
                    {"position": i + 1, "driver_id": driver, "confidence": conf}
                    for i, (driver, conf) in enumerate(sorted_preds)
                ],
                "podium": sorted_preds[:3],
            }

            self.logger.info(f"Generated prediction for {race_name}")
            return results
        except Exception as e:
            self.logger.error(f"Error generating prediction: {e}")
            raise

    def format_prediction_results(
        self, prediction: dict, drivers_info: Optional[dict] = None
    ) -> pd.DataFrame:
        """Format prediction results for display.

        Args:
            prediction: Prediction dictionary from generate_prediction
            drivers_info: Optional dict mapping driver_id to driver name/team

        Returns:
            DataFrame formatted for Streamlit display
        """
        rows = []
        for pred in prediction["predictions"]:
            driver_id = pred["driver_id"]
            driver_name = (
                drivers_info.get(driver_id, {}).get("name", driver_id)
                if drivers_info
                else driver_id
            )
            team_name = (
                drivers_info.get(driver_id, {}).get("team", "N/A")
                if drivers_info
                else "N/A"
            )

            rows.append(
                {
                    "Position": pred["position"],
                    "Driver": driver_name,
                    "Team": team_name,
                    "Confidence": f"{pred['confidence']:.1%}",
                    "Confidence Score": pred["confidence"],  # For sorting
                }
            )

        return pd.DataFrame(rows)

    def export_prediction(
        self, prediction: dict, format: str = "csv"
    ) -> Optional[bytes]:
        """Export prediction to CSV or JSON format.

        Args:
            prediction: Prediction dictionary
            format: Export format ('csv' or 'json')

        Returns:
            Bytes of exported data, or None if export fails
        """
        try:
            if format == "csv":
                df = pd.DataFrame(prediction["predictions"])
                return df.to_csv(index=False).encode("utf-8")
            if format == "json":
                import json

                return json.dumps(prediction, indent=2).encode("utf-8")
            raise ValueError(f"Unsupported export format: {format}")
        except Exception as e:
            self.logger.error(f"Error exporting prediction: {e}")
            return None
