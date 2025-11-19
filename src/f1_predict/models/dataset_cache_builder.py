"""Build and manage dataset cache for multi-modal training."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DatasetCacheBuilder:
    """Build training dataset cache from processed race data.

    Generates dataset_cache.json containing training samples with:
    - Speed trace image paths (from SpeedTraceGenerator)
    - Tabular features (weather, tire, driver, circuit)
    - Metadata about cache structure and creation
    """

    def __init__(
        self,
        processed_data_dir: str = 'data/processed',
        output_dir: str = 'data/multimodal/cache'
    ) -> None:
        """Initialize dataset cache builder.

        Args:
            processed_data_dir: Directory containing processed race data
            output_dir: Directory to save generated cache
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.output_dir = Path(output_dir)

    def extract_lap_speeds(
        self,
        race_id: str,  # noqa: ARG002
        driver_id: str,  # noqa: ARG002
        lap_data: list[float]
    ) -> list[float]:
        """Extract and validate lap speeds from race telemetry.

        Args:
            race_id: Unique race identifier
            driver_id: Unique driver identifier
            lap_data: List of lap speeds (numeric values)

        Returns:
            List of float lap speeds

        Raises:
            TypeError: If lap_data contains non-numeric values
            AttributeError: If lap_data is None
        """
        if lap_data is None:
            raise AttributeError("lap_data cannot be None")

        try:
            # Validate all values are numeric
            return [float(speed) for speed in lap_data]
        except (TypeError, ValueError) as e:
            raise TypeError(f"lap_data must contain numeric values: {e}") from e

    def extract_tabular_features(
        self,
        race_id: str,  # noqa: ARG002
        driver_id: str,  # noqa: ARG002
        weather_data: dict[str, Any],
        tire_data: dict[str, Any],
        driver_data: dict[str, Any],
        circuit_data: dict[str, Any]
    ) -> dict[str, float]:
        """Extract tabular features from various data sources.

        Args:
            race_id: Unique race identifier
            driver_id: Unique driver identifier
            weather_data: Weather conditions (temp, humidity, etc.)
            tire_data: Tire information (compound, degradation, etc.)
            driver_data: Driver information (age, experience, etc.)
            circuit_data: Circuit information (elevation, length, etc.)

        Returns:
            Dictionary of extracted features as floats
        """
        features: dict[str, float] = {}
        self._extract_weather_features(weather_data, features)
        self._extract_tire_features(tire_data, features)
        self._extract_driver_features(driver_data, features)
        self._extract_circuit_features(circuit_data, features)
        return features

    def _extract_weather_features(
        self,
        weather_data: dict[str, Any],
        features: dict[str, float]
    ) -> None:
        """Extract weather features into features dictionary."""
        if not weather_data:
            return
        if 'temp' in weather_data:
            features['temp'] = float(weather_data['temp'])
        if 'humidity' in weather_data:
            features['humidity'] = float(weather_data['humidity'])
        if 'wind_speed' in weather_data:
            features['wind_speed'] = float(weather_data['wind_speed'])

    def _extract_tire_features(
        self,
        tire_data: dict[str, Any],
        features: dict[str, float]
    ) -> None:
        """Extract tire features into features dictionary."""
        if not tire_data:
            return
        compound_map = {'soft': 1.0, 'medium': 2.0, 'hard': 3.0}
        if 'compound' in tire_data:
            features['tire_compound'] = compound_map.get(
                tire_data['compound'].lower(), 0.0
            )
        if 'age_laps' in tire_data:
            features['tire_age_laps'] = float(tire_data['age_laps'])

    def _extract_driver_features(
        self,
        driver_data: dict[str, Any],
        features: dict[str, float]
    ) -> None:
        """Extract driver features into features dictionary."""
        if not driver_data:
            return
        if 'age' in driver_data:
            features['driver_age'] = float(driver_data['age'])
        if 'experience_years' in driver_data:
            features['driver_experience_years'] = float(driver_data['experience_years'])

    def _extract_circuit_features(
        self,
        circuit_data: dict[str, Any],
        features: dict[str, float]
    ) -> None:
        """Extract circuit features into features dictionary."""
        if not circuit_data:
            return
        if 'elevation' in circuit_data:
            features['circuit_elevation'] = float(circuit_data['elevation'])
        if 'length_km' in circuit_data:
            features['circuit_length_km'] = float(circuit_data['length_km'])

    def build_cache(self, races: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate complete dataset cache from races.

        Args:
            races: List of race dictionaries with structure:
                {
                    'race_id': str,
                    'date': str,
                    'drivers': [
                        {
                            'driver_id': str,
                            'lap_data': list[float],
                            'weather_data': dict,
                            'tire_data': dict,
                            'driver_data': dict,
                            'circuit_data': dict
                        },
                        ...
                    ]
                }

        Returns:
            Cache dictionary with samples and metadata

        Raises:
            KeyError: If races missing required fields
            TypeError: If race structure is invalid
        """
        cache: dict[str, Any] = {
            'metadata': {
                'version': '1.0',
                'created_at': datetime.now(tz=timezone.utc).isoformat(),
                'sample_count': 0,
                'total_races': len(races)
            },
            'samples': []
        }

        for race in races:
            race_id = race['race_id']
            race_date = race['date']
            drivers = race.get('drivers', [])

            for driver in drivers:
                driver_id = driver['driver_id']
                lap_data = driver['lap_data']
                weather_data = driver.get('weather_data', {})
                tire_data = driver.get('tire_data', {})
                driver_data = driver.get('driver_data', {})
                circuit_data = driver.get('circuit_data', {})

                # Extract features
                self.extract_lap_speeds(race_id, driver_id, lap_data)
                features = self.extract_tabular_features(
                    race_id, driver_id,
                    weather_data, tire_data, driver_data, circuit_data
                )

                # Build sample
                sample: dict[str, Any] = {
                    'race_id': race_id,
                    'driver_id': driver_id,
                    'date': race_date,
                    'image_path': f'data/multimodal/speed_traces/{race_id}/{driver_id}.png',
                    'features': list(features.values()),
                    'feature_names': list(features.keys())
                }

                cache['samples'].append(sample)

        # Update sample count
        cache['metadata']['sample_count'] = len(cache['samples'])

        logger.info(f"Built cache with {len(cache['samples'])} samples from {len(races)} races")

        return cache

    def save_cache(
        self,
        cache: dict[str, Any],
        filename: str = 'dataset_cache.json'
    ) -> str:
        """Save cache to JSON file.

        Args:
            cache: Cache dictionary to save
            filename: Output filename

        Returns:
            Path to saved cache file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(cache, f, indent=2)

        logger.info(f"Saved cache to {output_path}")

        return str(output_path)

    def load_cache(
        self,
        filename: str = 'dataset_cache.json'
    ) -> dict[str, Any]:
        """Load cache from JSON file.

        Args:
            filename: Input filename

        Returns:
            Loaded cache dictionary

        Raises:
            FileNotFoundError: If cache file not found
        """
        cache_path = self.output_dir / filename

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        with open(cache_path) as f:
            cache = json.load(f)

        logger.info(f"Loaded cache from {cache_path}")

        return cache

    def validate_cache(self, cache: dict[str, Any]) -> bool:
        """Validate cache structure and completeness.

        Args:
            cache: Cache dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        # Check required top-level keys
        if 'samples' not in cache or 'metadata' not in cache:
            logger.warning("Cache missing required keys: samples, metadata")
            return False

        # Check metadata required keys
        metadata = cache['metadata']
        if 'version' not in metadata or 'created_at' not in metadata:
            logger.warning("Cache metadata missing required keys")
            return False

        # Validate samples completeness
        samples = cache['samples']
        if not isinstance(samples, list):
            logger.warning("Cache samples must be a list")
            return False

        for i, sample in enumerate(samples):
            required_keys = {'race_id', 'driver_id', 'image_path', 'features'}
            if not all(key in sample for key in required_keys):
                logger.warning(f"Sample {i} missing required fields: {required_keys}")
                return False

        logger.info(f"Cache validation passed ({len(samples)} samples)")

        return True
