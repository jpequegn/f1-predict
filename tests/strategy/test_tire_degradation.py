"""Tests for tire degradation model."""

import numpy as np
import pytest

from f1_predict.strategy.tire_degradation import (
    TireCompound,
    TireDegradationConfig,
    TireDegradationModel,
)


class TestTireCompound:
    """Tests for TireCompound enum."""

    def test_compound_values(self):
        """Test tire compound enum values."""
        assert TireCompound.SOFT.value == "soft"
        assert TireCompound.MEDIUM.value == "medium"
        assert TireCompound.HARD.value == "hard"
        assert TireCompound.INTERMEDIATE.value == "intermediate"
        assert TireCompound.WET.value == "wet"


class TestTireDegradationConfig:
    """Tests for TireDegradationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TireDegradationConfig()

        assert config.soft_deg_rate == 0.08
        assert config.medium_deg_rate == 0.05
        assert config.hard_deg_rate == 0.03
        assert config.optimal_temp == 90.0
        assert config.temp_deg_multiplier == 0.02

    def test_custom_config(self):
        """Test custom configuration."""
        config = TireDegradationConfig(
            soft_deg_rate=0.10,
            track_abrasiveness=1.2
        )

        assert config.soft_deg_rate == 0.10
        assert config.track_abrasiveness == 1.2


class TestTireDegradationModel:
    """Tests for TireDegradationModel."""

    def test_initialization_default(self):
        """Test model initialization with default config."""
        model = TireDegradationModel()

        assert model.config is not None
        assert model.config.soft_deg_rate == 0.08

    def test_initialization_custom(self):
        """Test model initialization with custom config."""
        config = TireDegradationConfig(soft_deg_rate=0.10)
        model = TireDegradationModel(config)

        assert model.config.soft_deg_rate == 0.10

    def test_soft_degrades_faster_than_hard(self):
        """Soft tires should degrade faster than hard tires."""
        model = TireDegradationModel()

        soft_deg = model.calculate_lap_time_delta(
            compound=TireCompound.SOFT,
            lap_number=10,
            track_temp=45.0,
            fuel_load=80.0
        )

        hard_deg = model.calculate_lap_time_delta(
            compound=TireCompound.HARD,
            lap_number=10,
            track_temp=45.0,
            fuel_load=80.0
        )

        assert soft_deg > hard_deg

    def test_temperature_effect_on_degradation(self):
        """Temperature deviation from optimal should increase degradation."""
        model = TireDegradationModel()

        optimal_deg = model.calculate_lap_time_delta(
            compound=TireCompound.MEDIUM,
            lap_number=5,
            track_temp=90.0,  # Optimal temp
            fuel_load=100.0
        )

        hot_deg = model.calculate_lap_time_delta(
            compound=TireCompound.MEDIUM,
            lap_number=5,
            track_temp=110.0,  # 20°C above optimal
            fuel_load=100.0
        )

        assert hot_deg > optimal_deg

    def test_driver_style_aggressive_increases_wear(self):
        """Aggressive driving should increase tire wear."""
        model = TireDegradationModel()

        neutral_deg = model.calculate_lap_time_delta(
            compound=TireCompound.MEDIUM,
            lap_number=10,
            track_temp=90.0,
            fuel_load=100.0,
            driver_style="neutral"
        )

        aggressive_deg = model.calculate_lap_time_delta(
            compound=TireCompound.MEDIUM,
            lap_number=10,
            track_temp=90.0,
            fuel_load=100.0,
            driver_style="aggressive"
        )

        assert aggressive_deg > neutral_deg

    def test_conservative_driving_reduces_wear(self):
        """Conservative driving should reduce tire wear."""
        model = TireDegradationModel()

        neutral_deg = model.calculate_lap_time_delta(
            compound=TireCompound.MEDIUM,
            lap_number=10,
            track_temp=90.0,
            fuel_load=100.0,
            driver_style="neutral"
        )

        conservative_deg = model.calculate_lap_time_delta(
            compound=TireCompound.MEDIUM,
            lap_number=10,
            track_temp=90.0,
            fuel_load=100.0,
            driver_style="conservative"
        )

        assert conservative_deg < neutral_deg

    def test_fuel_load_effect(self):
        """Heavier fuel load should reduce tire stress."""
        model = TireDegradationModel()

        heavy_fuel = model.calculate_lap_time_delta(
            compound=TireCompound.MEDIUM,
            lap_number=10,
            track_temp=90.0,
            fuel_load=110.0
        )

        light_fuel = model.calculate_lap_time_delta(
            compound=TireCompound.MEDIUM,
            lap_number=10,
            track_temp=90.0,
            fuel_load=20.0
        )

        # Light fuel = more tire stress in corners
        assert light_fuel > heavy_fuel

    def test_degradation_increases_with_laps(self):
        """Degradation should increase with more laps."""
        model = TireDegradationModel()

        lap5 = model.calculate_lap_time_delta(
            TireCompound.MEDIUM, 5, 90.0, 100.0
        )
        lap15 = model.calculate_lap_time_delta(
            TireCompound.MEDIUM, 15, 90.0, 90.0
        )

        assert lap15 > lap5

    def test_tire_cliff_effect(self):
        """Degradation should spike after cliff threshold."""
        model = TireDegradationModel()

        # Find lap number that causes cliff (>3s degradation)
        before_cliff = None
        after_cliff = None

        for lap in range(1, 50):
            deg = model.calculate_lap_time_delta(
                TireCompound.SOFT, lap, 110.0, 50.0
            )
            if deg < 3.0:
                before_cliff = deg
            elif deg > 3.0:
                after_cliff = deg
                break

        # Cliff multiplier is 1.5, so jump should be substantial
        if before_cliff and after_cliff:
            ratio = after_cliff / before_cliff
            assert ratio > 1.4  # Should see cliff effect

    def test_negative_lap_number_raises_error(self):
        """Negative lap number should raise ValueError."""
        model = TireDegradationModel()

        with pytest.raises(ValueError, match="lap_number must be >= 1"):
            model.calculate_lap_time_delta(
                TireCompound.MEDIUM, -1, 90.0, 100.0
            )

    def test_negative_fuel_load_raises_error(self):
        """Negative fuel load should raise ValueError."""
        model = TireDegradationModel()

        with pytest.raises(ValueError, match="fuel_load must be >= 0"):
            model.calculate_lap_time_delta(
                TireCompound.MEDIUM, 10, 90.0, -10.0
            )

    def test_predict_stint_performance(self):
        """Test stint performance prediction."""
        model = TireDegradationModel()

        conditions = {
            "track_temp": 45.0,
            "fuel_load": 110.0,
            "driver_style": "neutral"
        }

        deltas = model.predict_stint_performance(
            TireCompound.MEDIUM,
            stint_length=20,
            conditions=conditions
        )

        assert isinstance(deltas, np.ndarray)
        assert len(deltas) == 20
        assert all(deltas[i] <= deltas[i+1] for i in range(len(deltas)-1))

    def test_predict_stint_invalid_length(self):
        """Test stint prediction with invalid length."""
        model = TireDegradationModel()

        conditions = {"track_temp": 45.0, "fuel_load": 110.0}

        with pytest.raises(ValueError, match="stint_length must be >= 1"):
            model.predict_stint_performance(
                TireCompound.MEDIUM, 0, conditions
            )

    def test_predict_stint_missing_conditions(self):
        """Test stint prediction with missing conditions."""
        model = TireDegradationModel()

        with pytest.raises(KeyError):
            model.predict_stint_performance(
                TireCompound.MEDIUM, 10, {}
            )

    def test_estimate_optimal_stint_length(self):
        """Test optimal stint length estimation."""
        model = TireDegradationModel()

        conditions = {
            "track_temp": 45.0,
            "fuel_load": 110.0,
            "driver_style": "neutral"
        }

        stint_length = model.estimate_optimal_stint_length(
            TireCompound.SOFT,
            conditions,
            max_deg_threshold=2.0
        )

        assert isinstance(stint_length, int)
        assert 1 <= stint_length <= 60

        # Verify degradation at this length is <= threshold
        deg = model.calculate_lap_time_delta(
            TireCompound.SOFT,
            stint_length,
            conditions["track_temp"],
            max(0, conditions["fuel_load"] - stint_length),
            conditions["driver_style"]
        )
        assert deg <= 2.0

    def test_optimal_stint_hard_longer_than_soft(self):
        """Hard tires should have longer optimal stint."""
        model = TireDegradationModel()

        conditions = {
            "track_temp": 45.0,
            "fuel_load": 110.0
        }

        soft_stint = model.estimate_optimal_stint_length(
            TireCompound.SOFT, conditions
        )
        hard_stint = model.estimate_optimal_stint_length(
            TireCompound.HARD, conditions
        )

        assert hard_stint > soft_stint
