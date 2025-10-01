"""Tests for pit stop optimizer."""

import pytest

from f1_predict.strategy.pit_optimizer import PitStopOptimizer
from f1_predict.strategy.tire_degradation import TireCompound


class TestPitStopOptimizer:
    """Tests for PitStopOptimizer."""

    def test_initialization_default(self):
        """Test optimizer initialization with default parameters."""
        optimizer = PitStopOptimizer(race_laps=50)

        assert optimizer.race_laps == 50
        assert optimizer.pit_loss == 23.0
        assert optimizer.min_stint == 8
        assert optimizer.tire_model is not None

    def test_initialization_custom(self):
        """Test optimizer with custom parameters."""
        optimizer = PitStopOptimizer(
            race_laps=60,
            pit_loss_time=20.0,
            min_stint_length=10
        )

        assert optimizer.race_laps == 60
        assert optimizer.pit_loss == 20.0
        assert optimizer.min_stint == 10

    def test_invalid_race_laps(self):
        """Test initialization with invalid race laps."""
        with pytest.raises(ValueError, match="race_laps must be >= 1"):
            PitStopOptimizer(race_laps=0)

    def test_invalid_min_stint(self):
        """Test initialization with invalid min stint."""
        with pytest.raises(ValueError, match="min_stint_length must be >= 1"):
            PitStopOptimizer(race_laps=50, min_stint_length=0)

    def test_optimize_strategy_basic(self):
        """Test basic strategy optimization."""
        optimizer = PitStopOptimizer(race_laps=50)

        strategy = optimizer.optimize_strategy(
            available_compounds=[
                TireCompound.SOFT,
                TireCompound.MEDIUM,
                TireCompound.HARD
            ]
        )

        assert "num_stops" in strategy
        assert "total_time_loss" in strategy
        assert "pit_stops" in strategy
        assert "compounds_used" in strategy

        # Should have at least 1 stop
        assert strategy["num_stops"] >= 1
        assert len(strategy["pit_stops"]) == strategy["num_stops"]

    def test_mandatory_compounds_enforced(self):
        """Test that mandatory compound rule is enforced."""
        optimizer = PitStopOptimizer(race_laps=50)

        strategy = optimizer.optimize_strategy(
            available_compounds=[
                TireCompound.SOFT,
                TireCompound.MEDIUM,
                TireCompound.HARD
            ],
            mandatory_compounds=2
        )

        # Should use at least 2 different compounds
        assert len(strategy["compounds_used"]) >= 2

    def test_insufficient_compounds_raises_error(self):
        """Test error when not enough compounds available."""
        optimizer = PitStopOptimizer(race_laps=50)

        with pytest.raises(ValueError, match="at least 2 different compounds"):
            optimizer.optimize_strategy(
                available_compounds=[TireCompound.SOFT],
                mandatory_compounds=2
            )

    def test_pit_stops_within_race(self):
        """Test that all pit stops occur within race laps."""
        optimizer = PitStopOptimizer(race_laps=50)

        strategy = optimizer.optimize_strategy(
            available_compounds=[
                TireCompound.SOFT,
                TireCompound.MEDIUM,
                TireCompound.HARD
            ]
        )

        for stop in strategy["pit_stops"]:
            assert 1 <= stop["lap"] <= 50

    def test_strategy_with_custom_conditions(self):
        """Test strategy optimization with custom conditions."""
        optimizer = PitStopOptimizer(race_laps=60)

        conditions = {
            "track_temp": 50.0,
            "fuel_load": 100.0,
            "driver_style": "aggressive"
        }

        strategy = optimizer.optimize_strategy(
            available_compounds=[TireCompound.MEDIUM, TireCompound.HARD],
            track_conditions=conditions
        )

        assert strategy is not None
        assert strategy["num_stops"] >= 1

    def test_fallback_strategy(self):
        """Test fallback strategy creation."""
        optimizer = PitStopOptimizer(race_laps=50)

        # Create fallback for testing
        fallback = optimizer._create_fallback_strategy(
            [TireCompound.MEDIUM, TireCompound.HARD],
            {"track_temp": 45.0, "fuel_load": 110.0}
        )

        assert fallback["num_stops"] == 1
        assert len(fallback["pit_stops"]) == 1
        assert len(fallback["compounds_used"]) == 2

    def test_short_race_strategy(self):
        """Test strategy for short race."""
        optimizer = PitStopOptimizer(race_laps=15, min_stint_length=5)

        strategy = optimizer.optimize_strategy(
            available_compounds=[TireCompound.SOFT, TireCompound.MEDIUM]
        )

        # Short race should have 1-2 stops max
        assert strategy["num_stops"] <= 2

    def test_long_race_strategy(self):
        """Test strategy for longer race."""
        optimizer = PitStopOptimizer(race_laps=70)

        strategy = optimizer.optimize_strategy(
            available_compounds=[
                TireCompound.SOFT,
                TireCompound.MEDIUM,
                TireCompound.HARD
            ]
        )

        # Longer race might have 1-3 stops
        assert 1 <= strategy["num_stops"] <= 3

    def test_strategy_time_loss_positive(self):
        """Test that total time loss is positive."""
        optimizer = PitStopOptimizer(race_laps=50)

        strategy = optimizer.optimize_strategy(
            available_compounds=[TireCompound.MEDIUM, TireCompound.HARD]
        )

        assert strategy["total_time_loss"] > 0

    def test_calculate_stint_cost(self):
        """Test stint cost calculation."""
        optimizer = PitStopOptimizer(race_laps=50)

        conditions = {
            "track_temp": 45.0,
            "fuel_load": 110.0,
            "driver_style": "neutral"
        }

        cost = optimizer._calculate_stint_cost(
            TireCompound.MEDIUM,
            stint_length=20,
            conditions=conditions
        )

        assert cost >= 0
        assert isinstance(cost, float)

    def test_zero_stint_length_cost(self):
        """Test that zero stint length has zero cost."""
        optimizer = PitStopOptimizer(race_laps=50)

        cost = optimizer._calculate_stint_cost(
            TireCompound.MEDIUM,
            0,
            {"track_temp": 45.0, "fuel_load": 110.0}
        )

        assert cost == 0.0

    def test_select_optimal_compound(self):
        """Test optimal compound selection."""
        optimizer = PitStopOptimizer(race_laps=50)

        # Should prefer medium for balance
        compound = optimizer._select_optimal_compound(
            available=[TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD],
            already_used=set()
        )

        assert compound == TireCompound.MEDIUM

    def test_select_optimal_without_medium(self):
        """Test compound selection without medium available."""
        optimizer = PitStopOptimizer(race_laps=50)

        compound = optimizer._select_optimal_compound(
            available=[TireCompound.SOFT, TireCompound.HARD],
            already_used=set()
        )

        # Should choose hard for durability
        assert compound == TireCompound.HARD
