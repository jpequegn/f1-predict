"""Unit tests for pit stop strategy optimization."""

import pytest

from f1_predict.simulation.core.driver_state import TireCompound
from f1_predict.simulation.engine.pit_strategy import (
    PitStopOptimizer,
    PitStopWindow,
    TireStrategy,
)


class TestPitStopOptimizer:
    """Test PitStopOptimizer class."""

    def test_optimizer_creation(self):
        """Test creating pit stop optimizer."""
        optimizer = PitStopOptimizer(
            total_laps=58,
            avg_lap_time=81.5,
            fuel_capacity_laps=60,
        )
        assert optimizer.total_laps == 58
        assert optimizer.avg_lap_time == 81.5
        assert optimizer.fuel_capacity_laps == 60

    def test_optimize_strategy_full_fuel(self):
        """Test strategy optimization with full fuel."""
        optimizer = PitStopOptimizer(
            total_laps=58,
            avg_lap_time=81.5,
            fuel_capacity_laps=60,
        )
        strategy = optimizer.optimize_strategy(fuel_available=100.0)

        # With full fuel for 60-lap race, should be able to do one stop
        assert strategy in [TireStrategy.ONE_STOP, TireStrategy.NO_STOP]

    def test_optimize_strategy_low_fuel(self):
        """Test strategy optimization with low fuel."""
        optimizer = PitStopOptimizer(
            total_laps=58,
            avg_lap_time=81.5,
            fuel_capacity_laps=60,
        )
        strategy = optimizer.optimize_strategy(fuel_available=30.0)

        # With low fuel, should need multiple stops
        assert strategy in [TireStrategy.TWO_STOP, TireStrategy.THREE_STOP]

    def test_optimize_strategy_medium_fuel(self):
        """Test strategy optimization with medium fuel."""
        optimizer = PitStopOptimizer(
            total_laps=58,
            avg_lap_time=81.5,
            fuel_capacity_laps=60,
        )
        strategy = optimizer.optimize_strategy(fuel_available=60.0)

        # With medium fuel, typically two stops
        assert strategy == TireStrategy.TWO_STOP

    def test_calculate_pit_windows_no_stop(self):
        """Test pit windows for no-stop strategy."""
        optimizer = PitStopOptimizer(
            total_laps=58,
            avg_lap_time=81.5,
        )
        windows = optimizer.calculate_pit_windows(TireStrategy.NO_STOP)

        assert len(windows) == 0

    def test_calculate_pit_windows_one_stop(self):
        """Test pit windows for one-stop strategy."""
        optimizer = PitStopOptimizer(
            total_laps=58,
            avg_lap_time=81.5,
        )
        windows = optimizer.calculate_pit_windows(TireStrategy.ONE_STOP)

        assert len(windows) == 1
        window = windows[0]
        assert isinstance(window, PitStopWindow)
        assert window.start_lap < window.end_lap
        assert window.start_lap <= window.recommended_lap <= window.end_lap

    def test_calculate_pit_windows_two_stop(self):
        """Test pit windows for two-stop strategy."""
        optimizer = PitStopOptimizer(
            total_laps=58,
            avg_lap_time=81.5,
        )
        windows = optimizer.calculate_pit_windows(TireStrategy.TWO_STOP)

        assert len(windows) == 2
        # First pit should be before second pit
        assert windows[0].recommended_lap < windows[1].recommended_lap

    def test_calculate_pit_windows_three_stop(self):
        """Test pit windows for three-stop strategy."""
        optimizer = PitStopOptimizer(
            total_laps=58,
            avg_lap_time=81.5,
        )
        windows = optimizer.calculate_pit_windows(TireStrategy.THREE_STOP)

        assert len(windows) == 3
        # Each pit should be progressively later
        for i in range(2):
            assert windows[i].recommended_lap < windows[i + 1].recommended_lap

    def test_pit_windows_reasonable_spacing(self):
        """Test that pit windows are reasonably spaced."""
        optimizer = PitStopOptimizer(
            total_laps=58,
            avg_lap_time=81.5,
        )
        windows = optimizer.calculate_pit_windows(TireStrategy.TWO_STOP)

        # Windows should not overlap
        assert windows[0].end_lap <= windows[1].start_lap

    def test_select_tire_compound_dry(self):
        """Test tire selection in dry conditions."""
        optimizer = PitStopOptimizer(total_laps=58, avg_lap_time=81.5)

        # Short stint remaining
        tire = optimizer.select_tire_compound(30, 10, "dry")
        assert tire == TireCompound.SOFT  # Fastest for short stints

        # Medium stint
        tire = optimizer.select_tire_compound(30, 20, "dry")
        assert tire == TireCompound.MEDIUM

        # Long stint
        tire = optimizer.select_tire_compound(30, 40, "dry")
        assert tire == TireCompound.HARD  # Most durable

    def test_select_tire_compound_wet(self):
        """Test tire selection in wet conditions."""
        optimizer = PitStopOptimizer(total_laps=58, avg_lap_time=81.5)

        tire = optimizer.select_tire_compound(30, 20, "wet")
        assert tire == TireCompound.WET

    def test_select_tire_compound_intermediate(self):
        """Test tire selection in intermediate conditions."""
        optimizer = PitStopOptimizer(total_laps=58, avg_lap_time=81.5)

        tire = optimizer.select_tire_compound(30, 20, "intermediate")
        assert tire == TireCompound.INTERMEDIATE

    def test_calculate_stint_duration(self):
        """Test stint duration calculation."""
        optimizer = PitStopOptimizer(
            total_laps=58,
            avg_lap_time=81.5,
            fuel_capacity_laps=60,
        )

        # SOFT tires: ~35 lap limit
        duration = optimizer.calculate_stint_duration(TireCompound.SOFT, 100.0)
        assert duration <= 35

        # HARD tires: ~60 lap limit
        duration = optimizer.calculate_stint_duration(TireCompound.HARD, 100.0)
        assert duration <= 60

    def test_calculate_stint_duration_limited_fuel(self):
        """Test stint duration with limited fuel."""
        optimizer = PitStopOptimizer(
            total_laps=58,
            avg_lap_time=81.5,
            fuel_capacity_laps=60,
        )

        # With 50% fuel, should limit to ~30 laps
        duration = optimizer.calculate_stint_duration(TireCompound.SOFT, 50.0)
        assert duration <= 30

    def test_estimate_time_loss(self):
        """Test pit stop time loss estimation."""
        optimizer = PitStopOptimizer(total_laps=58, avg_lap_time=81.5)

        # No stops = no loss
        loss = optimizer.estimate_time_loss(0)
        assert loss == 0.0

        # One stop: ~65 seconds (25s pit + 40s regain)
        loss_one = optimizer.estimate_time_loss(1)
        assert 60 < loss_one < 70

        # Two stops: ~130 seconds
        loss_two = optimizer.estimate_time_loss(2)
        assert loss_two > loss_one

    def test_time_loss_scales_linearly(self):
        """Test that time loss scales roughly linearly with stops."""
        optimizer = PitStopOptimizer(total_laps=58, avg_lap_time=81.5)

        loss_one = optimizer.estimate_time_loss(1)
        loss_two = optimizer.estimate_time_loss(2)
        loss_three = optimizer.estimate_time_loss(3)

        # Each stop should add roughly the same time
        diff_1_2 = loss_two - loss_one
        diff_2_3 = loss_three - loss_two
        # Differences should be similar (within 10% tolerance)
        assert abs(diff_1_2 - diff_2_3) < loss_one * 0.1


class TestTireStrategy:
    """Test TireStrategy enum."""

    def test_tire_strategies_exist(self):
        """Test all tire strategies are defined."""
        assert TireStrategy.NO_STOP.value == "no_stop"
        assert TireStrategy.ONE_STOP.value == "one_stop"
        assert TireStrategy.TWO_STOP.value == "two_stop"
        assert TireStrategy.THREE_STOP.value == "three_stop"


class TestPitStopWindow:
    """Test PitStopWindow dataclass."""

    def test_pit_stop_window_creation(self):
        """Test creating pit stop window."""
        window = PitStopWindow(
            start_lap=20,
            end_lap=30,
            recommended_lap=25,
        )
        assert window.start_lap == 20
        assert window.end_lap == 30
        assert window.recommended_lap == 25

    def test_pit_stop_window_recommended_within_range(self):
        """Test that recommended lap is within window."""
        window = PitStopWindow(
            start_lap=20,
            end_lap=30,
            recommended_lap=25,
        )
        assert window.start_lap <= window.recommended_lap <= window.end_lap


class TestTireCompoundDegradation:
    """Test tire degradation rates."""

    def test_tire_degradation_rates_defined(self):
        """Test that tire degradation rates are defined."""
        optimizer = PitStopOptimizer(total_laps=58, avg_lap_time=81.5)

        # All compounds should have degradation rates
        assert TireCompound.SOFT in optimizer.TIRE_DEGRADATION
        assert TireCompound.MEDIUM in optimizer.TIRE_DEGRADATION
        assert TireCompound.HARD in optimizer.TIRE_DEGRADATION
        assert TireCompound.INTERMEDIATE in optimizer.TIRE_DEGRADATION
        assert TireCompound.WET in optimizer.TIRE_DEGRADATION

    def test_tire_degradation_rates_realistic(self):
        """Test that degradation rates are realistic."""
        optimizer = PitStopOptimizer(total_laps=58, avg_lap_time=81.5)

        rates = optimizer.TIRE_DEGRADATION

        # Soft should degrade faster than hard
        assert rates[TireCompound.SOFT] > rates[TireCompound.HARD]

        # All rates should be positive
        for rate in rates.values():
            assert rate > 0
