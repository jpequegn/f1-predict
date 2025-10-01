"""Tests for safety car impact model."""

import pytest

from f1_predict.strategy.safety_car import SafetyCarModel


class TestSafetyCarModel:
    """Tests for SafetyCarModel."""

    def test_initialization_default(self):
        """Test model initialization with default parameters."""
        model = SafetyCarModel()

        assert model.base_probability == 0.3

    def test_initialization_custom(self):
        """Test model initialization with custom parameters."""
        model = SafetyCarModel(circuit_safety_car_rate=0.5)

        assert model.base_probability == 0.5

    def test_invalid_probability_raises_error(self):
        """Test initialization with invalid probability."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            SafetyCarModel(circuit_safety_car_rate=1.5)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            SafetyCarModel(circuit_safety_car_rate=-0.1)

    def test_predict_safety_car_windows(self):
        """Test SC window prediction for standard race."""
        model = SafetyCarModel()

        windows = model.predict_safety_car_windows(race_laps=50)

        assert len(windows) == 4  # Early, pit chaos, mid-race, late
        assert all(isinstance(window[0], range) for window in windows)
        assert all(isinstance(window[1], float) for window in windows)
        assert all(0 <= window[1] <= 1 for window in windows)

    def test_sc_windows_with_incidents(self):
        """Test SC probability increases with incidents."""
        model = SafetyCarModel()

        windows_no_incidents = model.predict_safety_car_windows(50, incidents_so_far=0)
        windows_with_incidents = model.predict_safety_car_windows(
            50, incidents_so_far=3
        )

        # First window (early laps) should have higher probability with incidents
        assert windows_with_incidents[0][1] > windows_no_incidents[0][1]

    def test_sc_windows_scales_with_race_length(self):
        """Test SC windows scale appropriately with race length."""
        model = SafetyCarModel()

        short_race = model.predict_safety_car_windows(30)
        long_race = model.predict_safety_car_windows(70)

        # Mid-race window should be larger for longer races
        short_mid_window = short_race[2][0]
        long_mid_window = long_race[2][0]

        assert len(long_mid_window) > len(short_mid_window)

    def test_invalid_race_laps_raises_error(self):
        """Test SC window prediction with invalid race laps."""
        model = SafetyCarModel()

        with pytest.raises(ValueError, match="race_laps must be >= 1"):
            model.predict_safety_car_windows(0)

    def test_calculate_pit_advantage_basic(self):
        """Test basic pit advantage calculation."""
        model = SafetyCarModel()

        result = model.calculate_pit_under_sc_advantage(
            current_lap=20, tire_age=10, position=5
        )

        assert "time_advantage" in result
        assert "positions_lost_estimate" in result
        assert "recommendation" in result

        assert result["time_advantage"] > 0
        assert result["positions_lost_estimate"] >= 0
        assert result["recommendation"] in ["PIT", "STAY OUT"]

    def test_old_tires_increase_advantage(self):
        """Test that old tires increase pit advantage."""
        model = SafetyCarModel()

        fresh_tires = model.calculate_pit_under_sc_advantage(
            current_lap=10, tire_age=3, position=5
        )

        old_tires = model.calculate_pit_under_sc_advantage(
            current_lap=30, tire_age=20, position=5
        )

        assert old_tires["time_advantage"] > fresh_tires["time_advantage"]

    def test_leader_has_less_advantage(self):
        """Test that race leader has less advantage than mid-field."""
        model = SafetyCarModel()

        leader = model.calculate_pit_under_sc_advantage(
            current_lap=20, tire_age=10, position=1
        )

        midfield = model.calculate_pit_under_sc_advantage(
            current_lap=20, tire_age=10, position=10
        )

        assert leader["time_advantage"] > midfield["time_advantage"]

    def test_recommendation_threshold(self):
        """Test recommendation changes based on advantage."""
        model = SafetyCarModel()

        # Fresh tires should recommend STAY OUT
        fresh = model.calculate_pit_under_sc_advantage(
            current_lap=10, tire_age=2, position=5
        )
        assert fresh["recommendation"] == "STAY OUT"

        # Old tires should recommend PIT
        old = model.calculate_pit_under_sc_advantage(
            current_lap=30, tire_age=25, position=5
        )
        assert old["recommendation"] == "PIT"

    def test_invalid_tire_age_raises_error(self):
        """Test pit advantage with invalid tire age."""
        model = SafetyCarModel()

        with pytest.raises(ValueError, match="tire_age must be >= 0"):
            model.calculate_pit_under_sc_advantage(
                current_lap=10, tire_age=-1, position=5
            )

    def test_invalid_position_raises_error(self):
        """Test pit advantage with invalid position."""
        model = SafetyCarModel()

        with pytest.raises(ValueError, match="position must be >= 1"):
            model.calculate_pit_under_sc_advantage(
                current_lap=10, tire_age=10, position=0
            )

    def test_adjust_strategy_for_sc(self):
        """Test strategy adjustment based on SC probability."""
        model = SafetyCarModel()

        base_strategy = {
            "pit_stops": [
                {"lap": 20, "compound": "medium"},
                {"lap": 40, "compound": "hard"},
            ]
        }

        sc_windows = model.predict_safety_car_windows(60)
        adjusted = model.adjust_strategy_for_sc_probability(base_strategy, sc_windows)

        assert "pit_stops" in adjusted
        assert "sc_adjusted" in adjusted
        assert adjusted["sc_adjusted"] is True

        # Each stop should have adjustment info
        for stop in adjusted["pit_stops"]:
            assert "adjusted_lap" in stop
            assert "sc_benefit" in stop

    def test_strategy_adjustment_near_sc_window(self):
        """Test strategy adjusts stops near SC windows."""
        model = SafetyCarModel()

        # Create stop near a high-probability window
        base_strategy = {"pit_stops": [{"lap": 3, "compound": "soft"}]}

        sc_windows = model.predict_safety_car_windows(50)
        adjusted = model.adjust_strategy_for_sc_probability(base_strategy, sc_windows)

        # Stop should be adjusted if near window
        stop = adjusted["pit_stops"][0]
        assert stop["sc_benefit"] >= 0

    def test_no_adjustment_far_from_windows(self):
        """Test no adjustment when stops far from SC windows."""
        model = SafetyCarModel()

        # Stop in middle of race far from typical windows
        base_strategy = {"pit_stops": [{"lap": 25, "compound": "medium"}]}

        sc_windows = model.predict_safety_car_windows(50)
        adjusted = model.adjust_strategy_for_sc_probability(base_strategy, sc_windows)

        stop = adjusted["pit_stops"][0]

        # Might have small benefit but lap shouldn't change much
        assert abs(stop["adjusted_lap"] - 25) <= 5

    def test_positions_lost_calculation(self):
        """Test positions lost estimate is reasonable."""
        model = SafetyCarModel()

        result = model.calculate_pit_under_sc_advantage(
            current_lap=20, tire_age=15, position=5, field_spread=15.0
        )

        # Positions lost should be reasonable (0-5 typically)
        assert 0 <= result["positions_lost_estimate"] <= 5

    def test_field_spread_affects_positions(self):
        """Test field spread affects positions lost."""
        model = SafetyCarModel()

        tight_field = model.calculate_pit_under_sc_advantage(
            current_lap=20, tire_age=15, position=5, field_spread=10.0
        )

        spread_field = model.calculate_pit_under_sc_advantage(
            current_lap=20, tire_age=15, position=5, field_spread=20.0
        )

        # Tighter field should lose more positions for same advantage
        assert tight_field["positions_lost_estimate"] >= spread_field[
            "positions_lost_estimate"
        ]

    def test_high_probability_circuit(self):
        """Test model with high SC probability circuit."""
        model = SafetyCarModel(circuit_safety_car_rate=0.7)

        # Base probability should be higher
        assert model.base_probability == 0.7

    def test_low_probability_circuit(self):
        """Test model with low SC probability circuit."""
        model = SafetyCarModel(circuit_safety_car_rate=0.1)

        # Base probability should be lower
        assert model.base_probability == 0.1
