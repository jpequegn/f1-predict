"""Unit tests for DriverState class."""

import pytest

from f1_predict.simulation.core.driver_state import (
    DriverState,
    DriverStatus,
    TireCompound,
)


class TestDriverStateCreation:
    """Test DriverState initialization and validation."""

    def test_driver_state_creation(self):
        """Test basic driver state creation."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
            expected_lap_time=81.5,
        )
        assert driver.driver_id == "VER"
        assert driver.driver_name == "Max Verstappen"
        assert driver.position == 1
        assert driver.lap == 1
        assert driver.status == DriverStatus.RUNNING
        assert driver.fuel_level == 100.0
        assert driver.tire_compound == TireCompound.SOFT

    def test_driver_state_with_custom_values(self):
        """Test driver state with custom initial values."""
        driver = DriverState(
            driver_id="HAM",
            driver_name="Lewis Hamilton",
            position=5,
            lap=10,
            fuel_level=75.0,
            tire_compound=TireCompound.MEDIUM,
        )
        assert driver.position == 5
        assert driver.lap == 10
        assert driver.fuel_level == 75.0
        assert driver.tire_compound == TireCompound.MEDIUM

    def test_invalid_fuel_level(self):
        """Test that invalid fuel levels raise ValueError."""
        with pytest.raises(ValueError):
            DriverState(
                driver_id="VER",
                driver_name="Max Verstappen",
                fuel_level=150.0,  # Invalid: > 100
            )

        with pytest.raises(ValueError):
            DriverState(
                driver_id="VER",
                driver_name="Max Verstappen",
                fuel_level=-10.0,  # Invalid: < 0
            )

    def test_invalid_position(self):
        """Test that invalid positions raise ValueError."""
        with pytest.raises(ValueError):
            DriverState(
                driver_id="VER",
                driver_name="Max Verstappen",
                position=0,  # Invalid: must be >= 1
            )

    def test_invalid_lap(self):
        """Test that invalid lap numbers raise ValueError."""
        with pytest.raises(ValueError):
            DriverState(
                driver_id="VER",
                driver_name="Max Verstappen",
                lap=0,  # Invalid: must be >= 1
            )


class TestDriverStateUpdates:
    """Test driver state update methods."""

    def test_update_position(self):
        """Test position updates."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
            position=1,
        )
        driver.update_position(2)
        assert driver.position == 2

    def test_update_position_invalid(self):
        """Test that invalid positions raise ValueError."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
        )
        with pytest.raises(ValueError):
            driver.update_position(0)

    def test_update_gaps(self):
        """Test gap updates."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
        )
        driver.update_gaps(gap_to_leader=5.0, gap_to_previous=2.0)
        assert driver.gap_to_leader == 5.0
        assert driver.gap_to_previous == 2.0

    def test_update_gaps_negative_values(self):
        """Test that negative gaps are corrected to 0."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
        )
        driver.update_gaps(gap_to_leader=-5.0, gap_to_previous=-2.0)
        assert driver.gap_to_leader == 0.0
        assert driver.gap_to_previous == 0.0

    def test_consume_fuel(self):
        """Test fuel consumption."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
            fuel_level=100.0,
        )
        driver.consume_fuel(1.5)
        assert driver.fuel_level == 98.5

    def test_consume_fuel_to_empty(self):
        """Test fuel consumption until empty."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
            fuel_level=0.5,
        )
        driver.consume_fuel(1.0)
        assert driver.fuel_level == 0.0

    def test_pit_stop(self):
        """Test pit stop execution."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
            tire_compound=TireCompound.SOFT,
            laps_on_tire=20,
            pit_stop_count=0,
        )
        driver.pit_stop(TireCompound.MEDIUM, 25.0)

        assert driver.status == DriverStatus.PIT_STOP
        assert driver.tire_compound == TireCompound.MEDIUM
        assert driver.laps_on_tire == 0
        assert driver.pit_stop_count == 1
        assert driver.fuel_level == 100.0
        assert 25.0 in driver.pit_stop_durations

    def test_resume_racing(self):
        """Test resuming racing after pit stop."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
            status=DriverStatus.PIT_STOP,
        )
        driver.resume_racing()
        assert driver.status == DriverStatus.RUNNING

    def test_dnf(self):
        """Test DNF (did not finish) marking."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
        )
        driver.dnf("engine failure")
        assert driver.status == DriverStatus.DNF
        assert driver.dnf_reason == "engine failure"

    def test_finish_race(self):
        """Test finishing the race."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
        )
        driver.finish_race()
        assert driver.status == DriverStatus.FINISHED

    def test_complete_lap(self):
        """Test completing a lap."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
            lap=1,
        )
        driver.complete_lap(81.5)
        assert driver.lap == 2
        assert driver.laps_on_tire == 1
        assert driver.best_lap_time == 81.5

    def test_complete_lap_best_time(self):
        """Test that best lap time is tracked."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
        )
        driver.complete_lap(82.0)
        assert driver.best_lap_time == 82.0

        driver.complete_lap(81.5)  # Faster
        assert driver.best_lap_time == 81.5

        driver.complete_lap(82.0)  # Slower
        assert driver.best_lap_time == 81.5  # Still 81.5

    def test_complete_lap_increments_tires(self):
        """Test that laps on tire increments."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
        )
        driver.complete_lap(81.5)
        assert driver.laps_on_tire == 1

        driver.complete_lap(81.6)
        assert driver.laps_on_tire == 2


class TestDriverStateProperties:
    """Test driver state property methods."""

    def test_is_active_running(self):
        """Test is_active property when running."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
            status=DriverStatus.RUNNING,
        )
        assert driver.is_active is True

    def test_is_active_pit_stop(self):
        """Test is_active property during pit stop."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
            status=DriverStatus.PIT_STOP,
        )
        assert driver.is_active is False

    def test_is_active_dnf(self):
        """Test is_active property when DNF."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
            status=DriverStatus.DNF,
        )
        assert driver.is_active is False

    def test_is_active_finished(self):
        """Test is_active property when finished."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
            status=DriverStatus.FINISHED,
        )
        assert driver.is_active is False

    def test_is_finished(self):
        """Test is_finished property."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
            status=DriverStatus.FINISHED,
        )
        assert driver.is_finished is True

    def test_is_dnf(self):
        """Test is_dnf property."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
            status=DriverStatus.DNF,
        )
        assert driver.is_dnf is True


class TestDriverStateCopy:
    """Test driver state deep copy."""

    def test_copy_creates_independent_copy(self):
        """Test that copy creates an independent instance."""
        driver1 = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
            position=1,
            lap=5,
        )
        driver2 = driver1.copy()

        # Verify copy has same values
        assert driver2.driver_id == driver1.driver_id
        assert driver2.position == driver1.position
        assert driver2.lap == driver1.lap

        # Modify copy and verify original unchanged
        driver2.update_position(2)
        driver2.complete_lap(81.5)

        assert driver1.position == 1
        assert driver1.lap == 5
        assert driver2.position == 2
        assert driver2.lap == 6

    def test_copy_pit_stop_durations(self):
        """Test that pit stop durations list is copied."""
        driver1 = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
        )
        driver1.pit_stop(TireCompound.MEDIUM, 25.0)

        driver2 = driver1.copy()
        driver2.pit_stop(TireCompound.HARD, 24.5)

        assert len(driver1.pit_stop_durations) == 1
        assert len(driver2.pit_stop_durations) == 2


class TestTireCompound:
    """Test TireCompound enum."""

    def test_tire_compounds_exist(self):
        """Test that all tire compounds are defined."""
        assert TireCompound.SOFT.value == "SOFT"
        assert TireCompound.MEDIUM.value == "MEDIUM"
        assert TireCompound.HARD.value == "HARD"
        assert TireCompound.INTERMEDIATE.value == "INTERMEDIATE"
        assert TireCompound.WET.value == "WET"


class TestDriverStatus:
    """Test DriverStatus enum."""

    def test_driver_statuses_exist(self):
        """Test that all driver statuses are defined."""
        assert DriverStatus.RUNNING.value == "running"
        assert DriverStatus.PIT_STOP.value == "pit_stop"
        assert DriverStatus.DNF.value == "dnf"
        assert DriverStatus.FINISHED.value == "finished"
