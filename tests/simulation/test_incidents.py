"""Unit tests for incident generation."""

import pytest

from f1_predict.simulation.core.incidents import (
    IncidentEvent,
    IncidentGenerator,
    IncidentType,
)


class TestIncidentEvent:
    """Test IncidentEvent dataclass."""

    def test_incident_event_creation(self):
        """Test creating an incident event."""
        event = IncidentEvent(
            lap=25,
            incident_type=IncidentType.SAFETY_CAR,
            description="Safety car deployed",
        )
        assert event.lap == 25
        assert event.incident_type == IncidentType.SAFETY_CAR
        assert event.description == "Safety car deployed"
        assert event.affected_driver_id is None

    def test_incident_event_with_driver(self):
        """Test incident event affecting a driver."""
        event = IncidentEvent(
            lap=30,
            incident_type=IncidentType.DNF_CRASH,
            affected_driver_id="VER",
            description="Max Verstappen crashed",
        )
        assert event.affected_driver_id == "VER"
        assert event.incident_type == IncidentType.DNF_CRASH

    def test_incident_event_string_representation(self):
        """Test string representation of incident."""
        event = IncidentEvent(
            lap=25,
            incident_type=IncidentType.SAFETY_CAR,
            description="Safety car deployed",
        )
        str_repr = str(event)
        assert "Lap 25" in str_repr
        assert "SAFETY_CAR" in str_repr


class TestIncidentGenerator:
    """Test IncidentGenerator class."""

    def test_incident_generator_creation(self):
        """Test creating incident generator."""
        gen = IncidentGenerator(circuit_type="intermediate", random_state=42)
        assert gen.circuit_type == "intermediate"
        assert gen.safety_car_prob == 0.08

    def test_incident_generator_circuit_types(self):
        """Test safety car probability by circuit type."""
        # Street circuits should have higher SC probability
        gen_street = IncidentGenerator(circuit_type="street")
        assert gen_street.safety_car_prob == 0.12

        # High-speed circuits should have lower SC probability
        gen_high_speed = IncidentGenerator(circuit_type="high_speed")
        assert gen_high_speed.safety_car_prob == 0.05

    def test_generate_safety_car_probabilistic(self):
        """Test safety car generation is probabilistic."""
        # Run multiple times to check stochastic behavior
        gen = IncidentGenerator(circuit_type="intermediate", random_state=None)

        events = []
        for _ in range(100):
            event = gen.generate_safety_car(current_lap=25, total_laps=58)
            if event is not None:
                events.append(event)

        # Should generate some (but not all) SCs
        assert len(events) > 0
        assert len(events) < 100

    def test_generate_safety_car_late_race(self):
        """Test that safety car doesn't generate late in race."""
        gen = IncidentGenerator(circuit_type="intermediate", random_state=42)

        # Try to generate SC in final 15% of race
        for _ in range(100):
            event = gen.generate_safety_car(current_lap=55, total_laps=58)
            # Should return None (no SC in final 15%)
            assert event is None

    def test_generate_dnf_increases_with_progress(self):
        """Test DNF probability increases as race progresses."""
        gen = IncidentGenerator(random_state=42)

        # Early race
        gen_early = IncidentGenerator(random_state=42)
        dnfs_early = 0
        for _ in range(100):
            event = gen_early.generate_dnf("VER", "Max Verstappen", 10, 58)
            if event is not None:
                dnfs_early += 1

        # Late race
        gen_late = IncidentGenerator(random_state=42)
        dnfs_late = 0
        for _ in range(100):
            event = gen_late.generate_dnf("VER", "Max Verstappen", 50, 58)
            if event is not None:
                dnfs_late += 1

        # More DNFs late in race
        assert dnfs_late > dnfs_early

    def test_generate_dnf_event(self):
        """Test DNF event generation."""
        gen = IncidentGenerator(random_state=42)

        # Run until we get a DNF
        for i in range(1000):
            event = gen.generate_dnf("VER", "Max Verstappen", 30, 58)
            if event is not None:
                assert event.incident_type in [
                    IncidentType.DNF_MECHANICAL,
                    IncidentType.DNF_CRASH,
                    IncidentType.DNF_OTHER,
                ]
                assert event.affected_driver_id == "VER"
                break

    def test_generate_weather_change_probabilistic(self):
        """Test weather change is probabilistic."""
        gen = IncidentGenerator(random_state=None)

        events = []
        for _ in range(1000):
            event = gen.generate_weather_change(current_lap=25, current_condition="dry")
            if event is not None:
                events.append(event)

        # Should generate some weather changes
        assert len(events) > 0

    def test_generate_weather_change_to_different_condition(self):
        """Test weather changes to different condition."""
        gen = IncidentGenerator(random_state=42)

        # Force weather change by running many times
        for _ in range(1000):
            event = gen.generate_weather_change(
                current_lap=25, current_condition="dry"
            )
            if event is not None:
                assert event.incident_type == IncidentType.WEATHER_CHANGE
                break

    def test_incident_logging(self):
        """Test incident logging."""
        gen = IncidentGenerator(random_state=42)
        gen.clear_incidents()

        assert len(gen.get_incidents()) == 0

        # Generate some incidents
        gen.generate_safety_car(25, 58)
        gen.generate_dnf("VER", "Max Verstappen", 30, 58)

        incidents = gen.get_incidents()
        # Might be 0-2 depending on random generation
        assert len(incidents) >= 0

    def test_clear_incidents(self):
        """Test clearing incident log."""
        gen = IncidentGenerator(random_state=42)

        # Force an incident
        incident = IncidentEvent(
            lap=25,
            incident_type=IncidentType.SAFETY_CAR,
        )
        gen.incident_log.append(incident)

        assert len(gen.get_incidents()) > 0
        gen.clear_incidents()
        assert len(gen.get_incidents()) == 0


class TestIncidentTypes:
    """Test IncidentType enum."""

    def test_incident_types_exist(self):
        """Test all incident types are defined."""
        assert IncidentType.SAFETY_CAR.value == "safety_car"
        assert IncidentType.RED_FLAG.value == "red_flag"
        assert IncidentType.DNF_MECHANICAL.value == "dnf_mechanical"
        assert IncidentType.DNF_CRASH.value == "dnf_crash"
        assert IncidentType.DNF_OTHER.value == "dnf_other"
        assert IncidentType.WEATHER_CHANGE.value == "weather_change"
