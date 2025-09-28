"""Tests for data models."""

from datetime import date

import pytest
from pydantic import ValidationError

from f1_predict.data.models import (
    Circuit,
    Constructor,
    Driver,
    Location,
    QualifyingResult,
    Race,
    Result,
    Session,
)


class TestLocation:
    """Tests for Location model."""

    def test_location_valid(self):
        """Test valid location creation."""
        location = Location(lat=51.5074, long=-0.1278, locality="London", country="UK")
        assert location.lat == 51.5074
        assert location.long == -0.1278
        assert location.locality == "London"
        assert location.country == "UK"

    def test_location_invalid_lat(self):
        """Test location with invalid latitude."""
        with pytest.raises(ValidationError):
            Location(lat="invalid", long=-0.1278, locality="London", country="UK")


class TestCircuit:
    """Tests for Circuit model."""

    def test_circuit_valid(self):
        """Test valid circuit creation."""
        location = Location(lat=51.5074, long=-0.1278, locality="London", country="UK")
        circuit = Circuit(
            circuitId="silverstone",
            url="http://example.com",
            circuitName="Silverstone Circuit",
            location=location,
        )
        assert circuit.circuit_id == "silverstone"
        assert circuit.circuit_name == "Silverstone Circuit"
        assert circuit.location == location

    def test_circuit_alias_handling(self):
        """Test circuit with field aliases."""
        circuit_data = {
            "circuitId": "monaco",
            "url": "http://example.com",
            "circuitName": "Circuit de Monaco",
            "location": {
                "lat": 43.7347,
                "long": 7.4206,
                "locality": "Monte Carlo",
                "country": "Monaco",
            },
        }
        circuit = Circuit.model_validate(circuit_data)
        assert circuit.circuit_id == "monaco"
        assert circuit.circuit_name == "Circuit de Monaco"


class TestDriver:
    """Tests for Driver model."""

    def test_driver_valid(self):
        """Test valid driver creation."""
        driver = Driver(
            driverId="hamilton",
            permanentNumber=44,
            code="HAM",
            url="http://example.com",
            givenName="Lewis",
            familyName="Hamilton",
            dateOfBirth="1985-01-07",
            nationality="British",
        )
        assert driver.driver_id == "hamilton"
        assert driver.permanent_number == 44
        assert driver.code == "HAM"
        assert driver.given_name == "Lewis"
        assert driver.family_name == "Hamilton"
        assert driver.date_of_birth == date(1985, 1, 7)
        assert driver.nationality == "British"

    def test_driver_with_string_date(self):
        """Test driver with string date of birth."""
        driver_data = {
            "driverId": "verstappen",
            "url": "http://example.com",
            "givenName": "Max",
            "familyName": "Verstappen",
            "dateOfBirth": "1997-09-30",
            "nationality": "Dutch",
        }
        driver = Driver.model_validate(driver_data)
        assert driver.date_of_birth == date(1997, 9, 30)

    def test_driver_without_optional_fields(self):
        """Test driver without optional fields."""
        driver_data = {
            "driverId": "leclerc",
            "url": "http://example.com",
            "givenName": "Charles",
            "familyName": "Leclerc",
            "dateOfBirth": "1997-10-16",
            "nationality": "Monégasque",
        }
        driver = Driver.model_validate(driver_data)
        assert driver.permanent_number is None
        assert driver.code is None


class TestConstructor:
    """Tests for Constructor model."""

    def test_constructor_valid(self):
        """Test valid constructor creation."""
        constructor = Constructor(
            constructorId="mercedes",
            url="http://example.com",
            name="Mercedes",
            nationality="German",
        )
        assert constructor.constructor_id == "mercedes"
        assert constructor.name == "Mercedes"
        assert constructor.nationality == "German"


class TestSession:
    """Tests for Session model."""

    def test_session_with_date_and_time(self):
        """Test session with date and time."""
        session = Session(date="2023-07-07", time="14:00:00Z")
        assert session.date == date(2023, 7, 7)
        assert session.time is not None

    def test_session_with_date_only(self):
        """Test session with date only."""
        session = Session(date="2023-07-07")
        assert session.date == date(2023, 7, 7)
        assert session.time is None

    def test_session_invalid_time_format(self):
        """Test session with invalid time format."""
        session = Session(date="2023-07-07", time="invalid-time")
        # Invalid time should be set to None
        assert session.time is None


class TestRace:
    """Tests for Race model."""

    def test_race_valid(self):
        """Test valid race creation."""
        circuit_data = {
            "circuitId": "silverstone",
            "url": "http://example.com",
            "circuitName": "Silverstone Circuit",
            "location": {
                "lat": 52.0786,
                "long": -1.0169,
                "locality": "Silverstone",
                "country": "UK",
            },
        }

        race_data = {
            "season": "2023",
            "round": "10",
            "url": "http://example.com",
            "raceName": "British Grand Prix",
            "circuit": circuit_data,
            "date": "2023-07-09",
            "time": "14:00:00Z",
        }

        race = Race.model_validate(race_data)
        assert race.season == "2023"
        assert race.round == "10"
        assert race.race_name == "British Grand Prix"
        assert race.date == date(2023, 7, 9)
        assert race.time is not None
        assert race.circuit.circuit_id == "silverstone"

    def test_race_with_sessions(self):
        """Test race with practice and qualifying sessions."""
        circuit_data = {
            "circuitId": "monaco",
            "url": "http://example.com",
            "circuitName": "Circuit de Monaco",
            "location": {
                "lat": 43.7347,
                "long": 7.4206,
                "locality": "Monte Carlo",
                "country": "Monaco",
            },
        }

        race_data = {
            "season": "2023",
            "round": "6",
            "url": "http://example.com",
            "raceName": "Monaco Grand Prix",
            "circuit": circuit_data,
            "date": "2023-05-28",
            "FirstPractice": {"date": "2023-05-26", "time": "13:30:00Z"},
            "Qualifying": {"date": "2023-05-27", "time": "16:00:00Z"},
        }

        race = Race.model_validate(race_data)
        assert race.first_practice is not None
        assert race.first_practice.date == date(2023, 5, 26)
        assert race.qualifying is not None
        assert race.qualifying.date == date(2023, 5, 27)
        assert race.second_practice is None
        assert race.third_practice is None


class TestResult:
    """Tests for Result model."""

    def test_result_valid(self):
        """Test valid result creation."""
        driver_data = {
            "driverId": "hamilton",
            "url": "http://example.com",
            "givenName": "Lewis",
            "familyName": "Hamilton",
            "dateOfBirth": "1985-01-07",
            "nationality": "British",
        }

        constructor_data = {
            "constructorId": "mercedes",
            "url": "http://example.com",
            "name": "Mercedes",
            "nationality": "German",
        }

        result_data = {
            "number": 44,
            "position": 1,
            "positionText": "1",
            "points": 25.0,
            "driver": driver_data,
            "constructor": constructor_data,
            "grid": 1,
            "laps": 70,
            "status": "Finished",
        }

        result = Result.model_validate(result_data)
        assert result.number == 44
        assert result.position == 1
        assert result.position_text == "1"
        assert result.points == 25.0
        assert result.driver.driver_id == "hamilton"
        assert result.constructor.constructor_id == "mercedes"
        assert result.grid == 1
        assert result.laps == 70
        assert result.status == "Finished"

    def test_result_with_time(self):
        """Test result with finish time."""
        driver_data = {
            "driverId": "verstappen",
            "url": "http://example.com",
            "givenName": "Max",
            "familyName": "Verstappen",
            "dateOfBirth": "1997-09-30",
            "nationality": "Dutch",
        }

        constructor_data = {
            "constructorId": "red_bull",
            "url": "http://example.com",
            "name": "Red Bull",
            "nationality": "Austrian",
        }

        result_data = {
            "number": 1,
            "position": 1,
            "positionText": "1",
            "points": 25.0,
            "driver": driver_data,
            "constructor": constructor_data,
            "grid": 1,
            "laps": 70,
            "status": "Finished",
            "time": {"millis": 5434567, "time": "1:30:34.567"},
        }

        result = Result.model_validate(result_data)
        assert result.time is not None
        assert result.time.millis == 5434567
        assert result.time.time == "1:30:34.567"


class TestQualifyingResult:
    """Tests for QualifyingResult model."""

    def test_qualifying_result_valid(self):
        """Test valid qualifying result creation."""
        driver_data = {
            "driverId": "hamilton",
            "url": "http://example.com",
            "givenName": "Lewis",
            "familyName": "Hamilton",
            "dateOfBirth": "1985-01-07",
            "nationality": "British",
        }

        constructor_data = {
            "constructorId": "mercedes",
            "url": "http://example.com",
            "name": "Mercedes",
            "nationality": "German",
        }

        qualifying_data = {
            "number": 44,
            "position": 1,
            "driver": driver_data,
            "constructor": constructor_data,
            "Q1": "1:29.123",
            "Q2": "1:28.456",
            "Q3": "1:27.789",
        }

        qualifying_result = QualifyingResult.model_validate(qualifying_data)
        assert qualifying_result.number == 44
        assert qualifying_result.position == 1
        assert qualifying_result.driver.driver_id == "hamilton"
        assert qualifying_result.constructor.constructor_id == "mercedes"
        assert qualifying_result.q1 == "1:29.123"
        assert qualifying_result.q2 == "1:28.456"
        assert qualifying_result.q3 == "1:27.789"

    def test_qualifying_result_partial_times(self):
        """Test qualifying result with partial times."""
        driver_data = {
            "driverId": "norris",
            "url": "http://example.com",
            "givenName": "Lando",
            "familyName": "Norris",
            "dateOfBirth": "1999-11-13",
            "nationality": "British",
        }

        constructor_data = {
            "constructorId": "mclaren",
            "url": "http://example.com",
            "name": "McLaren",
            "nationality": "British",
        }

        qualifying_data = {
            "number": 4,
            "position": 11,
            "driver": driver_data,
            "constructor": constructor_data,
            "Q1": "1:30.456",
            # Q2 and Q3 missing (eliminated in Q1)
        }

        qualifying_result = QualifyingResult.model_validate(qualifying_data)
        assert qualifying_result.q1 == "1:30.456"
        assert qualifying_result.q2 is None
        assert qualifying_result.q3 is None


class TestModelValidation:
    """Tests for overall model validation."""

    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        with pytest.raises(ValidationError):
            Driver(
                # Missing required fields
                url="http://example.com",
                givenName="Test",
                nationality="Test",
            )

    def test_invalid_field_types(self):
        """Test validation with invalid field types."""
        with pytest.raises(ValidationError):
            Location(
                lat="invalid_number",  # Should be float
                long=0.0,
                locality="Test",
                country="Test",
            )

    def test_date_parsing_error(self):
        """Test date parsing with invalid format."""
        with pytest.raises(ValidationError):
            Driver(
                driverId="test",
                url="http://example.com",
                givenName="Test",
                familyName="Driver",
                dateOfBirth="invalid-date",  # Invalid format
                nationality="Test",
            )
