"""Tests for the Ergast API client."""

from unittest.mock import Mock, patch

import pytest

from f1_predict.api.ergast import ErgastAPIClient
from f1_predict.data.models import (
    Circuit,
    Constructor,
    ConstructorStanding,
    Driver,
    DriverStanding,
    QualifyingResult,
    Race,
    Result,
    Season,
)


class TestErgastAPIClient:
    """Tests for the ErgastAPIClient class."""

    @pytest.fixture
    def client(self):
        """Create a test Ergast API client."""
        return ErgastAPIClient()

    @pytest.fixture
    def mock_seasons_response(self):
        """Mock response for seasons endpoint."""
        return {
            "MRData": {
                "xmlns": "http://ergast.com/mrd/1.4",
                "series": "f1",
                "url": "http://ergast.com/api/f1/seasons.json",
                "limit": "30",
                "offset": "0",
                "total": "74",
                "SeasonTable": {
                    "Seasons": [
                        {
                            "season": "2023",
                            "url": "http://en.wikipedia.org/wiki/2023_Formula_One_World_Championship"
                        },
                        {
                            "season": "2022",
                            "url": "http://en.wikipedia.org/wiki/2022_Formula_One_World_Championship"
                        }
                    ]
                }
            }
        }

    @pytest.fixture
    def mock_races_response(self):
        """Mock response for races endpoint."""
        return {
            "MRData": {
                "xmlns": "http://ergast.com/mrd/1.4",
                "series": "f1",
                "url": "http://ergast.com/api/f1/2023/races.json",
                "limit": "30",
                "offset": "0",
                "total": "22",
                "RaceTable": {
                    "season": "2023",
                    "Races": [
                        {
                            "season": "2023",
                            "round": "1",
                            "url": "http://en.wikipedia.org/wiki/2023_Bahrain_Grand_Prix",
                            "raceName": "Bahrain Grand Prix",
                            "circuit": {
                                "circuitId": "bahrain",
                                "url": "http://en.wikipedia.org/wiki/Bahrain_International_Circuit",
                                "circuitName": "Bahrain International Circuit",
                                "location": {
                                    "lat": "26.0325",
                                    "long": "50.5106",
                                    "locality": "Sakhir",
                                    "country": "Bahrain"
                                }
                            },
                            "date": "2023-03-05",
                            "time": "15:00:00Z"
                        }
                    ]
                }
            }
        }

    @pytest.fixture
    def mock_drivers_response(self):
        """Mock response for drivers endpoint."""
        return {
            "MRData": {
                "xmlns": "http://ergast.com/mrd/1.4",
                "series": "f1",
                "url": "http://ergast.com/api/f1/drivers.json",
                "limit": "30",
                "offset": "0",
                "total": "857",
                "DriverTable": {
                    "Drivers": [
                        {
                            "driverId": "hamilton",
                            "permanentNumber": "44",
                            "code": "HAM",
                            "url": "http://en.wikipedia.org/wiki/Lewis_Hamilton",
                            "givenName": "Lewis",
                            "familyName": "Hamilton",
                            "dateOfBirth": "1985-01-07",
                            "nationality": "British"
                        },
                        {
                            "driverId": "verstappen",
                            "permanentNumber": "1",
                            "code": "VER",
                            "url": "http://en.wikipedia.org/wiki/Max_Verstappen",
                            "givenName": "Max",
                            "familyName": "Verstappen",
                            "dateOfBirth": "1997-09-30",
                            "nationality": "Dutch"
                        }
                    ]
                }
            }
        }

    @pytest.fixture
    def mock_results_response(self):
        """Mock response for race results endpoint."""
        return {
            "MRData": {
                "xmlns": "http://ergast.com/mrd/1.4",
                "series": "f1",
                "url": "http://ergast.com/api/f1/2023/1/results.json",
                "limit": "30",
                "offset": "0",
                "total": "20",
                "RaceTable": {
                    "season": "2023",
                    "round": "1",
                    "Races": [
                        {
                            "season": "2023",
                            "round": "1",
                            "url": "http://en.wikipedia.org/wiki/2023_Bahrain_Grand_Prix",
                            "raceName": "Bahrain Grand Prix",
                            "circuit": {
                                "circuitId": "bahrain",
                                "url": "http://en.wikipedia.org/wiki/Bahrain_International_Circuit",
                                "circuitName": "Bahrain International Circuit",
                                "location": {
                                    "lat": "26.0325",
                                    "long": "50.5106",
                                    "locality": "Sakhir",
                                    "country": "Bahrain"
                                }
                            },
                            "date": "2023-03-05",
                            "time": "15:00:00Z",
                            "Results": [
                                {
                                    "number": "1",
                                    "position": "1",
                                    "positionText": "1",
                                    "points": "25",
                                    "driver": {
                                        "driverId": "verstappen",
                                        "permanentNumber": "1",
                                        "code": "VER",
                                        "url": "http://en.wikipedia.org/wiki/Max_Verstappen",
                                        "givenName": "Max",
                                        "familyName": "Verstappen",
                                        "dateOfBirth": "1997-09-30",
                                        "nationality": "Dutch"
                                    },
                                    "constructor": {
                                        "constructorId": "red_bull",
                                        "url": "http://en.wikipedia.org/wiki/Red_Bull_Racing",
                                        "name": "Red Bull",
                                        "nationality": "Austrian"
                                    },
                                    "grid": "1",
                                    "laps": "57",
                                    "status": "Finished",
                                    "time": {
                                        "millis": "5434567",
                                        "time": "1:30:34.567"
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
        }

    def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.base_url == "https://ergast.com/api/f1"
        assert client.DEFAULT_BASE_URL == "https://ergast.com/api/f1"

    def test_client_initialization_with_custom_url(self):
        """Test client initialization with custom URL."""
        custom_url = "https://custom.api.com/f1"
        client = ErgastAPIClient(base_url=custom_url)
        assert client.base_url == custom_url

    @patch.object(ErgastAPIClient, 'get')
    def test_get_seasons(self, mock_get, client, mock_seasons_response):
        """Test getting seasons."""
        mock_get.return_value = mock_seasons_response

        seasons = client.get_seasons()

        assert len(seasons) == 2
        assert isinstance(seasons[0], Season)
        assert seasons[0].season == "2023"
        assert seasons[1].season == "2022"

        mock_get.assert_called_once_with("seasons.json", params={})

    @patch.object(ErgastAPIClient, 'get')
    def test_get_seasons_with_pagination(self, mock_get, client, mock_seasons_response):
        """Test getting seasons with pagination parameters."""
        mock_get.return_value = mock_seasons_response

        seasons = client.get_seasons(limit=10, offset=5)

        mock_get.assert_called_once_with("seasons.json", params={"limit": 10, "offset": 5})

    @patch.object(ErgastAPIClient, 'get')
    def test_get_races(self, mock_get, client, mock_races_response):
        """Test getting races."""
        mock_get.return_value = mock_races_response

        races = client.get_races(season=2023)

        assert len(races) == 1
        assert isinstance(races[0], Race)
        assert races[0].season == "2023"
        assert races[0].round == "1"
        assert races[0].race_name == "Bahrain Grand Prix"

        mock_get.assert_called_once_with("2023.json", params={})

    @patch.object(ErgastAPIClient, 'get')
    def test_get_races_specific_round(self, mock_get, client, mock_races_response):
        """Test getting a specific race."""
        mock_get.return_value = mock_races_response

        races = client.get_races(season=2023, round_number=1)

        mock_get.assert_called_once_with("2023/1.json", params={})

    @patch.object(ErgastAPIClient, 'get')
    def test_get_current_season_races(self, mock_get, client, mock_races_response):
        """Test getting current season races."""
        mock_get.return_value = mock_races_response

        races = client.get_current_season_races()

        mock_get.assert_called_once_with("current.json", params={})

    @patch.object(ErgastAPIClient, 'get')
    def test_get_drivers(self, mock_get, client, mock_drivers_response):
        """Test getting drivers."""
        mock_get.return_value = mock_drivers_response

        drivers = client.get_drivers()

        assert len(drivers) == 2
        assert isinstance(drivers[0], Driver)
        assert drivers[0].driver_id == "hamilton"
        assert drivers[0].given_name == "Lewis"
        assert drivers[0].family_name == "Hamilton"
        assert drivers[1].driver_id == "verstappen"

        mock_get.assert_called_once_with("drivers.json", params={})

    @patch.object(ErgastAPIClient, 'get')
    def test_get_drivers_for_season(self, mock_get, client, mock_drivers_response):
        """Test getting drivers for a specific season."""
        mock_get.return_value = mock_drivers_response

        drivers = client.get_drivers(season=2023)

        mock_get.assert_called_once_with("2023/drivers.json", params={})

    @patch.object(ErgastAPIClient, 'get')
    def test_get_driver(self, mock_get, client):
        """Test getting a specific driver."""
        single_driver_response = {
            "MRData": {
                "DriverTable": {
                    "Drivers": [
                        {
                            "driverId": "hamilton",
                            "permanentNumber": "44",
                            "code": "HAM",
                            "url": "http://en.wikipedia.org/wiki/Lewis_Hamilton",
                            "givenName": "Lewis",
                            "familyName": "Hamilton",
                            "dateOfBirth": "1985-01-07",
                            "nationality": "British"
                        }
                    ]
                }
            }
        }
        mock_get.return_value = single_driver_response

        driver = client.get_driver("hamilton")

        assert driver is not None
        assert isinstance(driver, Driver)
        assert driver.driver_id == "hamilton"
        assert driver.given_name == "Lewis"

        mock_get.assert_called_once_with("drivers/hamilton.json")

    @patch.object(ErgastAPIClient, 'get')
    def test_get_driver_not_found(self, mock_get, client):
        """Test getting a driver that doesn't exist."""
        empty_response = {
            "MRData": {
                "DriverTable": {
                    "Drivers": []
                }
            }
        }
        mock_get.return_value = empty_response

        driver = client.get_driver("nonexistent")

        assert driver is None

    @patch.object(ErgastAPIClient, 'get')
    def test_get_race_results(self, mock_get, client, mock_results_response):
        """Test getting race results."""
        mock_get.return_value = mock_results_response

        results = client.get_race_results(2023, 1)

        assert len(results) == 1
        assert isinstance(results[0], Result)
        assert results[0].position == 1
        assert results[0].driver.driver_id == "verstappen"
        assert results[0].constructor.constructor_id == "red_bull"
        assert results[0].points == 25.0

        mock_get.assert_called_once_with("2023/1/results.json", params={})

    @patch.object(ErgastAPIClient, 'get')
    def test_get_driver_standings(self, mock_get, client):
        """Test getting driver standings."""
        standings_response = {
            "MRData": {
                "StandingsTable": {
                    "season": "2023",
                    "StandingsLists": [
                        {
                            "season": "2023",
                            "round": "22",
                            "DriverStandings": [
                                {
                                    "position": "1",
                                    "positionText": "1",
                                    "points": "575",
                                    "wins": "19",
                                    "driver": {
                                        "driverId": "verstappen",
                                        "permanentNumber": "1",
                                        "code": "VER",
                                        "url": "http://en.wikipedia.org/wiki/Max_Verstappen",
                                        "givenName": "Max",
                                        "familyName": "Verstappen",
                                        "dateOfBirth": "1997-09-30",
                                        "nationality": "Dutch"
                                    },
                                    "Constructors": [
                                        {
                                            "constructorId": "red_bull",
                                            "url": "http://en.wikipedia.org/wiki/Red_Bull_Racing",
                                            "name": "Red Bull",
                                            "nationality": "Austrian"
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }
        }
        mock_get.return_value = standings_response

        standings = client.get_driver_standings(2023)

        assert len(standings) == 1
        assert isinstance(standings[0], DriverStanding)
        assert standings[0].position == 1
        assert standings[0].points == 575.0
        assert standings[0].wins == 19
        assert standings[0].driver.driver_id == "verstappen"

        mock_get.assert_called_once_with("2023/driverStandings.json", params={})

    @patch.object(ErgastAPIClient, 'get')
    def test_get_current_driver_standings(self, mock_get, client):
        """Test getting current driver standings."""
        mock_get.return_value = {"MRData": {"StandingsTable": {"StandingsLists": []}}}

        client.get_current_driver_standings()

        mock_get.assert_called_once_with("current/driverStandings.json", params={})

    @patch.object(ErgastAPIClient, 'get')
    def test_get_last_race_results(self, mock_get, client, mock_results_response):
        """Test getting last race results."""
        mock_get.return_value = mock_results_response

        results = client.get_last_race_results(2023)

        mock_get.assert_called_once_with("2023/last/results.json", params={})

    @patch.object(ErgastAPIClient, 'get')
    def test_search_drivers(self, mock_get, client, mock_drivers_response):
        """Test searching for drivers by name."""
        mock_get.return_value = mock_drivers_response

        drivers = client.search_drivers("Lewis")

        # Should find Hamilton
        assert len(drivers) == 1
        assert drivers[0].driver_id == "hamilton"

        mock_get.assert_called_once_with("drivers.json", params={"limit": 1000})

    @patch.object(ErgastAPIClient, 'get')
    def test_search_drivers_by_code(self, mock_get, client, mock_drivers_response):
        """Test searching for drivers by code."""
        mock_get.return_value = mock_drivers_response

        drivers = client.search_drivers("VER")

        # Should find Verstappen
        assert len(drivers) == 1
        assert drivers[0].driver_id == "verstappen"

    @patch.object(ErgastAPIClient, 'get')
    def test_empty_response_handling(self, mock_get, client):
        """Test handling of empty API responses."""
        empty_response = {
            "MRData": {
                "SeasonTable": {
                    "Seasons": []
                }
            }
        }
        mock_get.return_value = empty_response

        seasons = client.get_seasons()

        assert seasons == []

    @patch.object(ErgastAPIClient, 'get')
    def test_malformed_response_handling(self, mock_get, client):
        """Test handling of malformed API responses."""
        malformed_response = {
            "invalid": "response"
        }
        mock_get.return_value = malformed_response

        seasons = client.get_seasons()

        assert seasons == []

    @patch.object(ErgastAPIClient, 'get')
    def test_get_qualifying_results(self, mock_get, client):
        """Test getting qualifying results."""
        qualifying_response = {
            "MRData": {
                "RaceTable": {
                    "season": "2023",
                    "round": "1",
                    "Races": [
                        {
                            "season": "2023",
                            "round": "1",
                            "QualifyingResults": [
                                {
                                    "number": "1",
                                    "position": "1",
                                    "driver": {
                                        "driverId": "verstappen",
                                        "permanentNumber": "1",
                                        "code": "VER",
                                        "url": "http://en.wikipedia.org/wiki/Max_Verstappen",
                                        "givenName": "Max",
                                        "familyName": "Verstappen",
                                        "dateOfBirth": "1997-09-30",
                                        "nationality": "Dutch"
                                    },
                                    "constructor": {
                                        "constructorId": "red_bull",
                                        "url": "http://en.wikipedia.org/wiki/Red_Bull_Racing",
                                        "name": "Red Bull",
                                        "nationality": "Austrian"
                                    },
                                    "Q1": "1:29.708",
                                    "Q2": "1:29.439",
                                    "Q3": "1:29.708"
                                }
                            ]
                        }
                    ]
                }
            }
        }
        mock_get.return_value = qualifying_response

        qualifying_results = client.get_qualifying_results(2023, 1)

        assert len(qualifying_results) == 1
        assert isinstance(qualifying_results[0], QualifyingResult)
        assert qualifying_results[0].position == 1
        assert qualifying_results[0].driver.driver_id == "verstappen"
        assert qualifying_results[0].q1 == "1:29.708"
        assert qualifying_results[0].q2 == "1:29.439"
        assert qualifying_results[0].q3 == "1:29.708"

        mock_get.assert_called_once_with("2023/1/qualifying.json", params={})