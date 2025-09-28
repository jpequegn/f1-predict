"""Ergast API client for F1 data.

This module provides a comprehensive client for the Ergast API,
offering access to historical Formula 1 data including races,
results, standings, drivers, constructors, and circuits.
"""

import logging
from typing import List, Optional, Union

from f1_predict.api.base import BaseAPIClient
from f1_predict.data.models import (
    Circuit,
    CircuitResponse,
    Constructor,
    ConstructorResponse,
    ConstructorStanding,
    Driver,
    DriverResponse,
    DriverStanding,
    QualifyingResponse,
    QualifyingResult,
    Race,
    RaceResponse,
    Result,
    ResultResponse,
    Season,
    SeasonResponse,
    StandingsResponse,
)


class ErgastAPIClient(BaseAPIClient):
    """Ergast API client for Formula 1 data.

    This client provides methods to fetch various F1 data including:
    - Seasons and races
    - Race results and qualifying
    - Driver and constructor standings
    - Driver and constructor information
    - Circuit information

    The client handles rate limiting, error handling, and data validation
    automatically.
    """

    DEFAULT_BASE_URL = "https://ergast.com/api/f1"

    def __init__(
        self,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """Initialize Ergast API client.

        Args:
            base_url: Optional custom base URL (defaults to official Ergast API)
            **kwargs: Additional arguments passed to BaseAPIClient
        """
        base_url = base_url or self.DEFAULT_BASE_URL
        super().__init__(base_url, **kwargs)
        self.logger = logging.getLogger("ErgastAPIClient")

    # Season and Race methods

    def get_seasons(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Season]:
        """Get list of F1 seasons.

        Args:
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of Season objects
        """
        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        response = self.get("seasons.json", params=params)

        if isinstance(response, dict) and "MRData" in response:
            seasons_data = response["MRData"].get("SeasonTable", {}).get("Seasons", [])
            return [Season.model_validate(season) for season in seasons_data]

        return []

    def get_races(
        self,
        season: Optional[Union[int, str]] = None,
        round_number: Optional[Union[int, str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Race]:
        """Get races for a season or specific race.

        Args:
            season: Season year (e.g., 2023, "current")
            round_number: Round number within season
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of Race objects
        """
        # Build endpoint
        endpoint_parts = []
        if season is not None:
            endpoint_parts.append(str(season))
        if round_number is not None:
            endpoint_parts.append(str(round_number))

        endpoint = "/".join(endpoint_parts) + ".json" if endpoint_parts else "races.json"

        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        response = self.get(endpoint, params=params)

        if isinstance(response, dict) and "MRData" in response:
            races_data = response["MRData"].get("RaceTable", {}).get("Races", [])
            return [Race.model_validate(race) for race in races_data]

        return []

    def get_current_season_races(self) -> List[Race]:
        """Get races for the current season.

        Returns:
            List of Race objects for current season
        """
        return self.get_races(season="current")

    # Results methods

    def get_race_results(
        self,
        season: Union[int, str],
        round_number: Union[int, str],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Result]:
        """Get results for a specific race.

        Args:
            season: Season year
            round_number: Round number within season
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of Result objects
        """
        endpoint = f"{season}/{round_number}/results.json"

        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        response = self.get(endpoint, params=params)

        if isinstance(response, dict) and "MRData" in response:
            race_table = response["MRData"].get("RaceTable", {})
            races = race_table.get("Races", [])
            if races:
                return [Result.model_validate(result) for result in races[0].get("Results", [])]

        return []

    def get_driver_results(
        self,
        driver_id: str,
        season: Optional[Union[int, str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Result]:
        """Get results for a specific driver.

        Args:
            driver_id: Driver ID (e.g., "hamilton", "verstappen")
            season: Optional season year
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of Result objects
        """
        endpoint_parts = []
        if season is not None:
            endpoint_parts.append(str(season))
        endpoint_parts.extend(["drivers", driver_id, "results.json"])

        endpoint = "/".join(endpoint_parts)

        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        response = self.get(endpoint, params=params)

        if isinstance(response, dict) and "MRData" in response:
            race_table = response["MRData"].get("RaceTable", {})
            races = race_table.get("Races", [])
            results = []
            for race in races:
                results.extend([Result.model_validate(result) for result in race.get("Results", [])])
            return results

        return []

    def get_constructor_results(
        self,
        constructor_id: str,
        season: Optional[Union[int, str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Result]:
        """Get results for a specific constructor.

        Args:
            constructor_id: Constructor ID (e.g., "mercedes", "red_bull")
            season: Optional season year
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of Result objects
        """
        endpoint_parts = []
        if season is not None:
            endpoint_parts.append(str(season))
        endpoint_parts.extend(["constructors", constructor_id, "results.json"])

        endpoint = "/".join(endpoint_parts)

        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        response = self.get(endpoint, params=params)

        if isinstance(response, dict) and "MRData" in response:
            race_table = response["MRData"].get("RaceTable", {})
            races = race_table.get("Races", [])
            results = []
            for race in races:
                results.extend([Result.model_validate(result) for result in race.get("Results", [])])
            return results

        return []

    # Qualifying methods

    def get_qualifying_results(
        self,
        season: Union[int, str],
        round_number: Union[int, str],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[QualifyingResult]:
        """Get qualifying results for a specific race.

        Args:
            season: Season year
            round_number: Round number within season
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of QualifyingResult objects
        """
        endpoint = f"{season}/{round_number}/qualifying.json"

        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        response = self.get(endpoint, params=params)

        if isinstance(response, dict) and "MRData" in response:
            race_table = response["MRData"].get("RaceTable", {})
            races = race_table.get("Races", [])
            if races:
                qualifying_results = races[0].get("QualifyingResults", [])
                return [QualifyingResult.model_validate(result) for result in qualifying_results]

        return []

    # Standings methods

    def get_driver_standings(
        self,
        season: Union[int, str],
        round_number: Optional[Union[int, str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[DriverStanding]:
        """Get driver championship standings.

        Args:
            season: Season year
            round_number: Optional round number (defaults to final standings)
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of DriverStanding objects
        """
        endpoint_parts = [str(season)]
        if round_number is not None:
            endpoint_parts.append(str(round_number))
        endpoint_parts.append("driverStandings.json")

        endpoint = "/".join(endpoint_parts)

        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        response = self.get(endpoint, params=params)

        if isinstance(response, dict) and "MRData" in response:
            standings_table = response["MRData"].get("StandingsTable", {})
            standings_lists = standings_table.get("StandingsLists", [])
            if standings_lists:
                driver_standings = standings_lists[0].get("DriverStandings", [])
                return [DriverStanding.model_validate(standing) for standing in driver_standings]

        return []

    def get_constructor_standings(
        self,
        season: Union[int, str],
        round_number: Optional[Union[int, str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[ConstructorStanding]:
        """Get constructor championship standings.

        Args:
            season: Season year
            round_number: Optional round number (defaults to final standings)
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of ConstructorStanding objects
        """
        endpoint_parts = [str(season)]
        if round_number is not None:
            endpoint_parts.append(str(round_number))
        endpoint_parts.append("constructorStandings.json")

        endpoint = "/".join(endpoint_parts)

        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        response = self.get(endpoint, params=params)

        if isinstance(response, dict) and "MRData" in response:
            standings_table = response["MRData"].get("StandingsTable", {})
            standings_lists = standings_table.get("StandingsLists", [])
            if standings_lists:
                constructor_standings = standings_lists[0].get("ConstructorStandings", [])
                return [ConstructorStanding.model_validate(standing) for standing in constructor_standings]

        return []

    def get_current_driver_standings(self) -> List[DriverStanding]:
        """Get current season driver standings.

        Returns:
            List of DriverStanding objects for current season
        """
        return self.get_driver_standings("current")

    def get_current_constructor_standings(self) -> List[ConstructorStanding]:
        """Get current season constructor standings.

        Returns:
            List of ConstructorStanding objects for current season
        """
        return self.get_constructor_standings("current")

    # Driver and Constructor information methods

    def get_drivers(
        self,
        season: Optional[Union[int, str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Driver]:
        """Get list of drivers.

        Args:
            season: Optional season year to filter drivers
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of Driver objects
        """
        endpoint_parts = []
        if season is not None:
            endpoint_parts.append(str(season))
        endpoint_parts.append("drivers.json")

        endpoint = "/".join(endpoint_parts)

        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        response = self.get(endpoint, params=params)

        if isinstance(response, dict) and "MRData" in response:
            drivers_data = response["MRData"].get("DriverTable", {}).get("Drivers", [])
            return [Driver.model_validate(driver) for driver in drivers_data]

        return []

    def get_driver(self, driver_id: str) -> Optional[Driver]:
        """Get specific driver information.

        Args:
            driver_id: Driver ID (e.g., "hamilton", "verstappen")

        Returns:
            Driver object or None if not found
        """
        endpoint = f"drivers/{driver_id}.json"
        response = self.get(endpoint)

        if isinstance(response, dict) and "MRData" in response:
            drivers_data = response["MRData"].get("DriverTable", {}).get("Drivers", [])
            if drivers_data:
                return Driver.model_validate(drivers_data[0])

        return None

    def get_constructors(
        self,
        season: Optional[Union[int, str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Constructor]:
        """Get list of constructors.

        Args:
            season: Optional season year to filter constructors
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of Constructor objects
        """
        endpoint_parts = []
        if season is not None:
            endpoint_parts.append(str(season))
        endpoint_parts.append("constructors.json")

        endpoint = "/".join(endpoint_parts)

        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        response = self.get(endpoint, params=params)

        if isinstance(response, dict) and "MRData" in response:
            constructors_data = response["MRData"].get("ConstructorTable", {}).get("Constructors", [])
            return [Constructor.model_validate(constructor) for constructor in constructors_data]

        return []

    def get_constructor(self, constructor_id: str) -> Optional[Constructor]:
        """Get specific constructor information.

        Args:
            constructor_id: Constructor ID (e.g., "mercedes", "red_bull")

        Returns:
            Constructor object or None if not found
        """
        endpoint = f"constructors/{constructor_id}.json"
        response = self.get(endpoint)

        if isinstance(response, dict) and "MRData" in response:
            constructors_data = response["MRData"].get("ConstructorTable", {}).get("Constructors", [])
            if constructors_data:
                return Constructor.model_validate(constructors_data[0])

        return None

    # Circuit methods

    def get_circuits(
        self,
        season: Optional[Union[int, str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Circuit]:
        """Get list of circuits.

        Args:
            season: Optional season year to filter circuits
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of Circuit objects
        """
        endpoint_parts = []
        if season is not None:
            endpoint_parts.append(str(season))
        endpoint_parts.append("circuits.json")

        endpoint = "/".join(endpoint_parts)

        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        response = self.get(endpoint, params=params)

        if isinstance(response, dict) and "MRData" in response:
            circuits_data = response["MRData"].get("CircuitTable", {}).get("Circuits", [])
            return [Circuit.model_validate(circuit) for circuit in circuits_data]

        return []

    def get_circuit(self, circuit_id: str) -> Optional[Circuit]:
        """Get specific circuit information.

        Args:
            circuit_id: Circuit ID (e.g., "monaco", "silverstone")

        Returns:
            Circuit object or None if not found
        """
        endpoint = f"circuits/{circuit_id}.json"
        response = self.get(endpoint)

        if isinstance(response, dict) and "MRData" in response:
            circuits_data = response["MRData"].get("CircuitTable", {}).get("Circuits", [])
            if circuits_data:
                return Circuit.model_validate(circuits_data[0])

        return None

    # Convenience methods

    def get_last_race_results(self, season: Optional[Union[int, str]] = None) -> List[Result]:
        """Get results from the last completed race.

        Args:
            season: Optional season year (defaults to current)

        Returns:
            List of Result objects from last race
        """
        season = season or "current"
        return self.get_race_results(season, "last")

    def get_next_race(self, season: Optional[Union[int, str]] = None) -> Optional[Race]:
        """Get the next upcoming race.

        Args:
            season: Optional season year (defaults to current)

        Returns:
            Next Race object or None
        """
        season = season or "current"
        races = self.get_races(season)

        # In a real implementation, you would filter for upcoming races
        # For now, return the first race as a placeholder
        return races[0] if races else None

    def search_drivers(self, name: str) -> List[Driver]:
        """Search for drivers by name.

        Args:
            name: Driver name to search for

        Returns:
            List of matching Driver objects
        """
        # Get all drivers and filter by name
        all_drivers = self.get_drivers(limit=1000)  # Get a large number
        name_lower = name.lower()

        return [
            driver
            for driver in all_drivers
            if name_lower in driver.given_name.lower()
            or name_lower in driver.family_name.lower()
            or (driver.code and name_lower in driver.code.lower())
        ]

    def search_constructors(self, name: str) -> List[Constructor]:
        """Search for constructors by name.

        Args:
            name: Constructor name to search for

        Returns:
            List of matching Constructor objects
        """
        # Get all constructors and filter by name
        all_constructors = self.get_constructors(limit=1000)  # Get a large number
        name_lower = name.lower()

        return [constructor for constructor in all_constructors if name_lower in constructor.name.lower()]