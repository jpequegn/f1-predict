"""Historical F1 data collection module.

This module provides functionality to collect historical Formula 1 data
from the Ergast API for the years 2020-2024, including race results,
qualifying results, lap times, and pit stop data.
"""

import csv
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Optional

from f1_predict.api.ergast import ErgastAPIClient


class F1DataCollector:
    """Collects and stores historical F1 data from the Ergast API."""

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the data collector.

        Args:
            data_dir: Base directory for storing data files. Defaults to 'data/'
        """
        self.client = ErgastAPIClient()
        self.logger = logging.getLogger(__name__)

        # Set up data directory structure
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Define the seasons to collect data for
        self.seasons = list(range(2020, 2025))  # 2020-2024

        self.logger.info(f"F1DataCollector initialized for seasons {self.seasons}")

    def collect_all_data(self, force_refresh: bool = False) -> dict[str, str]:
        """Collect all types of F1 data for the specified seasons.

        Args:
            force_refresh: If True, re-download data even if files exist

        Returns:
            Dictionary with status of each data collection operation
        """
        self.logger.info("Starting comprehensive F1 data collection")

        results = {}

        # Collect race results
        try:
            race_file = self.collect_race_results(force_refresh=force_refresh)
            results["race_results"] = f"Success: {race_file}"
        except Exception as e:
            results["race_results"] = f"Error: {str(e)}"
            self.logger.error(f"Failed to collect race results: {e}")

        # Collect qualifying results
        try:
            qualifying_file = self.collect_qualifying_results(
                force_refresh=force_refresh
            )
            results["qualifying_results"] = f"Success: {qualifying_file}"
        except Exception as e:
            results["qualifying_results"] = f"Error: {str(e)}"
            self.logger.error(f"Failed to collect qualifying results: {e}")

        # Collect race schedules
        try:
            schedule_file = self.collect_race_schedules(force_refresh=force_refresh)
            results["race_schedules"] = f"Success: {schedule_file}"
        except Exception as e:
            results["race_schedules"] = f"Error: {str(e)}"
            self.logger.error(f"Failed to collect race schedules: {e}")

        # Note: Lap times and pit stop data require additional API endpoints
        # that may not be available in the current Ergast implementation
        results["lap_times"] = "Not implemented - requires additional API endpoints"
        results["pit_stops"] = "Not implemented - requires additional API endpoints"

        self.logger.info("Data collection completed")
        return results

    def collect_race_results(self, force_refresh: bool = False) -> str:
        """Collect race results for all seasons.

        Args:
            force_refresh: If True, re-download data even if file exists

        Returns:
            Path to the saved CSV file
        """
        output_file = self.raw_dir / "race_results_2020_2024.csv"

        if output_file.exists() and not force_refresh:
            self.logger.info(f"Race results file exists: {output_file}")
            return str(output_file)

        self.logger.info("Collecting race results for seasons 2020-2024")

        all_results = []

        for season in self.seasons:
            self.logger.info(f"Collecting race results for season {season}")

            try:
                # Get races for the season
                races = self.client.get_races(season)

                for race in races:
                    # Get results for each race
                    try:
                        results = self.client.get_race_results(season, race.round)

                        for result in results:
                            race_result = {
                                "season": season,
                                "round": race.round,
                                "race_name": race.race_name,
                                "circuit_id": race.circuit.circuit_id,
                                "circuit_name": race.circuit.circuit_name,
                                "location": race.circuit.location.locality,
                                "country": race.circuit.location.country,
                                "date": race.date.isoformat(),
                                "position": result.position,
                                "position_text": result.position_text,
                                "points": result.points,
                                "driver_id": result.driver.driver_id,
                                "driver_name": f"{result.driver.given_name} {result.driver.family_name}",
                                "driver_nationality": result.driver.nationality,
                                "constructor_id": result.constructor.constructor_id,
                                "constructor_name": result.constructor.name,
                                "constructor_nationality": result.constructor.nationality,
                                "grid": result.grid,
                                "laps": result.laps,
                                "status": result.status,
                                "time_millis": result.time.millis
                                if result.time
                                else None,
                                "time_formatted": result.time.time
                                if result.time
                                else None,
                            }
                            all_results.append(race_result)

                    except Exception as e:
                        self.logger.warning(
                            f"Failed to get results for {season} round {race.round}: {e}"
                        )
                        continue

            except Exception as e:
                self.logger.error(f"Failed to get races for season {season}: {e}")
                continue

        # Save to CSV
        self._save_to_csv(all_results, output_file)
        self.logger.info(f"Saved {len(all_results)} race results to {output_file}")

        # Also save as JSON
        json_file = self.raw_dir / "race_results_2020_2024.json"
        self._save_to_json(all_results, json_file)

        return str(output_file)

    def collect_qualifying_results(self, force_refresh: bool = False) -> str:
        """Collect qualifying results for all seasons.

        Args:
            force_refresh: If True, re-download data even if file exists

        Returns:
            Path to the saved CSV file
        """
        output_file = self.raw_dir / "qualifying_results_2020_2024.csv"

        if output_file.exists() and not force_refresh:
            self.logger.info(f"Qualifying results file exists: {output_file}")
            return str(output_file)

        self.logger.info("Collecting qualifying results for seasons 2020-2024")

        all_qualifying = []

        for season in self.seasons:
            self.logger.info(f"Collecting qualifying results for season {season}")

            try:
                # Get races for the season
                races = self.client.get_races(season)

                for race in races:
                    # Get qualifying results for each race
                    try:
                        qualifying_results = self.client.get_qualifying_results(
                            season, race.round
                        )

                        for result in qualifying_results:
                            qualifying_result = {
                                "season": season,
                                "round": race.round,
                                "race_name": race.race_name,
                                "circuit_id": race.circuit.circuit_id,
                                "circuit_name": race.circuit.circuit_name,
                                "location": race.circuit.location.locality,
                                "country": race.circuit.location.country,
                                "date": race.date.isoformat(),
                                "position": result.position,
                                "driver_id": result.driver.driver_id,
                                "driver_name": f"{result.driver.given_name} {result.driver.family_name}",
                                "driver_nationality": result.driver.nationality,
                                "constructor_id": result.constructor.constructor_id,
                                "constructor_name": result.constructor.name,
                                "constructor_nationality": result.constructor.nationality,
                                "q1": result.q1,
                                "q2": result.q2,
                                "q3": result.q3,
                            }
                            all_qualifying.append(qualifying_result)

                    except Exception as e:
                        self.logger.warning(
                            f"Failed to get qualifying for {season} round {race.round}: {e}"
                        )
                        continue

            except Exception as e:
                self.logger.error(f"Failed to get races for season {season}: {e}")
                continue

        # Save to CSV
        self._save_to_csv(all_qualifying, output_file)
        self.logger.info(
            f"Saved {len(all_qualifying)} qualifying results to {output_file}"
        )

        # Also save as JSON
        json_file = self.raw_dir / "qualifying_results_2020_2024.json"
        self._save_to_json(all_qualifying, json_file)

        return str(output_file)

    def collect_race_schedules(self, force_refresh: bool = False) -> str:
        """Collect race schedules for all seasons.

        Args:
            force_refresh: If True, re-download data even if file exists

        Returns:
            Path to the saved CSV file
        """
        output_file = self.raw_dir / "race_schedules_2020_2024.csv"

        if output_file.exists() and not force_refresh:
            self.logger.info(f"Race schedules file exists: {output_file}")
            return str(output_file)

        self.logger.info("Collecting race schedules for seasons 2020-2024")

        all_schedules = []

        for season in self.seasons:
            self.logger.info(f"Collecting race schedule for season {season}")

            try:
                races = self.client.get_races(season)

                for race in races:
                    schedule = {
                        "season": season,
                        "round": race.round,
                        "race_name": race.race_name,
                        "circuit_id": race.circuit.circuit_id,
                        "circuit_name": race.circuit.circuit_name,
                        "location": race.circuit.location.locality,
                        "country": race.circuit.location.country,
                        "latitude": race.circuit.location.lat,
                        "longitude": race.circuit.location.long,
                        "date": race.date.isoformat(),
                        "time": race.time.isoformat() if race.time else None,
                        "url": race.url,
                        "fp1_date": race.first_practice.date.isoformat()
                        if race.first_practice
                        else None,
                        "fp1_time": race.first_practice.time.isoformat()
                        if race.first_practice and race.first_practice.time
                        else None,
                        "fp2_date": race.second_practice.date.isoformat()
                        if race.second_practice
                        else None,
                        "fp2_time": race.second_practice.time.isoformat()
                        if race.second_practice and race.second_practice.time
                        else None,
                        "fp3_date": race.third_practice.date.isoformat()
                        if race.third_practice
                        else None,
                        "fp3_time": race.third_practice.time.isoformat()
                        if race.third_practice and race.third_practice.time
                        else None,
                        "qualifying_date": race.qualifying.date.isoformat()
                        if race.qualifying
                        else None,
                        "qualifying_time": race.qualifying.time.isoformat()
                        if race.qualifying and race.qualifying.time
                        else None,
                        "sprint_date": race.sprint.date.isoformat()
                        if race.sprint
                        else None,
                        "sprint_time": race.sprint.time.isoformat()
                        if race.sprint and race.sprint.time
                        else None,
                    }
                    all_schedules.append(schedule)

            except Exception as e:
                self.logger.error(f"Failed to get schedule for season {season}: {e}")
                continue

        # Save to CSV
        self._save_to_csv(all_schedules, output_file)
        self.logger.info(f"Saved {len(all_schedules)} race schedules to {output_file}")

        # Also save as JSON
        json_file = self.raw_dir / "race_schedules_2020_2024.json"
        self._save_to_json(all_schedules, json_file)

        return str(output_file)

    def refresh_data(self) -> dict[str, str]:
        """Refresh all collected data by re-downloading from the API.

        Returns:
            Dictionary with status of each refresh operation
        """
        self.logger.info("Refreshing all F1 data")
        return self.collect_all_data(force_refresh=True)

    def get_data_summary(self) -> dict[str, any]:
        """Get a summary of collected data files.

        Returns:
            Dictionary with information about available data files
        """
        summary = {
            "data_directory": str(self.data_dir),
            "raw_files": {},
            "processed_files": {},
            "last_updated": datetime.now().isoformat(),
        }

        # Check raw files
        for file_path in self.raw_dir.glob("*.csv"):
            stat = file_path.stat()
            summary["raw_files"][file_path.name] = {
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }

        # Check processed files
        for file_path in self.processed_dir.glob("*.csv"):
            stat = file_path.stat()
            summary["processed_files"][file_path.name] = {
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }

        return summary

    def _save_to_csv(self, data: list[dict], file_path: Path) -> None:
        """Save data to CSV file.

        Args:
            data: List of dictionaries to save
            file_path: Path to save the CSV file
        """
        if not data:
            self.logger.warning("No data to save to CSV")
            return

        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    def _save_to_json(self, data: list[dict], file_path: Path) -> None:
        """Save data to JSON file.

        Args:
            data: List of dictionaries to save
            file_path: Path to save the JSON file
        """
        with open(file_path, "w", encoding="utf-8") as jsonfile:
            json.dump(data, jsonfile, indent=2, ensure_ascii=False)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.client:
            self.client.close()
