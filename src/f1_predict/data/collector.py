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

    def collect_and_clean_all_data(
        self, force_refresh: bool = False, enable_cleaning: bool = True, enable_enrichment: bool = False
    ) -> dict[str, any]:
        """Collect and optionally clean all types of F1 data.

        Args:
            force_refresh: If True, re-download data even if files exist
            enable_cleaning: If True, clean the data after collection
            enable_enrichment: If True, enrich data with external sources

        Returns:
            Dictionary with status of collection and cleaning operations
        """
        self.logger.info("Starting comprehensive F1 data collection and cleaning")

        # First collect raw data
        collection_results = self.collect_all_data(force_refresh=force_refresh)

        results = {"collection": collection_results}

        if enable_cleaning:
            try:
                cleaning_results = self.clean_collected_data()
                results["cleaning"] = cleaning_results

            except ImportError:
                self.logger.warning("Data cleaning module not available")
                results["cleaning"] = {"error": "Cleaning module not available"}
            except Exception as e:
                self.logger.error(f"Data cleaning failed: {e}")
                results["cleaning"] = {"error": str(e)}

        if enable_enrichment:
            try:
                enrichment_results = self.enrich_collected_data()
                results["enrichment"] = enrichment_results

            except ImportError:
                self.logger.warning("External data enrichment module not available")
                results["enrichment"] = {"error": "Enrichment module not available"}
            except Exception as e:
                self.logger.error(f"Data enrichment failed: {e}")
                results["enrichment"] = {"error": str(e)}

        return results

    def clean_collected_data(self) -> dict[str, any]:
        """Clean previously collected data.

        Returns:
            Dictionary with cleaning results for each data type
        """
        from f1_predict.data.cleaning import DataCleaner

        self.logger.info("Starting data cleaning pipeline")

        cleaner = DataCleaner(enable_logging=True)
        results = {}

        # Clean race results
        race_results_file = self.raw_dir / "race_results_2020_2024.json"
        if race_results_file.exists():
            try:
                with open(race_results_file) as f:
                    race_data = json.load(f)

                cleaned_data, report = cleaner.clean_race_results(race_data)

                # Save cleaned data
                cleaned_file = self.processed_dir / "race_results_cleaned.json"
                self._save_to_json(cleaned_data, cleaned_file)

                # Also save as CSV
                csv_file = self.processed_dir / "race_results_cleaned.csv"
                self._save_to_csv(cleaned_data, csv_file)

                results["race_results"] = {
                    "status": "success",
                    "input_count": len(race_data),
                    "output_count": len(cleaned_data),
                    "quality_score": report.quality_score,
                    "cleaned_file": str(cleaned_file),
                    "cleaned_csv": str(csv_file),
                    "quality_passed": cleaner.validate_data_quality(report),
                }

                self.logger.info(
                    f"Race results cleaned: {len(cleaned_data)}/{len(race_data)} records (quality: {report.quality_score:.1f}%)"
                )

            except Exception as e:
                results["race_results"] = {"status": "error", "message": str(e)}
                self.logger.error(f"Failed to clean race results: {e}")
        else:
            results["race_results"] = {
                "status": "skipped",
                "message": "Raw data file not found",
            }

        # Clean qualifying results
        qualifying_file = self.raw_dir / "qualifying_results_2020_2024.json"
        if qualifying_file.exists():
            try:
                with open(qualifying_file) as f:
                    qualifying_data = json.load(f)

                cleaned_data, report = cleaner.clean_qualifying_results(qualifying_data)

                # Save cleaned data
                cleaned_file = self.processed_dir / "qualifying_results_cleaned.json"
                self._save_to_json(cleaned_data, cleaned_file)

                # Also save as CSV
                csv_file = self.processed_dir / "qualifying_results_cleaned.csv"
                self._save_to_csv(cleaned_data, csv_file)

                results["qualifying_results"] = {
                    "status": "success",
                    "input_count": len(qualifying_data),
                    "output_count": len(cleaned_data),
                    "quality_score": report.quality_score,
                    "cleaned_file": str(cleaned_file),
                    "cleaned_csv": str(csv_file),
                    "quality_passed": cleaner.validate_data_quality(report),
                }

                self.logger.info(
                    f"Qualifying results cleaned: {len(cleaned_data)}/{len(qualifying_data)} records (quality: {report.quality_score:.1f}%)"
                )

            except Exception as e:
                results["qualifying_results"] = {"status": "error", "message": str(e)}
                self.logger.error(f"Failed to clean qualifying results: {e}")
        else:
            results["qualifying_results"] = {
                "status": "skipped",
                "message": "Raw data file not found",
            }

        # Clean race schedules
        schedules_file = self.raw_dir / "race_schedules_2020_2024.json"
        if schedules_file.exists():
            try:
                with open(schedules_file) as f:
                    schedules_data = json.load(f)

                cleaned_data, report = cleaner.clean_race_schedules(schedules_data)

                # Save cleaned data
                cleaned_file = self.processed_dir / "race_schedules_cleaned.json"
                self._save_to_json(cleaned_data, cleaned_file)

                # Also save as CSV
                csv_file = self.processed_dir / "race_schedules_cleaned.csv"
                self._save_to_csv(cleaned_data, csv_file)

                results["race_schedules"] = {
                    "status": "success",
                    "input_count": len(schedules_data),
                    "output_count": len(cleaned_data),
                    "quality_score": report.quality_score,
                    "cleaned_file": str(cleaned_file),
                    "cleaned_csv": str(csv_file),
                    "quality_passed": cleaner.validate_data_quality(report),
                }

                self.logger.info(
                    f"Race schedules cleaned: {len(cleaned_data)}/{len(schedules_data)} records (quality: {report.quality_score:.1f}%)"
                )

            except Exception as e:
                results["race_schedules"] = {"status": "error", "message": str(e)}
                self.logger.error(f"Failed to clean race schedules: {e}")
        else:
            results["race_schedules"] = {
                "status": "skipped",
                "message": "Raw data file not found",
            }

        # Generate cleaning summary
        summary_file = self.processed_dir / "cleaning_summary.json"
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "cleaning_results": results,
                "overall_status": "success"
                if all(
                    r.get("status") == "success"
                    for r in results.values()
                    if r.get("status") != "skipped"
                )
                else "partial_success",
            }

            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            self.logger.info(f"Cleaning summary saved to: {summary_file}")

        except Exception as e:
            self.logger.warning(f"Failed to save cleaning summary: {e}")

        self.logger.info("Data cleaning pipeline completed")
        return results

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

    def enrich_collected_data(self) -> dict[str, any]:
        """Enrich collected data with external sources (weather, track, tire data).

        Returns:
            Dictionary with enrichment results for each data type
        """
        from f1_predict.data.track_data import TrackDataManager
        from f1_predict.data.weather_collector import WeatherDataCollector

        self.logger.info("Starting data enrichment pipeline")

        results = {}

        # Load schedule data for circuit information
        schedules_file = self.raw_dir / "race_schedules_2020_2024.json"
        if not schedules_file.exists():
            results["error"] = "Race schedules not found. Run data collection first."
            return results

        with open(schedules_file) as f:
            schedules = json.load(f)

        # Initialize external data collectors
        track_manager = TrackDataManager()
        enriched_data = []

        # Enrich each race with external data
        for schedule in schedules:
            try:
                circuit_id = schedule.get("circuit_id")
                season = schedule.get("season")
                round_num = schedule.get("round")

                # Get track characteristics
                track_chars = track_manager.get_track(circuit_id)

                enrichment = {
                    "season": season,
                    "round": round_num,
                    "circuit_id": circuit_id,
                    "track_characteristics": track_chars.model_dump() if track_chars else None,
                }

                enriched_data.append(enrichment)

                self.logger.debug(
                    f"Enriched {season} round {round_num} with track data: {circuit_id}"
                )

            except Exception as e:
                self.logger.warning(f"Failed to enrich race {season} round {round_num}: {e}")
                continue

        # Save enriched data
        if enriched_data:
            enriched_file = self.processed_dir / "enriched_race_data.json"
            self._save_to_json(enriched_data, enriched_file)

            results["enriched_data"] = {
                "status": "success",
                "enriched_count": len(enriched_data),
                "total_schedules": len(schedules),
                "enriched_file": str(enriched_file),
                "coverage": len(enriched_data) / len(schedules) if schedules else 0,
            }

            self.logger.info(
                f"Enriched {len(enriched_data)}/{len(schedules)} races with external data"
            )
        else:
            results["enriched_data"] = {
                "status": "failed",
                "message": "No data was enriched",
            }

        # Note about weather data
        results["weather_note"] = (
            "Weather data collection requires OPENWEATHER_API_KEY environment variable. "
            "Historical weather data may require paid OpenWeatherMap subscription."
        )

        self.logger.info("Data enrichment pipeline completed")
        return results

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.client:
            self.client.close()
