"""Data cleaning pipeline for F1 prediction data.

This module provides comprehensive data cleaning functionality including:
- Missing data handling
- Driver/team name standardization
- Data type conversion and formatting
- Data validation
- Quality checks and reporting
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from datetime import time as dt_time
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class DataQualityReport:
    """Report containing data quality metrics and issues."""

    total_records: int = 0
    missing_values: Dict[str, int] = field(default_factory=dict)
    data_type_issues: List[str] = field(default_factory=list)
    standardization_changes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class DataCleaner:
    """Main data cleaning pipeline for F1 data."""

    def __init__(self, enable_logging: bool = True):
        """Initialize the data cleaner.

        Args:
            enable_logging: Whether to enable detailed logging
        """
        self.logger = logging.getLogger(__name__)
        self.enable_logging = enable_logging

        # Initialize standardization mappings
        self._init_driver_mappings()
        self._init_constructor_mappings()
        self._init_circuit_mappings()

        # Quality thresholds
        self.quality_thresholds = {
            'missing_data_percent': 5.0,  # Max 5% missing data
            'invalid_data_percent': 2.0,  # Max 2% invalid data
            'min_quality_score': 85.0     # Min 85% quality score
        }

    def clean_race_results(self, results_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], DataQualityReport]:
        """Clean race results data.

        Args:
            results_data: List of race result dictionaries

        Returns:
            Tuple of (cleaned_data, quality_report)
        """
        if self.enable_logging:
            self.logger.info(f"Cleaning {len(results_data)} race results")

        cleaned_data = []
        report = DataQualityReport(total_records=len(results_data))

        for result in results_data:
            try:
                cleaned_result = self._clean_race_result(result, report)
                if cleaned_result:
                    cleaned_data.append(cleaned_result)
            except Exception as e:
                report.validation_errors.append(f"Error cleaning result: {str(e)}")
                if self.enable_logging:
                    self.logger.warning(f"Failed to clean race result: {e}")

        # Calculate quality score
        report.quality_score = self._calculate_quality_score(report)

        if self.enable_logging:
            self.logger.info(f"Cleaned {len(cleaned_data)}/{len(results_data)} results (quality: {report.quality_score:.1f}%)")

        return cleaned_data, report

    def clean_qualifying_results(self, qualifying_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], DataQualityReport]:
        """Clean qualifying results data.

        Args:
            qualifying_data: List of qualifying result dictionaries

        Returns:
            Tuple of (cleaned_data, quality_report)
        """
        if self.enable_logging:
            self.logger.info(f"Cleaning {len(qualifying_data)} qualifying results")

        cleaned_data = []
        report = DataQualityReport(total_records=len(qualifying_data))

        for result in qualifying_data:
            try:
                cleaned_result = self._clean_qualifying_result(result, report)
                if cleaned_result:
                    cleaned_data.append(cleaned_result)
            except Exception as e:
                report.validation_errors.append(f"Error cleaning qualifying result: {str(e)}")
                if self.enable_logging:
                    self.logger.warning(f"Failed to clean qualifying result: {e}")

        # Calculate quality score
        report.quality_score = self._calculate_quality_score(report)

        if self.enable_logging:
            self.logger.info(f"Cleaned {len(cleaned_data)}/{len(qualifying_data)} qualifying results (quality: {report.quality_score:.1f}%)")

        return cleaned_data, report

    def clean_race_schedules(self, schedule_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], DataQualityReport]:
        """Clean race schedule data.

        Args:
            schedule_data: List of race schedule dictionaries

        Returns:
            Tuple of (cleaned_data, quality_report)
        """
        if self.enable_logging:
            self.logger.info(f"Cleaning {len(schedule_data)} race schedules")

        cleaned_data = []
        report = DataQualityReport(total_records=len(schedule_data))

        for schedule in schedule_data:
            try:
                cleaned_schedule = self._clean_race_schedule(schedule, report)
                if cleaned_schedule:
                    cleaned_data.append(cleaned_schedule)
            except Exception as e:
                report.validation_errors.append(f"Error cleaning schedule: {str(e)}")
                if self.enable_logging:
                    self.logger.warning(f"Failed to clean race schedule: {e}")

        # Calculate quality score
        report.quality_score = self._calculate_quality_score(report)

        if self.enable_logging:
            self.logger.info(f"Cleaned {len(cleaned_data)}/{len(schedule_data)} schedules (quality: {report.quality_score:.1f}%)")

        return cleaned_data, report

    def validate_data_quality(self, report: DataQualityReport) -> bool:
        """Validate data quality against thresholds.

        Args:
            report: Data quality report to validate

        Returns:
            True if data meets quality standards
        """
        if report.total_records == 0:
            return False

        # Calculate missing data percentage
        total_missing = sum(report.missing_values.values())
        missing_percent = (total_missing / report.total_records) * 100

        # Calculate invalid data percentage
        invalid_percent = (len(report.validation_errors) / report.total_records) * 100

        # Check all thresholds
        quality_checks = [
            missing_percent <= self.quality_thresholds['missing_data_percent'],
            invalid_percent <= self.quality_thresholds['invalid_data_percent'],
            report.quality_score >= self.quality_thresholds['min_quality_score']
        ]

        return all(quality_checks)

    def _clean_race_result(self, result: Dict[str, Any], report: DataQualityReport) -> Optional[Dict[str, Any]]:
        """Clean a single race result."""
        cleaned = result.copy()

        # Handle missing values
        cleaned = self._handle_missing_values(cleaned, report, 'race_result')

        # Standardize names
        cleaned = self._standardize_driver_name(cleaned, report)
        cleaned = self._standardize_constructor_name(cleaned, report)

        # Convert and validate data types
        cleaned = self._convert_race_result_types(cleaned, report)

        # Validate result data
        if not self._validate_race_result(cleaned, report):
            return None

        return cleaned

    def _clean_qualifying_result(self, result: Dict[str, Any], report: DataQualityReport) -> Optional[Dict[str, Any]]:
        """Clean a single qualifying result."""
        cleaned = result.copy()

        # Handle missing values
        cleaned = self._handle_missing_values(cleaned, report, 'qualifying_result')

        # Standardize names
        cleaned = self._standardize_driver_name(cleaned, report)
        cleaned = self._standardize_constructor_name(cleaned, report)

        # Convert and validate data types
        cleaned = self._convert_qualifying_result_types(cleaned, report)

        # Clean qualifying times
        cleaned = self._clean_qualifying_times(cleaned, report)

        # Validate result data
        if not self._validate_qualifying_result(cleaned, report):
            return None

        return cleaned

    def _clean_race_schedule(self, schedule: Dict[str, Any], report: DataQualityReport) -> Optional[Dict[str, Any]]:
        """Clean a single race schedule."""
        cleaned = schedule.copy()

        # Handle missing values
        cleaned = self._handle_missing_values(cleaned, report, 'race_schedule')

        # Standardize circuit name
        cleaned = self._standardize_circuit_name(cleaned, report)

        # Convert and validate data types
        cleaned = self._convert_schedule_types(cleaned, report)

        # Validate schedule data
        if not self._validate_race_schedule(cleaned, report):
            return None

        return cleaned

    def _handle_missing_values(self, data: Dict[str, Any], report: DataQualityReport, data_type: str) -> Dict[str, Any]:
        """Handle missing values in data."""
        cleaned = data.copy()

        for key, value in data.items():
            if value is None or value == '' or (isinstance(value, str) and value.strip() == ''):
                # Track missing values
                field_key = f"{data_type}.{key}"
                report.missing_values[field_key] = report.missing_values.get(field_key, 0) + 1

                # Apply appropriate default values
                cleaned[key] = self._get_default_value(key, data_type)

        return cleaned

    def _get_default_value(self, field: str, data_type: str) -> Any:
        """Get appropriate default value for a field."""
        # Default values based on field type and context
        defaults = {
            'position': None,  # Keep None for missing positions
            'points': 0.0,
            'grid': 0,
            'laps': 0,
            'status': 'Unknown',
            'time_millis': None,
            'time_formatted': None,
            'q1': None,
            'q2': None,
            'q3': None,
            'nationality': 'Unknown',
            'country': 'Unknown',
            'locality': 'Unknown'
        }

        return defaults.get(field)

    def _standardize_driver_name(self, data: Dict[str, Any], report: DataQualityReport) -> Dict[str, Any]:
        """Standardize driver names across seasons."""
        if 'driver_id' not in data and 'driver_name' not in data:
            return data

        cleaned = data.copy()

        # Use driver_id as primary identifier for standardization
        if 'driver_id' in data and data['driver_id']:
            driver_id = data['driver_id']
            if driver_id in self.driver_name_mappings:
                standard_name = self.driver_name_mappings[driver_id]
                if 'driver_name' in cleaned and cleaned['driver_name'] != standard_name:
                    # Track standardization change
                    if 'driver_names' not in report.standardization_changes:
                        report.standardization_changes['driver_names'] = {}
                    report.standardization_changes['driver_names'][cleaned['driver_name']] = standard_name
                    cleaned['driver_name'] = standard_name

        return cleaned

    def _standardize_constructor_name(self, data: Dict[str, Any], report: DataQualityReport) -> Dict[str, Any]:
        """Standardize constructor/team names across seasons."""
        if 'constructor_id' not in data and 'constructor_name' not in data:
            return data

        cleaned = data.copy()

        # Use constructor_id as primary identifier for standardization
        if 'constructor_id' in data and data['constructor_id']:
            constructor_id = data['constructor_id']
            if constructor_id in self.constructor_name_mappings:
                standard_name = self.constructor_name_mappings[constructor_id]
                if 'constructor_name' in cleaned and cleaned['constructor_name'] != standard_name:
                    # Track standardization change
                    if 'constructor_names' not in report.standardization_changes:
                        report.standardization_changes['constructor_names'] = {}
                    report.standardization_changes['constructor_names'][cleaned['constructor_name']] = standard_name
                    cleaned['constructor_name'] = standard_name

        return cleaned

    def _standardize_circuit_name(self, data: Dict[str, Any], report: DataQualityReport) -> Dict[str, Any]:
        """Standardize circuit names across seasons."""
        if 'circuit_id' not in data and 'circuit_name' not in data:
            return data

        cleaned = data.copy()

        # Use circuit_id as primary identifier for standardization
        if 'circuit_id' in data and data['circuit_id']:
            circuit_id = data['circuit_id']
            if circuit_id in self.circuit_name_mappings:
                standard_name = self.circuit_name_mappings[circuit_id]
                if 'circuit_name' in cleaned and cleaned['circuit_name'] != standard_name:
                    # Track standardization change
                    if 'circuit_names' not in report.standardization_changes:
                        report.standardization_changes['circuit_names'] = {}
                    report.standardization_changes['circuit_names'][cleaned['circuit_name']] = standard_name
                    cleaned['circuit_name'] = standard_name

        return cleaned

    def _convert_race_result_types(self, data: Dict[str, Any], report: DataQualityReport) -> Dict[str, Any]:
        """Convert and validate data types for race results."""
        cleaned = data.copy()

        # Convert numeric fields
        numeric_fields = ['season', 'round', 'position', 'points', 'grid', 'laps', 'time_millis']
        for field in numeric_fields:
            if field in cleaned:
                cleaned[field] = self._convert_to_numeric(cleaned[field], field, report)

        # Convert date field
        if 'date' in cleaned:
            cleaned['date'] = self._convert_to_date(cleaned['date'], report)

        # Ensure string fields are properly formatted
        string_fields = ['race_name', 'driver_name', 'constructor_name', 'status']
        for field in string_fields:
            if field in cleaned and cleaned[field] is not None:
                cleaned[field] = str(cleaned[field]).strip()

        return cleaned

    def _convert_qualifying_result_types(self, data: Dict[str, Any], report: DataQualityReport) -> Dict[str, Any]:
        """Convert and validate data types for qualifying results."""
        cleaned = data.copy()

        # Convert numeric fields
        numeric_fields = ['season', 'round', 'position']
        for field in numeric_fields:
            if field in cleaned:
                cleaned[field] = self._convert_to_numeric(cleaned[field], field, report)

        # Convert date field
        if 'date' in cleaned:
            cleaned['date'] = self._convert_to_date(cleaned['date'], report)

        # Ensure string fields are properly formatted
        string_fields = ['race_name', 'driver_name', 'constructor_name']
        for field in string_fields:
            if field in cleaned and cleaned[field] is not None:
                cleaned[field] = str(cleaned[field]).strip()

        return cleaned

    def _convert_schedule_types(self, data: Dict[str, Any], report: DataQualityReport) -> Dict[str, Any]:
        """Convert and validate data types for race schedules."""
        cleaned = data.copy()

        # Convert numeric fields
        numeric_fields = ['season', 'round', 'latitude', 'longitude']
        for field in numeric_fields:
            if field in cleaned:
                cleaned[field] = self._convert_to_numeric(cleaned[field], field, report)

        # Convert date field
        if 'date' in cleaned:
            cleaned['date'] = self._convert_to_date(cleaned['date'], report)

        # Convert time fields
        time_fields = ['time', 'fp1_time', 'fp2_time', 'fp3_time', 'qualifying_time', 'sprint_time']
        for field in time_fields:
            if field in cleaned and cleaned[field] is not None:
                cleaned[field] = self._convert_to_time(cleaned[field], report)

        return cleaned

    def _convert_to_numeric(self, value: Any, field: str, report: DataQualityReport) -> Optional[Union[int, float]]:
        """Convert value to appropriate numeric type."""
        if value is None or value == '':
            return None

        try:
            # Try integer first for position, grid, laps, season, round
            if field in ['season', 'round', 'position', 'grid', 'laps']:
                return int(float(str(value)))
            # Float for points, time_millis, coordinates
            return float(value)
        except (ValueError, TypeError):
            report.data_type_issues.append(f"Invalid {field}: {value}")
            return None

    def _convert_to_date(self, value: Any, report: DataQualityReport) -> Optional[str]:
        """Convert value to date string in ISO format."""
        if value is None or value == '':
            return None

        if isinstance(value, date):
            return value.isoformat()

        if isinstance(value, str):
            try:
                # Try parsing common date formats
                parsed_date = None
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                    try:
                        parsed_date = datetime.strptime(value, fmt).date()
                        break
                    except ValueError:
                        continue

                if parsed_date:
                    return parsed_date.isoformat()
                report.data_type_issues.append(f"Invalid date format: {value}")
                return None
            except Exception:
                report.data_type_issues.append(f"Invalid date: {value}")
                return None

        report.data_type_issues.append(f"Invalid date type: {type(value)} - {value}")
        return None

    def _convert_to_time(self, value: Any, report: DataQualityReport) -> Optional[str]:
        """Convert value to time string in ISO format."""
        if value is None or value == '':
            return None

        if isinstance(value, dt_time):
            return value.isoformat()

        if isinstance(value, str):
            try:
                # Try parsing common time formats
                parsed_time = None
                for fmt in ['%H:%M:%SZ', '%H:%M:%S', '%H:%M']:
                    try:
                        parsed_time = datetime.strptime(value, fmt).time()
                        break
                    except ValueError:
                        continue

                if parsed_time:
                    return parsed_time.isoformat()
                # Return original if it looks like a time string
                if re.match(r'\d{1,2}:\d{2}', value):
                    return value
                return None
            except Exception:
                return None

        return None

    def _clean_qualifying_times(self, data: Dict[str, Any], report: DataQualityReport) -> Dict[str, Any]:
        """Clean and validate qualifying time formats."""
        cleaned = data.copy()

        time_fields = ['q1', 'q2', 'q3']
        for field in time_fields:
            if field in cleaned and cleaned[field] is not None:
                cleaned[field] = self._normalize_lap_time(cleaned[field], report)

        return cleaned

    def _normalize_lap_time(self, time_str: str, report: DataQualityReport) -> Optional[str]:
        """Normalize lap time string to consistent format."""
        if not time_str or time_str.strip() == '':
            return None

        time_str = str(time_str).strip()

        # Skip if already None or empty
        if time_str.lower() in ['none', 'null', '']:
            return None

        # Check for valid lap time format (e.g., "1:23.456")
        if re.match(r'^\d{1,2}:\d{2}\.\d{3}$', time_str):
            return time_str

        # Try to fix common format issues
        # Remove extra whitespace and normalize
        time_str = re.sub(r'\s+', '', time_str)

        # If it looks like a lap time but format is slightly off
        lap_time_pattern = re.match(r'^(\d{1,2}):(\d{2})\.(\d{1,3})$', time_str)
        if lap_time_pattern:
            minutes, seconds, milliseconds = lap_time_pattern.groups()
            # Ensure milliseconds are 3 digits
            milliseconds = milliseconds.ljust(3, '0')[:3]
            return f"{minutes}:{seconds}.{milliseconds}"

        # If it doesn't match expected format, log as issue but keep original
        report.data_type_issues.append(f"Non-standard lap time format: {time_str}")
        return time_str

    def _validate_race_result(self, data: Dict[str, Any], report: DataQualityReport) -> bool:
        """Validate race result data."""
        required_fields = ['season', 'round', 'driver_id', 'constructor_id']

        for field in required_fields:
            if field not in data or data[field] is None:
                report.validation_errors.append(f"Missing required field: {field}")
                return False

        # Validate data ranges
        if data.get('season'):
            if not (1950 <= int(data['season']) <= datetime.now().year + 1):
                report.validation_errors.append(f"Invalid season: {data['season']}")
                return False

        if data.get('position') is not None:
            if not (1 <= data['position'] <= 30):
                report.validation_errors.append(f"Invalid position: {data['position']}")
                return False

        if data.get('points') is not None:
            if data['points'] < 0 or data['points'] > 50:
                report.validation_errors.append(f"Invalid points: {data['points']}")
                return False

        return True

    def _validate_qualifying_result(self, data: Dict[str, Any], report: DataQualityReport) -> bool:
        """Validate qualifying result data."""
        required_fields = ['season', 'round', 'driver_id', 'constructor_id', 'position']

        for field in required_fields:
            if field not in data or data[field] is None:
                report.validation_errors.append(f"Missing required field: {field}")
                return False

        # Validate data ranges
        if data.get('season'):
            if not (1950 <= int(data['season']) <= datetime.now().year + 1):
                report.validation_errors.append(f"Invalid season: {data['season']}")
                return False

        if data.get('position'):
            if not (1 <= data['position'] <= 30):
                report.validation_errors.append(f"Invalid qualifying position: {data['position']}")
                return False

        return True

    def _validate_race_schedule(self, data: Dict[str, Any], report: DataQualityReport) -> bool:
        """Validate race schedule data."""
        required_fields = ['season', 'round', 'race_name', 'circuit_id']

        for field in required_fields:
            if field not in data or data[field] is None:
                report.validation_errors.append(f"Missing required field: {field}")
                return False

        # Validate data ranges
        if data.get('season'):
            if not (1950 <= int(data['season']) <= datetime.now().year + 1):
                report.validation_errors.append(f"Invalid season: {data['season']}")
                return False

        if data.get('latitude') is not None:
            if not (-90 <= data['latitude'] <= 90):
                report.validation_errors.append(f"Invalid latitude: {data['latitude']}")
                return False

        if data.get('longitude') is not None:
            if not (-180 <= data['longitude'] <= 180):
                report.validation_errors.append(f"Invalid longitude: {data['longitude']}")
                return False

        return True

    def _calculate_quality_score(self, report: DataQualityReport) -> float:
        """Calculate overall data quality score (0-100)."""
        if report.total_records == 0:
            return 0.0

        # Base score
        score = 100.0

        # Deduct for missing values (max 20 points)
        total_missing = sum(report.missing_values.values())
        missing_percent = (total_missing / report.total_records) * 100
        score -= min(20, missing_percent * 4)  # 4 points per percent missing

        # Deduct for data type issues (max 30 points)
        type_issues_percent = (len(report.data_type_issues) / report.total_records) * 100
        score -= min(30, type_issues_percent * 10)  # 10 points per percent with type issues

        # Deduct for validation errors (max 40 points)
        validation_errors_percent = (len(report.validation_errors) / report.total_records) * 100
        score -= min(40, validation_errors_percent * 20)  # 20 points per percent with errors

        # Add points for standardization (up to 10 points)
        total_standardizations = sum(len(changes) for changes in report.standardization_changes.values())
        if total_standardizations > 0:
            score += min(10, (total_standardizations / report.total_records) * 100)

        return max(0.0, score)

    def _init_driver_mappings(self):
        """Initialize driver name standardization mappings."""
        # These mappings ensure consistent driver names across seasons
        # Based on common variations and historical data
        self.driver_name_mappings = {
            'hamilton': 'Lewis Hamilton',
            'verstappen': 'Max Verstappen',
            'leclerc': 'Charles Leclerc',
            'sainz': 'Carlos Sainz Jr.',
            'russell': 'George Russell',
            'perez': 'Sergio Pérez',
            'norris': 'Lando Norris',
            'alonso': 'Fernando Alonso',
            'ocon': 'Esteban Ocon',
            'gasly': 'Pierre Gasly',
            'vettel': 'Sebastian Vettel',
            'stroll': 'Lance Stroll',
            'ricciardo': 'Daniel Ricciardo',
            'tsunoda': 'Yuki Tsunoda',
            'bottas': 'Valtteri Bottas',
            'zhou': 'Zhou Guanyu',
            'magnussen': 'Kevin Magnussen',
            'schumacher': 'Mick Schumacher',
            'albon': 'Alexander Albon',
            'latifi': 'Nicholas Latifi',
            'piastri': 'Oscar Piastri',
            'hulkenberg': 'Nico Hülkenberg',
            'sargeant': 'Logan Sargeant',
            'de_vries': 'Nyck de Vries'
        }

    def _init_constructor_mappings(self):
        """Initialize constructor name standardization mappings."""
        # These mappings ensure consistent constructor names across seasons
        self.constructor_name_mappings = {
            'mercedes': 'Mercedes',
            'red_bull': 'Red Bull Racing',
            'ferrari': 'Ferrari',
            'mclaren': 'McLaren',
            'alpine': 'Alpine F1 Team',
            'alphatauri': 'AlphaTauri',
            'aston_martin': 'Aston Martin',
            'williams': 'Williams',
            'alfa': 'Alfa Romeo',
            'haas': 'Haas F1 Team',
            'racing_point': 'Racing Point',  # Historical
            'renault': 'Renault'  # Historical
        }

    def _init_circuit_mappings(self):
        """Initialize circuit name standardization mappings."""
        # These mappings ensure consistent circuit names across seasons
        self.circuit_name_mappings = {
            'silverstone': 'Silverstone Circuit',
            'monaco': 'Circuit de Monaco',
            'monza': 'Autodromo Nazionale di Monza',
            'spa': 'Circuit de Spa-Francorchamps',
            'suzuka': 'Suzuka Circuit',
            'interlagos': 'Autódromo José Carlos Pace',
            'albert_park': 'Albert Park Circuit',
            'bahrain': 'Bahrain International Circuit',
            'imola': 'Autodromo Enzo e Dino Ferrari',
            'barcelona': 'Circuit de Barcelona-Catalunya',
            'red_bull_ring': 'Red Bull Ring',
            'hungaroring': 'Hungaroring',
            'zandvoort': 'Circuit Zandvoort',
            'baku': 'Baku City Circuit',
            'singapore': 'Marina Bay Street Circuit',
            'austin': 'Circuit of the Americas',
            'mexico': 'Autódromo Hermanos Rodríguez',
            'yas_marina': 'Yas Marina Circuit',
            'jeddah': 'Jeddah Corniche Circuit',
            'miami': 'Miami International Autodrome',
            'las_vegas': 'Las Vegas Strip Circuit'
        }


class DataQualityValidator:
    """Validator for checking data quality standards."""

    def __init__(self, strict_mode: bool = False):
        """Initialize validator.

        Args:
            strict_mode: Whether to use strict validation rules
        """
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(__name__)

    def validate_dataset(self, data: List[Dict[str, Any]], data_type: str) -> DataQualityReport:
        """Validate entire dataset quality.

        Args:
            data: List of data dictionaries
            data_type: Type of data ('race_results', 'qualifying_results', 'race_schedules')

        Returns:
            Data quality report
        """
        report = DataQualityReport(total_records=len(data))

        if not data:
            report.validation_errors.append("Empty dataset")
            return report

        # Check for duplicate records
        self._check_duplicates(data, report, data_type)

        # Check data consistency
        self._check_consistency(data, report, data_type)

        # Check completeness
        self._check_completeness(data, report, data_type)

        # Calculate final quality score
        report.quality_score = self._calculate_dataset_quality_score(data, report)

        return report

    def _check_duplicates(self, data: List[Dict[str, Any]], report: DataQualityReport, data_type: str):
        """Check for duplicate records."""
        if data_type == 'race_results':
            # Check for duplicate results (same season, round, driver)
            seen = set()
            for item in data:
                key = (item.get('season'), item.get('round'), item.get('driver_id'))
                if key in seen:
                    report.validation_errors.append(f"Duplicate race result: {key}")
                seen.add(key)

        elif data_type == 'qualifying_results':
            # Check for duplicate qualifying results
            seen = set()
            for item in data:
                key = (item.get('season'), item.get('round'), item.get('driver_id'))
                if key in seen:
                    report.validation_errors.append(f"Duplicate qualifying result: {key}")
                seen.add(key)

    def _check_consistency(self, data: List[Dict[str, Any]], report: DataQualityReport, data_type: str):
        """Check data consistency across records."""
        if data_type == 'race_results':
            # Check that positions are consistent within each race
            races = {}
            for item in data:
                race_key = (item.get('season'), item.get('round'))
                if race_key not in races:
                    races[race_key] = []
                if item.get('position') is not None:
                    races[race_key].append(item.get('position'))

            for race_key, positions in races.items():
                if len(set(positions)) != len(positions):
                    report.validation_errors.append(f"Duplicate positions in race {race_key}")

    def _check_completeness(self, data: List[Dict[str, Any]], report: DataQualityReport, data_type: str):
        """Check data completeness."""
        required_fields = {
            'race_results': ['season', 'round', 'driver_id', 'constructor_id'],
            'qualifying_results': ['season', 'round', 'driver_id', 'constructor_id', 'position'],
            'race_schedules': ['season', 'round', 'race_name', 'circuit_id']
        }

        if data_type not in required_fields:
            return

        fields = required_fields[data_type]
        missing_counts = {field: 0 for field in fields}

        for item in data:
            for field in fields:
                if field not in item or item[field] is None or item[field] == '':
                    missing_counts[field] += 1

        # Add missing counts to report
        for field, count in missing_counts.items():
            if count > 0:
                report.missing_values[f"{data_type}.{field}"] = count

    def _calculate_dataset_quality_score(self, data: List[Dict[str, Any]], report: DataQualityReport) -> float:
        """Calculate overall dataset quality score."""
        if not data:
            return 0.0

        score = 100.0
        total_records = len(data)

        # Deduct for validation errors
        error_rate = len(report.validation_errors) / total_records
        score -= error_rate * 50  # 50 points max deduction for errors

        # Deduct for missing values
        total_missing = sum(report.missing_values.values())
        missing_rate = total_missing / (total_records * 10)  # Assume avg 10 fields per record
        score -= missing_rate * 30  # 30 points max deduction for missing data

        return max(0.0, min(100.0, score))
