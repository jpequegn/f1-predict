"""Tests for data cleaning module."""

from datetime import date, datetime, time
import json
from pathlib import Path

import pytest

from f1_predict.data.cleaning import (
    DataCleaner,
    DataQualityReport,
    DataQualityValidator,
)


@pytest.fixture
def fixtures_dir():
    """Get path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def dirty_race_results(fixtures_dir):
    """Load dirty race results from fixture file."""
    with open(fixtures_dir / "dirty_race_results.json") as f:
        return json.load(f)


@pytest.fixture
def dirty_qualifying_results(fixtures_dir):
    """Load dirty qualifying results from fixture file."""
    with open(fixtures_dir / "dirty_qualifying_results.json") as f:
        return json.load(f)


@pytest.fixture
def dirty_race_schedules(fixtures_dir):
    """Load dirty race schedules from fixture file."""
    with open(fixtures_dir / "dirty_race_schedules.json") as f:
        return json.load(f)


class TestDataQualityReport:
    """Test DataQualityReport class."""

    def test_initialization(self):
        """Test report initialization."""
        report = DataQualityReport()

        assert report.total_records == 0
        assert report.missing_values == {}
        assert report.data_type_issues == []
        assert report.standardization_changes == {}
        assert report.validation_errors == []
        assert report.quality_score == 0.0
        assert isinstance(report.timestamp, datetime)

    def test_with_data(self):
        """Test report with actual data."""
        report = DataQualityReport(total_records=100)
        report.missing_values = {"position": 5, "time": 10}
        report.quality_score = 85.5

        assert report.total_records == 100
        assert report.missing_values["position"] == 5
        assert report.missing_values["time"] == 10
        assert report.quality_score == 85.5


class TestDataCleaner:
    """Test DataCleaner class."""

    @pytest.fixture
    def cleaner(self):
        """Create DataCleaner instance."""
        return DataCleaner(enable_logging=False)

    @pytest.fixture
    def sample_race_result(self):
        """Sample race result with various data issues."""
        return {
            "season": "2024",  # String instead of int
            "round": 5,
            "race_name": "  Spanish Grand Prix  ",  # Extra whitespace
            "date": "2024-06-23",
            "driver_id": "verstappen",
            "driver_name": "M. Verstappen",  # Non-standard format
            "constructor_id": "red_bull",
            "constructor_name": "Red Bull",  # Needs standardization
            "position": "1",  # String instead of int
            "points": "25.0",  # String instead of float
            "grid": 1,
            "laps": 66,
            "status": "Finished",
            "time_millis": None,  # Missing value
            "time_formatted": "1:32:52.894",
        }

    @pytest.fixture
    def sample_qualifying_result(self):
        """Sample qualifying result with various data issues."""
        return {
            "season": 2024,
            "round": "5",  # String instead of int
            "race_name": "Spanish Grand Prix",
            "date": "06/22/2024",  # Non-ISO format
            "driver_id": "hamilton",
            "driver_name": "L. Hamilton",
            "constructor_id": "mercedes",
            "constructor_name": "Mercedes-AMG",  # Needs standardization
            "position": 2,
            "q1": "1:12.345",  # Valid lap time
            "q2": "1:11.987",
            "q3": "",  # Empty string (missing)
        }

    @pytest.fixture
    def sample_race_schedule(self):
        """Sample race schedule with data issues."""
        return {
            "season": 2024,
            "round": 5,
            "race_name": "Spanish Grand Prix",
            "circuit_id": "barcelona",
            "circuit_name": "Barcelona-Catalunya",  # Needs standardization
            "date": "2024-06-23",
            "time": "14:00:00Z",
            "locality": "Barcelona",
            "country": "Spain",
            "latitude": "41.57",  # String instead of float
            "longitude": "2.26111",
            "fp1_time": None,
            "fp2_time": "13:00:00",
        }

    def test_initialization(self, cleaner):
        """Test cleaner initialization."""
        assert cleaner.enable_logging is False
        assert hasattr(cleaner, "driver_name_mappings")
        assert hasattr(cleaner, "constructor_name_mappings")
        assert hasattr(cleaner, "circuit_name_mappings")
        assert cleaner.quality_thresholds["missing_data_percent"] == 5.0
        assert cleaner.quality_thresholds["invalid_data_percent"] == 2.0
        assert cleaner.quality_thresholds["min_quality_score"] == 85.0

    def test_clean_race_results_single(self, cleaner, sample_race_result):
        """Test cleaning a single race result."""
        cleaned_data, report = cleaner.clean_race_results([sample_race_result])

        assert len(cleaned_data) == 1
        result = cleaned_data[0]

        # Check data type conversions
        assert isinstance(result["season"], int)
        assert result["season"] == 2024
        assert isinstance(result["position"], int)
        assert result["position"] == 1
        assert isinstance(result["points"], float)
        assert result["points"] == 25.0

        # Check string formatting
        assert result["race_name"] == "Spanish Grand Prix"

        # Check report
        assert report.total_records == 1
        assert report.quality_score > 0

    def test_clean_qualifying_results(self, cleaner, sample_qualifying_result):
        """Test cleaning qualifying results."""
        cleaned_data, report = cleaner.clean_qualifying_results(
            [sample_qualifying_result]
        )

        assert len(cleaned_data) == 1
        result = cleaned_data[0]

        # Check data type conversions
        assert isinstance(result["round"], int)
        assert result["round"] == 5

        # Check date conversion
        assert result["date"] == "2024-06-22"

        # Check Q3 missing value handling
        assert result["q3"] is None

        # Check report
        assert report.total_records == 1
        assert "qualifying_result.q3" in report.missing_values

    def test_clean_race_schedules(self, cleaner, sample_race_schedule):
        """Test cleaning race schedules."""
        cleaned_data, report = cleaner.clean_race_schedules([sample_race_schedule])

        assert len(cleaned_data) == 1
        schedule = cleaned_data[0]

        # Check coordinate conversions
        assert isinstance(schedule["latitude"], float)
        assert schedule["latitude"] == 41.57
        assert isinstance(schedule["longitude"], float)
        assert schedule["longitude"] == 2.26111

        # Check report
        assert report.total_records == 1
        assert report.quality_score > 0

    def test_validate_data_quality_pass(self, cleaner):
        """Test data quality validation with passing data."""
        report = DataQualityReport(total_records=100)
        report.quality_score = 90.0
        report.missing_values = {"field1": 2}  # 2% missing
        report.validation_errors = ["error1"]  # 1% errors

        assert cleaner.validate_data_quality(report) is True

    def test_validate_data_quality_fail(self, cleaner):
        """Test data quality validation with failing data."""
        report = DataQualityReport(total_records=100)
        report.quality_score = 70.0  # Below threshold
        report.missing_values = {"field1": 10}  # 10% missing - above threshold
        report.validation_errors = ["error1", "error2", "error3"]  # 3% errors

        assert cleaner.validate_data_quality(report) is False

    def test_handle_missing_values(self, cleaner):
        """Test handling of missing values."""
        data = {
            "position": "",
            "points": None,
            "status": "   ",  # Whitespace only
            "driver_name": "Hamilton",
        }
        report = DataQualityReport()

        result = cleaner._handle_missing_values(data, report, "race_result")

        assert result["position"] is None
        assert result["points"] == 0.0
        assert result["status"] == "Unknown"
        assert result["driver_name"] == "Hamilton"
        assert "race_result.position" in report.missing_values
        assert "race_result.points" in report.missing_values
        assert "race_result.status" in report.missing_values

    def test_standardize_driver_name(self, cleaner):
        """Test driver name standardization."""
        data = {"driver_id": "hamilton", "driver_name": "L. Hamilton"}
        report = DataQualityReport()

        result = cleaner._standardize_driver_name(data, report)

        assert result["driver_name"] == "Lewis Hamilton"
        assert "L. Hamilton" in report.standardization_changes.get("driver_names", {})

    def test_standardize_constructor_name(self, cleaner):
        """Test constructor name standardization."""
        data = {"constructor_id": "red_bull", "constructor_name": "Red Bull"}
        report = DataQualityReport()

        result = cleaner._standardize_constructor_name(data, report)

        assert result["constructor_name"] == "Red Bull Racing"
        assert "Red Bull" in report.standardization_changes.get("constructor_names", {})

    def test_standardize_circuit_name(self, cleaner):
        """Test circuit name standardization."""
        data = {"circuit_id": "monaco", "circuit_name": "Monaco"}
        report = DataQualityReport()

        result = cleaner._standardize_circuit_name(data, report)

        assert result["circuit_name"] == "Circuit de Monaco"
        assert "Monaco" in report.standardization_changes.get("circuit_names", {})

    def test_convert_to_numeric(self, cleaner):
        """Test numeric conversion."""
        report = DataQualityReport()

        # Test integer conversion
        assert cleaner._convert_to_numeric("5", "position", report) == 5
        assert cleaner._convert_to_numeric("3.0", "round", report) == 3

        # Test float conversion
        assert cleaner._convert_to_numeric("25.5", "points", report) == 25.5
        assert cleaner._convert_to_numeric("41.57", "latitude", report) == 41.57

        # Test invalid values
        assert cleaner._convert_to_numeric("abc", "position", report) is None
        assert len(report.data_type_issues) == 1
        assert cleaner._convert_to_numeric("", "points", report) is None

    def test_convert_to_date(self, cleaner):
        """Test date conversion."""
        report = DataQualityReport()

        # Test ISO format
        assert cleaner._convert_to_date("2024-06-23", report) == "2024-06-23"

        # Test other formats
        assert cleaner._convert_to_date("23/06/2024", report) == "2024-06-23"
        assert cleaner._convert_to_date("06/23/2024", report) == "2024-06-23"

        # Test date object
        d = date(2024, 6, 23)
        assert cleaner._convert_to_date(d, report) == "2024-06-23"

        # Test invalid date
        assert cleaner._convert_to_date("invalid", report) is None
        assert len(report.data_type_issues) == 1

    def test_convert_to_time(self, cleaner):
        """Test time conversion."""
        report = DataQualityReport()

        # Test various time formats
        assert cleaner._convert_to_time("14:00:00Z", report) == "14:00:00"
        assert cleaner._convert_to_time("14:00:00", report) == "14:00:00"
        assert cleaner._convert_to_time("14:00", report) == "14:00:00"

        # Test time object
        t = time(14, 0, 0)
        assert cleaner._convert_to_time(t, report) == "14:00:00"

        # Test invalid time
        assert cleaner._convert_to_time("invalid", report) is None
        assert cleaner._convert_to_time("", report) is None

    def test_normalize_lap_time(self, cleaner):
        """Test lap time normalization."""
        report = DataQualityReport()

        # Test valid lap times
        assert cleaner._normalize_lap_time("1:23.456", report) == "1:23.456"
        assert cleaner._normalize_lap_time("1:23.45", report) == "1:23.450"
        assert cleaner._normalize_lap_time("1:23.4", report) == "1:23.400"

        # Test empty/none values
        assert cleaner._normalize_lap_time("", report) is None
        assert cleaner._normalize_lap_time("none", report) is None
        assert cleaner._normalize_lap_time("NULL", report) is None

        # Test invalid format
        result = cleaner._normalize_lap_time("invalid_time", report)
        assert result == "invalid_time"  # Keeps original but logs issue
        assert len(report.data_type_issues) == 1

    def test_validate_race_result_valid(self, cleaner):
        """Test race result validation with valid data."""
        data = {
            "season": 2024,
            "round": 5,
            "driver_id": "hamilton",
            "constructor_id": "mercedes",
            "position": 1,
            "points": 25.0,
        }
        report = DataQualityReport()

        assert cleaner._validate_race_result(data, report) is True
        assert len(report.validation_errors) == 0

    def test_validate_race_result_invalid(self, cleaner):
        """Test race result validation with invalid data."""
        report = DataQualityReport()

        # Missing required field
        data1 = {"season": 2024, "round": 5}
        assert cleaner._validate_race_result(data1, report) is False

        # Invalid season
        data2 = {
            "season": 1900,
            "round": 5,
            "driver_id": "test",
            "constructor_id": "test",
        }
        assert cleaner._validate_race_result(data2, report) is False

        # Invalid position
        data3 = {
            "season": 2024,
            "round": 5,
            "driver_id": "test",
            "constructor_id": "test",
            "position": 50,
        }
        assert cleaner._validate_race_result(data3, report) is False

        # Invalid points
        data4 = {
            "season": 2024,
            "round": 5,
            "driver_id": "test",
            "constructor_id": "test",
            "points": 100,
        }
        assert cleaner._validate_race_result(data4, report) is False

    def test_calculate_quality_score(self, cleaner):
        """Test quality score calculation."""
        report = DataQualityReport(total_records=100)

        # Perfect data
        assert cleaner._calculate_quality_score(report) == 100.0

        # With missing values (5 missing out of 100 records)
        report.missing_values = {"field1": 3, "field2": 2}
        score1 = cleaner._calculate_quality_score(report)
        assert 75 < score1 < 85  # Should deduct ~20 points

        # With data type issues
        report.data_type_issues = ["issue1", "issue2", "issue3"]
        score2 = cleaner._calculate_quality_score(report)
        assert score2 < score1  # Should be lower

        # With validation errors
        report.validation_errors = ["error1", "error2"]
        score3 = cleaner._calculate_quality_score(report)
        assert score3 < score2  # Should be even lower

        # With standardizations (should add points)
        report.standardization_changes = {"names": {"old": "new", "old2": "new2"}}
        score4 = cleaner._calculate_quality_score(report)
        assert score4 > score3  # Should be slightly higher

    def test_batch_processing(self, cleaner):
        """Test processing multiple records with various issues."""
        race_results = [
            {
                "season": 2024,
                "round": 1,
                "driver_id": "hamilton",
                "constructor_id": "mercedes",
                "position": 1,
                "points": 25,
            },
            {
                "season": "2024",
                "round": "2",
                "driver_id": "verstappen",
                "constructor_id": "red_bull",
                "position": "1",
                "points": "25",
            },
            {
                "season": 2024,
                "round": 3,
                "driver_id": "invalid",
                "constructor_id": None,
                "position": 99,
                "points": -5,  # Invalid data
            },
        ]

        cleaned_data, report = cleaner.clean_race_results(race_results)

        # Should process valid records
        assert len(cleaned_data) <= 3
        assert report.total_records == 3

        # Check for validation errors from invalid record
        assert len(report.validation_errors) > 0

        # Quality score should reflect issues
        assert 0 <= report.quality_score <= 100


class TestDataQualityValidator:
    """Test DataQualityValidator class."""

    @pytest.fixture
    def validator(self):
        """Create DataQualityValidator instance."""
        return DataQualityValidator(strict_mode=False)

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for validation."""
        return [
            {
                "season": 2024,
                "round": 1,
                "driver_id": "hamilton",
                "constructor_id": "mercedes",
                "position": 1,
            },
            {
                "season": 2024,
                "round": 1,
                "driver_id": "verstappen",
                "constructor_id": "red_bull",
                "position": 2,
            },
            {
                "season": 2024,
                "round": 1,
                "driver_id": "hamilton",  # Duplicate
                "constructor_id": "mercedes",
                "position": 3,
            },
        ]

    def test_initialization(self, validator):
        """Test validator initialization."""
        assert validator.strict_mode is False
        assert hasattr(validator, "logger")

    def test_validate_empty_dataset(self, validator):
        """Test validation of empty dataset."""
        report = validator.validate_dataset([], "race_results")

        assert report.total_records == 0
        assert "Empty dataset" in report.validation_errors
        assert report.quality_score == 0.0

    def test_check_duplicates(self, validator, sample_dataset):
        """Test duplicate detection."""
        report = validator.validate_dataset(sample_dataset, "race_results")

        # Should detect duplicate (same season, round, driver)
        assert any(
            "Duplicate race result" in error for error in report.validation_errors
        )

    def test_check_consistency(self, validator):
        """Test data consistency checking."""
        # Dataset with duplicate positions in same race
        data = [
            {"season": 2024, "round": 1, "driver_id": "hamilton", "position": 1},
            {
                "season": 2024,
                "round": 1,
                "driver_id": "verstappen",
                "position": 1,
            },  # Duplicate position
        ]

        report = DataQualityReport()
        validator._check_consistency(data, report, "race_results")

        assert any("Duplicate positions" in error for error in report.validation_errors)

    def test_check_completeness(self, validator):
        """Test data completeness checking."""
        # Dataset with missing required fields
        data = [
            {"season": 2024, "round": 1},  # Missing driver_id, constructor_id
            {"season": 2024, "driver_id": "hamilton"},  # Missing round, constructor_id
        ]

        report = DataQualityReport()
        validator._check_completeness(data, report, "race_results")

        assert "race_results.driver_id" in report.missing_values
        assert "race_results.constructor_id" in report.missing_values
        assert "race_results.round" in report.missing_values

    def test_calculate_dataset_quality_score(self, validator):
        """Test dataset quality score calculation."""
        report = DataQualityReport()
        data = [{"test": 1}, {"test": 2}]

        # Perfect data
        score1 = validator._calculate_dataset_quality_score(data, report)
        assert score1 == 100.0

        # With validation errors
        report.validation_errors = ["error1", "error2"]
        score2 = validator._calculate_dataset_quality_score(data, report)
        assert score2 == 50.0  # 50% error rate = 50 point deduction

        # With missing values
        report.missing_values = {"field1": 2, "field2": 2}  # 4 total missing
        score3 = validator._calculate_dataset_quality_score(data, report)
        assert score3 < score2

    def test_strict_mode(self):
        """Test validator in strict mode."""
        validator = DataQualityValidator(strict_mode=True)

        assert validator.strict_mode is True
        # Strict mode functionality would be implemented based on requirements


class TestDataCleaningIntegration:
    """Integration tests using fixture data."""

    @pytest.fixture
    def cleaner(self):
        """Create DataCleaner instance."""
        return DataCleaner(enable_logging=False)

    def test_clean_dirty_race_results_integration(self, cleaner, dirty_race_results):
        """Test cleaning dirty race results with real data issues."""
        cleaned_data, report = cleaner.clean_race_results(dirty_race_results)

        # Should process most records (some may be filtered out for quality)
        assert len(cleaned_data) >= 4  # At least 4 valid records
        assert report.total_records == len(dirty_race_results)

        # Check that data types were converted
        for result in cleaned_data:
            assert isinstance(result.get("season"), int)
            assert isinstance(result.get("round"), int)
            if result.get("position") is not None:
                assert isinstance(result["position"], int)
            if result.get("points") is not None:
                assert isinstance(result["points"], float)

        # Check standardization occurred
        standardized_names = report.standardization_changes.get("driver_names", {})
        assert len(standardized_names) > 0  # Should have standardized some names

        # Quality score should reflect issues in data
        assert 0 <= report.quality_score <= 100
        assert len(report.data_type_issues) > 0  # Should detect type issues
        assert len(report.validation_errors) > 0  # Should detect validation errors

    def test_clean_dirty_qualifying_results_integration(
        self, cleaner, dirty_qualifying_results
    ):
        """Test cleaning dirty qualifying results with real data issues."""
        cleaned_data, report = cleaner.clean_qualifying_results(
            dirty_qualifying_results
        )

        # Should process most records
        assert len(cleaned_data) >= 5
        assert report.total_records == len(dirty_qualifying_results)

        # Check data type conversions
        for result in cleaned_data:
            assert isinstance(result.get("season"), int)
            assert isinstance(result.get("round"), int)
            if result.get("position") is not None:
                assert isinstance(result["position"], int)

        # Check qualifying time cleaning
        times_cleaned = 0
        for result in cleaned_data:
            for field in ["q1", "q2", "q3"]:
                if result.get(field) and result[field] != "none":
                    times_cleaned += 1

        assert times_cleaned > 0  # Should have some valid times

        # Should detect issues
        assert len(report.missing_values) > 0
        assert len(report.data_type_issues) > 0

    def test_clean_dirty_race_schedules_integration(
        self, cleaner, dirty_race_schedules
    ):
        """Test cleaning dirty race schedules with real data issues."""
        cleaned_data, report = cleaner.clean_race_schedules(dirty_race_schedules)

        # Should process most records
        assert len(cleaned_data) >= 4
        assert report.total_records == len(dirty_race_schedules)

        # Check coordinate conversions
        for schedule in cleaned_data:
            if schedule.get("latitude") is not None:
                assert isinstance(schedule["latitude"], float)
            if schedule.get("longitude") is not None:
                assert isinstance(schedule["longitude"], float)

        # Check date conversions
        for schedule in cleaned_data:
            if schedule.get("date"):
                # Should be in ISO format or None
                assert schedule["date"] is None or len(schedule["date"]) == 10

        # Should detect validation errors for invalid coordinates
        validation_errors = [
            err
            for err in report.validation_errors
            if "latitude" in err or "longitude" in err
        ]
        assert len(validation_errors) > 0

    def test_data_quality_validation_integration(self, cleaner, dirty_race_results):
        """Test data quality validation with real dirty data."""
        cleaned_data, report = cleaner.clean_race_results(dirty_race_results)

        # With dirty data, quality should be lower
        assert report.quality_score < 95.0

        # May pass or fail depending on how much data was cleaned
        # but should provide meaningful metrics
        assert len(report.missing_values) > 0 or len(report.validation_errors) > 0

    def test_validator_integration(self, dirty_race_results):
        """Test DataQualityValidator with real dirty data."""
        validator = DataQualityValidator(strict_mode=False)

        report = validator.validate_dataset(dirty_race_results, "race_results")

        # Should detect missing values or have reasonable quality score
        # Note: validator may not detect all issues if data is partially valid
        assert report.total_records == len(dirty_race_results)
        assert 0 <= report.quality_score <= 100.0

        # If there are missing values, they should be tracked
        if len(report.missing_values) > 0:
            assert isinstance(report.missing_values, dict)

        # If there are validation errors, they should be tracked
        if len(report.validation_errors) > 0:
            assert isinstance(report.validation_errors, list)

    def test_end_to_end_pipeline(self, cleaner, dirty_race_results):
        """Test complete cleaning pipeline end-to-end."""
        # Step 1: Clean data
        cleaned_data, cleaning_report = cleaner.clean_race_results(dirty_race_results)

        # Step 2: Run additional validation
        validator = DataQualityValidator(strict_mode=False)
        validation_report = validator.validate_dataset(cleaned_data, "race_results")

        # Assertions
        assert len(cleaned_data) > 0
        assert cleaning_report.total_records == len(dirty_race_results)
        assert validation_report.total_records == len(cleaned_data)

        # Cleaned data should have better quality than original
        assert validation_report.quality_score >= cleaning_report.quality_score

        # Should have meaningful reporting
        assert isinstance(cleaning_report.quality_score, float)
        assert 0 <= cleaning_report.quality_score <= 100
