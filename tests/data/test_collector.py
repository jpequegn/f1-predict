"""Tests for the F1 data collector."""

import json
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import pytest

from f1_predict.data.collector import F1DataCollector
from f1_predict.data.models import Circuit, Constructor, Driver, Location, Race, Result


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_client():
    """Create a mock Ergast API client."""
    return Mock()


@pytest.fixture
def sample_location():
    """Create a sample location for testing."""
    return Location(lat=52.0786, long=-1.0169, locality="Silverstone", country="UK")


@pytest.fixture
def sample_circuit(sample_location):
    """Create a sample circuit for testing."""
    return Circuit(
        circuit_id="silverstone",
        url="http://example.com",
        circuit_name="Silverstone Circuit",
        location=sample_location,
    )


@pytest.fixture
def sample_driver():
    """Create a sample driver for testing."""
    return Driver(
        driver_id="hamilton",
        permanent_number=44,
        code="HAM",
        url="http://example.com",
        given_name="Lewis",
        family_name="Hamilton",
        date_of_birth="1985-01-07",
        nationality="British",
    )


@pytest.fixture
def sample_constructor():
    """Create a sample constructor for testing."""
    return Constructor(
        constructor_id="mercedes",
        url="http://example.com",
        name="Mercedes",
        nationality="German",
    )


@pytest.fixture
def sample_race(sample_circuit):
    """Create a sample race for testing."""
    return Race(
        season="2023",
        round="1",
        url="http://example.com",
        race_name="British Grand Prix",
        circuit=sample_circuit,
        date="2023-07-09",
    )


@pytest.fixture
def sample_result(sample_driver, sample_constructor):
    """Create a sample race result for testing."""
    return Result(
        number=44,
        position=1,
        position_text="1",
        points=25.0,
        driver=sample_driver,
        constructor=sample_constructor,
        grid=1,
        laps=70,
        status="Finished",
    )


class TestF1DataCollector:
    """Tests for the F1DataCollector class."""

    def test_collector_initialization(self, temp_data_dir):
        """Test collector initialization."""
        collector = F1DataCollector(data_dir=temp_data_dir)

        assert collector.data_dir == Path(temp_data_dir)
        assert collector.raw_dir == Path(temp_data_dir) / "raw"
        assert collector.processed_dir == Path(temp_data_dir) / "processed"
        assert collector.seasons == [2020, 2021, 2022, 2023, 2024]

        # Check directories are created
        assert collector.raw_dir.exists()
        assert collector.processed_dir.exists()

    def test_collector_default_data_dir(self):
        """Test collector with default data directory."""
        collector = F1DataCollector()

        assert collector.data_dir == Path("data")
        assert collector.raw_dir == Path("data") / "raw"
        assert collector.processed_dir == Path("data") / "processed"

    @patch("f1_predict.data.collector.ErgastAPIClient")
    def test_collect_race_results(
        self, mock_client_class, temp_data_dir, sample_race, sample_result
    ):
        """Test race results collection."""
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock API responses
        mock_client.get_races.return_value = [sample_race]
        mock_client.get_race_results.return_value = [sample_result]

        collector = F1DataCollector(data_dir=temp_data_dir)
        collector.seasons = [2023]  # Test with single season

        # Collect race results
        output_file = collector.collect_race_results()

        # Verify file was created
        assert Path(output_file).exists()
        assert "race_results_2020_2024.csv" in output_file

        # Verify CSV content
        with open(output_file) as f:
            content = f.read()
            assert "season,round,race_name" in content
            assert "2023,1,British Grand Prix" in content
            assert "hamilton,Lewis Hamilton" in content

        # Verify JSON file was also created
        json_file = Path(temp_data_dir) / "raw" / "race_results_2020_2024.json"
        assert json_file.exists()

    @patch("f1_predict.data.collector.ErgastAPIClient")
    def test_collect_race_results_existing_file(self, mock_client_class, temp_data_dir):
        """Test that existing files are not overwritten unless force_refresh is True."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        collector = F1DataCollector(data_dir=temp_data_dir)

        # Create existing file
        output_file = collector.raw_dir / "race_results_2020_2024.csv"
        output_file.write_text("existing content")

        # Collect without force_refresh
        result_file = collector.collect_race_results(force_refresh=False)

        # Should return existing file without calling API
        assert result_file == str(output_file)
        assert output_file.read_text() == "existing content"
        mock_client.get_races.assert_not_called()

        # Collect with force_refresh
        mock_client.get_races.return_value = []
        collector.collect_race_results(force_refresh=True)

        # Should have called API
        mock_client.get_races.assert_called()

    @patch("f1_predict.data.collector.ErgastAPIClient")
    def test_collect_qualifying_results(
        self, mock_client_class, temp_data_dir, sample_race
    ):
        """Test qualifying results collection."""
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create sample qualifying result
        from f1_predict.data.models import QualifyingResult

        # Create the required driver and constructor
        sample_driver = Driver(
            driver_id="hamilton",
            url="http://example.com",
            given_name="Lewis",
            family_name="Hamilton",
            date_of_birth="1985-01-07",
            nationality="British",
        )

        sample_constructor = Constructor(
            constructor_id="mercedes",
            url="http://example.com",
            name="Mercedes",
            nationality="German",
        )

        qualifying_result = QualifyingResult(
            number=44,
            position=1,
            driver=sample_driver,
            constructor=sample_constructor,
            q1="1:29.123",
            q2="1:28.456",
            q3="1:27.789",
        )

        # Mock API responses
        mock_client.get_races.return_value = [sample_race]
        mock_client.get_qualifying_results.return_value = [qualifying_result]

        collector = F1DataCollector(data_dir=temp_data_dir)
        collector.seasons = [2023]  # Test with single season

        # Collect qualifying results
        output_file = collector.collect_qualifying_results()

        # Verify file was created
        assert Path(output_file).exists()
        assert "qualifying_results_2020_2024.csv" in output_file

        # Verify CSV content
        with open(output_file) as f:
            content = f.read()
            assert "season,round,race_name" in content
            assert "q1,q2,q3" in content
            assert "1:29.123,1:28.456,1:27.789" in content

    @patch("f1_predict.data.collector.ErgastAPIClient")
    def test_collect_race_schedules(
        self, mock_client_class, temp_data_dir, sample_race
    ):
        """Test race schedules collection."""
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock API responses
        mock_client.get_races.return_value = [sample_race]

        collector = F1DataCollector(data_dir=temp_data_dir)
        collector.seasons = [2023]  # Test with single season

        # Collect race schedules
        output_file = collector.collect_race_schedules()

        # Verify file was created
        assert Path(output_file).exists()
        assert "race_schedules_2020_2024.csv" in output_file

        # Verify CSV content
        with open(output_file) as f:
            content = f.read()
            assert "season,round,race_name" in content
            assert "circuit_id,circuit_name" in content
            assert "2023,1,British Grand Prix" in content

    @patch("f1_predict.data.collector.ErgastAPIClient")
    def test_collect_all_data(self, mock_client_class, temp_data_dir):
        """Test collecting all data types."""
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_races.return_value = []

        collector = F1DataCollector(data_dir=temp_data_dir)

        # Collect all data
        results = collector.collect_all_data()

        # Verify all data types are included in results
        expected_keys = [
            "race_results",
            "qualifying_results",
            "race_schedules",
            "lap_times",
            "pit_stops",
        ]
        for key in expected_keys:
            assert key in results

        # Verify lap times and pit stops are marked as not implemented
        assert "Not implemented" in results["lap_times"]
        assert "Not implemented" in results["pit_stops"]

    @patch("f1_predict.data.collector.ErgastAPIClient")
    def test_refresh_data(self, mock_client_class, temp_data_dir):
        """Test data refresh functionality."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_races.return_value = []

        collector = F1DataCollector(data_dir=temp_data_dir)

        # Mock collect_all_data to verify force_refresh=True is passed
        with patch.object(collector, "collect_all_data") as mock_collect:
            mock_collect.return_value = {"test": "result"}

            result = collector.refresh_data()

            mock_collect.assert_called_once_with(force_refresh=True)
            assert result == {"test": "result"}

    def test_get_data_summary(self, temp_data_dir):
        """Test data summary functionality."""
        collector = F1DataCollector(data_dir=temp_data_dir)

        # Create some test files
        (collector.raw_dir / "test_raw.csv").write_text("test raw data")
        (collector.processed_dir / "test_processed.csv").write_text(
            "test processed data"
        )

        # Get summary
        summary = collector.get_data_summary()

        # Verify summary structure
        assert "data_directory" in summary
        assert "raw_files" in summary
        assert "processed_files" in summary
        assert "last_updated" in summary

        # Verify files are listed
        assert "test_raw.csv" in summary["raw_files"]
        assert "test_processed.csv" in summary["processed_files"]

        # Verify file info includes size and modification time
        raw_info = summary["raw_files"]["test_raw.csv"]
        assert "size_bytes" in raw_info
        assert "modified" in raw_info

    def test_save_to_csv(self, temp_data_dir):
        """Test CSV saving functionality."""
        collector = F1DataCollector(data_dir=temp_data_dir)

        test_data = [
            {"name": "Lewis", "team": "Mercedes", "points": 25},
            {"name": "Max", "team": "Red Bull", "points": 18},
        ]

        test_file = collector.raw_dir / "test.csv"
        collector._save_to_csv(test_data, test_file)

        # Verify file was created and contains correct data
        assert test_file.exists()
        content = test_file.read_text()
        assert "name,team,points" in content
        assert "Lewis,Mercedes,25" in content
        assert "Max,Red Bull,18" in content

    def test_save_to_json(self, temp_data_dir):
        """Test JSON saving functionality."""
        collector = F1DataCollector(data_dir=temp_data_dir)

        test_data = [
            {"name": "Lewis", "team": "Mercedes", "points": 25},
            {"name": "Max", "team": "Red Bull", "points": 18},
        ]

        test_file = collector.raw_dir / "test.json"
        collector._save_to_json(test_data, test_file)

        # Verify file was created and contains correct data
        assert test_file.exists()
        with open(test_file) as f:
            loaded_data = json.load(f)

        assert loaded_data == test_data

    def test_save_empty_data(self, temp_data_dir):
        """Test saving empty data."""
        collector = F1DataCollector(data_dir=temp_data_dir)

        test_file = collector.raw_dir / "empty.csv"
        collector._save_to_csv([], test_file)

        # Should not create file for empty data
        assert not test_file.exists()

    @patch("f1_predict.data.collector.ErgastAPIClient")
    def test_api_error_handling(self, mock_client_class, temp_data_dir):
        """Test error handling during API calls."""
        # Setup mock client to raise errors
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_races.side_effect = Exception("API Error")

        collector = F1DataCollector(data_dir=temp_data_dir)
        collector.seasons = [2023]

        # Collect data should handle errors gracefully
        output_file = collector.collect_race_results()

        # Should return expected file path even if no data collected
        assert "race_results_2020_2024.csv" in output_file
        # File should not exist since no data was collected
        assert not Path(output_file).exists()

    def test_context_manager(self, temp_data_dir):
        """Test context manager functionality."""
        with F1DataCollector(data_dir=temp_data_dir) as collector:
            assert collector is not None
            assert hasattr(collector, "client")

        # Context manager should close client
        # Note: The actual client closing is mocked, so we just verify
        # the context manager works without errors
