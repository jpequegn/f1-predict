"""Tests for enhanced CLI with Click and Rich."""

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
import pytest

from f1_predict.cli_enhanced import cli


@pytest.fixture
def runner():
    """Create Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_data_dir(tmp_path):
    """Create mock data directory structure."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)

    # Create sample data files
    race_results = [{"season": 2024, "round": 1, "driver": "Verstappen", "position": 1}]
    with open(raw_dir / "race_results.json", "w") as f:
        json.dump(race_results, f)

    qualifying_results = [
        {"season": 2024, "round": 1, "driver": "Verstappen", "position": 1}
    ]
    with open(raw_dir / "qualifying_results.json", "w") as f:
        json.dump(qualifying_results, f)

    return data_dir


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_cli_help(self, runner):
        """Test that CLI help works."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "F1 Race Prediction System" in result.output
        assert "predict" in result.output
        assert "data" in result.output

    def test_cli_version(self, runner):
        """Test version flag."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_cli_verbose_flag(self, runner):
        """Test verbose flag is passed to context."""
        result = runner.invoke(cli, ["--verbose", "--help"])
        assert result.exit_code == 0


class TestPredictCommand:
    """Test predict command functionality."""

    def test_predict_help(self, runner):
        """Test predict command help."""
        result = runner.invoke(cli, ["predict", "--help"])
        assert result.exit_code == 0
        assert "Predict race results" in result.output
        assert "--race" in result.output
        assert "--model" in result.output
        assert "--format" in result.output

    def test_predict_basic(self, runner):
        """Test basic prediction with table output."""
        result = runner.invoke(cli, ["predict", "--race", "Monaco"])
        assert result.exit_code == 0
        assert "Generating predictions" in result.output
        assert "Monaco" in result.output
        assert "Race Predictions" in result.output

    def test_predict_with_model(self, runner):
        """Test prediction with specific model."""
        result = runner.invoke(
            cli, ["predict", "--race", "Monaco", "--model", "xgboost"]
        )
        assert result.exit_code == 0
        assert "xgboost" in result.output

    def test_predict_json_output(self, runner):
        """Test prediction with JSON output format."""
        result = runner.invoke(
            cli,
            [
                "predict",
                "--race",
                "Monaco",
                "--model",
                "ensemble",
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0

        # Parse JSON output
        output_data = json.loads(result.output)
        assert "race" in output_data
        assert "model" in output_data
        assert "predictions" in output_data
        assert output_data["race"] == "Monaco"
        assert output_data["model"] == "ensemble"

    def test_predict_csv_output(self, runner):
        """Test prediction with CSV output format."""
        result = runner.invoke(
            cli,
            ["predict", "--race", "Monaco", "--format", "csv"],
        )
        assert result.exit_code == 0
        assert "position,driver,team,confidence" in result.output

    def test_predict_top_n(self, runner):
        """Test prediction with limited results."""
        result = runner.invoke(
            cli,
            ["predict", "--race", "Monaco", "--top-n", "3"],
        )
        assert result.exit_code == 0
        # Should show 3 predictions in table


class TestDataCommands:
    """Test data management commands."""

    def test_data_help(self, runner):
        """Test data command help."""
        result = runner.invoke(cli, ["data", "--help"])
        assert result.exit_code == 0
        assert "Manage F1 data" in result.output
        assert "update" in result.output
        assert "show" in result.output
        assert "stats" in result.output

    @patch("f1_predict.cli_enhanced.F1DataCollector")
    def test_data_update_all(self, mock_collector_class, runner, tmp_path):
        """Test updating all data types."""
        # Mock collector
        mock_collector = MagicMock()
        mock_collector.collect_all_data.return_value = {
            "race_results": "data/race_results.json",
            "qualifying_results": "data/qualifying_results.json",
            "race_schedules": "data/race_schedules.json",
        }
        mock_collector_class.return_value = mock_collector

        result = runner.invoke(
            cli, ["data", "update", "--type", "all", "--data-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "Updating" in result.output
        assert "Data Collection Results" in result.output
        mock_collector.collect_all_data.assert_called_once()

    @patch("f1_predict.cli_enhanced.F1DataCollector")
    def test_data_update_race_results(self, mock_collector_class, runner, tmp_path):
        """Test updating race results only."""
        mock_collector = MagicMock()
        mock_collector.collect_race_results.return_value = "data/race_results.json"
        mock_collector_class.return_value = mock_collector

        result = runner.invoke(
            cli,
            ["data", "update", "--type", "race-results", "--data-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        mock_collector.collect_race_results.assert_called_once()

    def test_data_show(self, runner, mock_data_dir):
        """Test showing data."""
        result = runner.invoke(
            cli,
            [
                "data",
                "show",
                "--type",
                "race-results",
                "--data-dir",
                str(mock_data_dir),
            ],
        )
        assert result.exit_code == 0
        assert "Data Summary" in result.output
        assert "Sample Data" in result.output

    def test_data_show_with_season(self, runner, mock_data_dir):
        """Test showing data filtered by season."""
        result = runner.invoke(
            cli,
            [
                "data",
                "show",
                "--type",
                "race-results",
                "--season",
                "2024",
                "--data-dir",
                str(mock_data_dir),
            ],
        )
        assert result.exit_code == 0
        assert "2024" in result.output

    def test_data_show_missing_file(self, runner, tmp_path):
        """Test showing data when file doesn't exist."""
        result = runner.invoke(
            cli,
            [
                "data",
                "show",
                "--type",
                "race-results",
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_data_stats(self, runner, mock_data_dir):
        """Test data statistics command."""
        result = runner.invoke(cli, ["data", "stats", "--data-dir", str(mock_data_dir)])
        assert result.exit_code == 0
        assert "Data Collection Summary" in result.output
        assert "Race Results" in result.output
        assert "Qualifying Results" in result.output

    def test_data_stats_empty_dir(self, runner, tmp_path):
        """Test data stats with empty directory."""
        data_dir = tmp_path / "empty_data"
        data_dir.mkdir()

        result = runner.invoke(cli, ["data", "stats", "--data-dir", str(data_dir)])
        assert result.exit_code == 0
        assert "No data collected" in result.output


class TestDocsCommand:
    """Test documentation command."""

    def test_docs_help(self, runner):
        """Test docs command help."""
        result = runner.invoke(cli, ["docs", "--help"])
        assert result.exit_code == 0
        assert "documentation" in result.output.lower()

    def test_docs_general(self, runner):
        """Test general documentation."""
        result = runner.invoke(cli, ["docs"])
        assert result.exit_code == 0
        assert "Documentation" in result.output
        assert "predict" in result.output
        assert "data" in result.output
        assert "analyze" in result.output
        assert "model" in result.output

    def test_docs_predict_topic(self, runner):
        """Test predict topic documentation."""
        result = runner.invoke(cli, ["docs", "predict"])
        assert result.exit_code == 0
        assert "Predict Command Examples" in result.output
        assert "Monaco" in result.output

    def test_docs_data_topic(self, runner):
        """Test data topic documentation."""
        result = runner.invoke(cli, ["docs", "data"])
        assert result.exit_code == 0
        assert "data update" in result.output
        assert "data show" in result.output

    def test_docs_unknown_topic(self, runner):
        """Test docs with unknown topic."""
        result = runner.invoke(cli, ["docs", "unknown"])
        assert result.exit_code == 0
        assert "Unknown topic" in result.output


class TestCLIIntegration:
    """Integration tests for CLI."""

    @patch("f1_predict.cli_enhanced.F1DataCollector")
    def test_full_workflow(self, mock_collector_class, runner, tmp_path):
        """Test a full workflow: update data, show data, predict."""
        # Mock collector
        mock_collector = MagicMock()
        mock_collector.collect_all_data.return_value = {
            "race_results": str(tmp_path / "race_results.json")
        }
        mock_collector_class.return_value = mock_collector

        # Step 1: Update data
        result = runner.invoke(
            cli, ["data", "update", "--type", "all", "--data-dir", str(tmp_path)]
        )
        assert result.exit_code == 0

        # Step 2: Make prediction
        result = runner.invoke(cli, ["predict", "--race", "Monaco"])
        assert result.exit_code == 0

    def test_verbose_mode(self, runner):
        """Test that verbose mode works across commands."""
        result = runner.invoke(cli, ["--verbose", "predict", "--race", "Monaco"])
        assert result.exit_code == 0

    def test_multiple_output_formats(self, runner):
        """Test switching between output formats."""
        # Table format
        result = runner.invoke(
            cli, ["predict", "--race", "Monaco", "--format", "table"]
        )
        assert result.exit_code == 0
        assert "Race Predictions" in result.output

        # JSON format
        result = runner.invoke(cli, ["predict", "--race", "Monaco", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "predictions" in data

        # CSV format
        result = runner.invoke(cli, ["predict", "--race", "Monaco", "--format", "csv"])
        assert result.exit_code == 0
        assert "position,driver" in result.output


class TestErrorHandling:
    """Test error handling in CLI."""

    def test_unknown_command(self, runner):
        """Test unknown command."""
        result = runner.invoke(cli, ["unknown"])
        assert result.exit_code != 0

    def test_invalid_model_choice(self, runner):
        """Test invalid model choice."""
        result = runner.invoke(cli, ["predict", "--model", "invalid_model"])
        assert result.exit_code != 0

    def test_invalid_format_choice(self, runner):
        """Test invalid format choice."""
        result = runner.invoke(cli, ["predict", "--format", "invalid_format"])
        assert result.exit_code != 0

    @patch("f1_predict.cli_enhanced.F1DataCollector")
    def test_data_update_error(self, mock_collector_class, runner, tmp_path):
        """Test error handling during data update."""
        mock_collector = MagicMock()
        mock_collector.collect_all_data.side_effect = Exception("API Error")
        mock_collector_class.return_value = mock_collector

        result = runner.invoke(
            cli, ["data", "update", "--type", "all", "--data-dir", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "Error" in result.output
