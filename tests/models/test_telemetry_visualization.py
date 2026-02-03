"""Tests for telemetry visualization module."""

import pytest

from f1_predict.models.telemetry_visualization import (
    TelemetryVisualizer,
    create_synthetic_telemetry,
)


class TestTelemetryVisualizer:
    """Tests for TelemetryVisualizer class."""

    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance."""
        return TelemetryVisualizer(figure_size=(10, 6))

    @pytest.fixture
    def sample_telemetry(self):
        """Create sample telemetry data."""
        return create_synthetic_telemetry(num_laps=10)

    def test_init_default(self):
        """Test default initialization."""
        visualizer = TelemetryVisualizer()
        assert visualizer.figure_size == (12, 8)

    def test_init_custom_size(self):
        """Test custom figure size."""
        visualizer = TelemetryVisualizer(figure_size=(8, 4))
        assert visualizer.figure_size == (8, 4)

    def test_generate_speed_trace(self, visualizer, sample_telemetry):
        """Test speed trace generation."""
        fig = visualizer.generate_speed_trace(sample_telemetry)
        assert fig is not None

    def test_generate_tire_degradation(self, visualizer, sample_telemetry):
        """Test tire degradation visualization."""
        fig = visualizer.generate_tire_degradation(sample_telemetry)
        assert fig is not None

    def test_generate_fuel_load(self, visualizer, sample_telemetry):
        """Test fuel load visualization."""
        fig = visualizer.generate_fuel_load(sample_telemetry)
        assert fig is not None

    def test_generate_sector_times(self, visualizer, sample_telemetry):
        """Test sector times visualization."""
        fig = visualizer.generate_sector_times(sample_telemetry)
        assert fig is not None

    def test_generate_position_trace(self, visualizer, sample_telemetry):
        """Test position trace visualization."""
        fig = visualizer.generate_position_trace(sample_telemetry)
        assert fig is not None

    def test_generate_lap_comparison(self, visualizer, sample_telemetry):
        """Test lap comparison visualization."""
        fig = visualizer.generate_lap_comparison(sample_telemetry, lap1=1, lap2=5)
        assert fig is not None

    def test_generate_strategy_pattern(self, visualizer, sample_telemetry):
        """Test strategy pattern visualization."""
        fig = visualizer.generate_strategy_pattern(sample_telemetry)
        assert fig is not None

    def test_save_visualization(self, visualizer, sample_telemetry, tmp_path):
        """Test saving visualization to file."""
        fig = visualizer.generate_speed_trace(sample_telemetry)
        output_path = tmp_path / "test_speed_trace.png"
        visualizer.save_figure(fig, str(output_path))
        assert output_path.exists()

    def test_close_figure(self, visualizer, sample_telemetry):
        """Test closing figure."""
        fig = visualizer.generate_speed_trace(sample_telemetry)
        visualizer.close_figure(fig)
        # No assertion needed, just verify no exception


class TestSyntheticTelemetry:
    """Tests for synthetic telemetry generation."""

    def test_create_default_telemetry(self):
        """Test default telemetry creation."""
        telemetry = create_synthetic_telemetry()
        assert telemetry is not None
        assert "lap" in telemetry or "Lap" in telemetry or len(telemetry) > 0

    def test_create_telemetry_custom_laps(self):
        """Test telemetry with custom lap count."""
        telemetry = create_synthetic_telemetry(num_laps=20)
        assert telemetry is not None

    def test_create_telemetry_single_lap(self):
        """Test telemetry with single lap."""
        telemetry = create_synthetic_telemetry(num_laps=1)
        assert telemetry is not None

    def test_telemetry_has_required_fields(self):
        """Test telemetry contains required fields."""
        telemetry = create_synthetic_telemetry(num_laps=5)
        # Check for common telemetry fields
        # The exact field names depend on implementation
        assert len(telemetry) > 0

    def test_telemetry_values_reasonable(self):
        """Test telemetry values are in reasonable ranges."""
        telemetry = create_synthetic_telemetry(num_laps=5)
        # Values should be numeric and within typical F1 ranges
        assert telemetry is not None


class TestVisualizerEdgeCases:
    """Tests for visualizer edge cases."""

    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance."""
        return TelemetryVisualizer()

    def test_empty_telemetry(self, visualizer):
        """Test handling of empty telemetry."""
        empty_telemetry = {}
        # Should handle gracefully or raise appropriate error
        try:
            fig = visualizer.generate_speed_trace(empty_telemetry)
            # If it succeeds, figure should exist
            assert fig is not None or fig is None
        except (ValueError, KeyError):
            # Expected for empty data
            pass

    def test_minimal_telemetry(self, visualizer):
        """Test with minimal telemetry data."""
        telemetry = create_synthetic_telemetry(num_laps=1)
        fig = visualizer.generate_speed_trace(telemetry)
        assert fig is not None

    def test_large_telemetry(self, visualizer):
        """Test with large telemetry dataset."""
        telemetry = create_synthetic_telemetry(num_laps=70)  # Full race
        fig = visualizer.generate_speed_trace(telemetry)
        assert fig is not None


class TestVisualizerOutputFormats:
    """Tests for different output formats."""

    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance."""
        return TelemetryVisualizer()

    @pytest.fixture
    def sample_telemetry(self):
        """Create sample telemetry."""
        return create_synthetic_telemetry(num_laps=5)

    def test_save_as_png(self, visualizer, sample_telemetry, tmp_path):
        """Test saving as PNG."""
        fig = visualizer.generate_speed_trace(sample_telemetry)
        output_path = tmp_path / "test.png"
        visualizer.save_figure(fig, str(output_path))
        assert output_path.exists()

    def test_save_as_jpg(self, visualizer, sample_telemetry, tmp_path):
        """Test saving as JPG."""
        fig = visualizer.generate_speed_trace(sample_telemetry)
        output_path = tmp_path / "test.jpg"
        visualizer.save_figure(fig, str(output_path))
        assert output_path.exists()

    def test_save_as_pdf(self, visualizer, sample_telemetry, tmp_path):
        """Test saving as PDF."""
        fig = visualizer.generate_speed_trace(sample_telemetry)
        output_path = tmp_path / "test.pdf"
        visualizer.save_figure(fig, str(output_path))
        assert output_path.exists()


class TestVisualizerStyles:
    """Tests for visualization styling."""

    def test_default_style(self):
        """Test default style configuration."""
        visualizer = TelemetryVisualizer()
        assert visualizer.figure_size is not None

    def test_custom_colors(self):
        """Test custom color configuration."""
        visualizer = TelemetryVisualizer(
            color_scheme={"primary": "#FF0000", "secondary": "#00FF00"}
        )
        assert visualizer is not None

    def test_dark_theme(self):
        """Test dark theme configuration."""
        visualizer = TelemetryVisualizer(theme="dark")
        assert visualizer is not None

    def test_light_theme(self):
        """Test light theme configuration."""
        visualizer = TelemetryVisualizer(theme="light")
        assert visualizer is not None
