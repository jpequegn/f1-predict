"""Tests for speed trace generator module."""

from pathlib import Path
import tempfile

import pytest

from f1_predict.models.speed_trace_generator import SpeedTraceGenerator


class TestSpeedTraceGeneratorInitialization:
    """Test generator creation and initialization."""

    def test_generator_initialization_default_output_dir(self):
        """Test generator initializes with default output directory."""
        generator = SpeedTraceGenerator()
        assert generator is not None
        assert hasattr(generator, 'output_dir')

    def test_generator_initialization_custom_output_dir(self):
        """Test generator initializes with custom output directory."""
        custom_dir = 'data/custom_traces'
        generator = SpeedTraceGenerator(output_dir=custom_dir)
        assert str(generator.output_dir) == custom_dir

    def test_output_dir_as_pathlib_path(self):
        """Test output_dir is properly converted to Path object."""
        generator = SpeedTraceGenerator(output_dir='data/traces')
        assert isinstance(generator.output_dir, Path)


class TestSpeedTraceGeneratorTraceGeneration:
    """Test speed trace generation functionality."""

    def test_generate_trace_returns_path(self):
        """Test generate_trace returns a path string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)
            lap_data = [200, 205, 208, 210, 209, 207]

            result = generator.generate_trace(
                race_id='test_race_1',
                driver_id='driver_1',
                lap_data=lap_data
            )

            assert isinstance(result, str)
            assert result.endswith('.png')

    def test_generate_trace_creates_file(self):
        """Test that PNG file is actually created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)
            lap_data = [200, 205, 208, 210, 209]

            path = generator.generate_trace(
                race_id='race_001',
                driver_id='driver_99',
                lap_data=lap_data
            )

            assert Path(path).exists()
            assert Path(path).is_file()

    def test_generate_trace_creates_directory_structure(self):
        """Test that race-specific directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)
            lap_data = [200, 205, 210]

            generator.generate_trace(
                race_id='race_2024_01',
                driver_id='driver_001',
                lap_data=lap_data
            )

            race_dir = Path(tmpdir) / 'race_2024_01'
            assert race_dir.exists()
            assert race_dir.is_dir()

    def test_generate_trace_with_single_lap(self):
        """Test generation with minimal data (single lap)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)

            path = generator.generate_trace(
                race_id='race_minimal',
                driver_id='driver_1',
                lap_data=[200.0]
            )

            assert Path(path).exists()

    def test_generate_trace_with_many_laps(self):
        """Test generation with many laps (100+)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)
            lap_data = [200 + i * 0.1 for i in range(120)]

            path = generator.generate_trace(
                race_id='race_long',
                driver_id='driver_1',
                lap_data=lap_data
            )

            assert Path(path).exists()

    def test_generate_trace_with_varying_speeds(self):
        """Test generation with realistic varying speed patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)
            # Simulating speed degradation pattern
            lap_data = [
                220, 219, 218, 217,  # Fast start
                218, 219, 220, 221,  # Middle steady
                215, 210, 205, 200   # End degradation
            ]

            path = generator.generate_trace(
                race_id='race_pattern',
                driver_id='driver_1',
                lap_data=lap_data
            )

            assert Path(path).exists()

    def test_generate_trace_idempotency(self):
        """Test that regenerating same trace recreates file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)
            lap_data = [200, 205, 210]

            path1 = generator.generate_trace(
                race_id='race_idempotent',
                driver_id='driver_1',
                lap_data=lap_data
            )

            # Get file modification time
            mtime1 = Path(path1).stat().st_mtime

            # Wait a bit and regenerate
            import time
            time.sleep(0.1)

            path2 = generator.generate_trace(
                race_id='race_idempotent',
                driver_id='driver_1',
                lap_data=lap_data
            )

            assert path1 == path2
            # File should be overwritten with newer timestamp
            mtime2 = Path(path2).stat().st_mtime
            assert mtime2 >= mtime1


class TestSpeedTraceGeneratorBatchOperations:
    """Test batch generation functionality."""

    def test_generate_batch_returns_dict(self):
        """Test generate_batch returns dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)
            races = [
                {
                    'race_id': 'race_1',
                    'drivers': [
                        {'driver_id': 'd1', 'lap_data': [200, 205, 210]},
                        {'driver_id': 'd2', 'lap_data': [198, 203, 208]}
                    ]
                }
            ]

            result = generator.generate_batch(races)
            assert isinstance(result, dict)

    def test_generate_batch_creates_all_traces(self):
        """Test batch generation creates traces for all drivers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)
            races = [
                {
                    'race_id': 'batch_race_1',
                    'drivers': [
                        {'driver_id': 'driver_1', 'lap_data': [200, 205, 210]},
                        {'driver_id': 'driver_2', 'lap_data': [195, 200, 205]},
                        {'driver_id': 'driver_3', 'lap_data': [205, 210, 215]}
                    ]
                }
            ]

            result = generator.generate_batch(races)

            assert len(result) == 3
            for path in result.values():
                assert Path(path).exists()

    def test_generate_batch_multiple_races(self):
        """Test batch generation across multiple races."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)
            races = [
                {
                    'race_id': f'race_{i}',
                    'drivers': [
                        {'driver_id': f'driver_{j}', 'lap_data': [200 + j*5 for _ in range(5)]}
                        for j in range(3)
                    ]
                }
                for i in range(3)
            ]

            result = generator.generate_batch(races)

            assert len(result) == 9  # 3 races * 3 drivers
            assert all(Path(p).exists() for p in result.values())

    def test_generate_batch_returns_correct_keys(self):
        """Test batch result uses correct key format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)
            races = [
                {
                    'race_id': 'keyed_race',
                    'drivers': [
                        {'driver_id': 'driver_alpha', 'lap_data': [200, 205]}
                    ]
                }
            ]

            result = generator.generate_batch(races)

            assert 'keyed_race_driver_alpha' in result


class TestSpeedTraceGeneratorErrorHandling:
    """Test error handling and edge cases."""

    def test_generate_trace_with_empty_lap_data(self):
        """Test handling of empty lap data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)

            with pytest.raises((ValueError, IndexError)):
                generator.generate_trace(
                    race_id='empty_race',
                    driver_id='driver_1',
                    lap_data=[]
                )

    def test_generate_trace_with_none_lap_data(self):
        """Test handling of None lap data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)

            # None lap_data should raise an error during iteration/conversion
            with pytest.raises((TypeError, ValueError, AttributeError)):
                generator.generate_trace(
                    race_id='none_race',
                    driver_id='driver_1',
                    lap_data=None
                )

    def test_generate_trace_with_invalid_speeds(self):
        """Test handling of non-numeric speeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)

            # Should handle gracefully or raise appropriate error
            with pytest.raises((TypeError, ValueError)):
                generator.generate_trace(
                    race_id='invalid_race',
                    driver_id='driver_1',
                    lap_data=['not_a_number', 'also_invalid']
                )

    def test_generate_trace_with_negative_speeds(self):
        """Test handling of negative speed values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)

            # Should either handle gracefully or validate
            path = generator.generate_trace(
                race_id='negative_race',
                driver_id='driver_1',
                lap_data=[200, -50, 210]  # Invalid negative speed
            )

            # Either created file or raised error
            if path:
                assert Path(path).exists()

    def test_generate_batch_with_empty_races_list(self):
        """Test batch generation with empty races list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)
            result = generator.generate_batch([])
            assert isinstance(result, dict)
            assert len(result) == 0

    def test_generate_batch_with_missing_fields(self):
        """Test batch generation with malformed race data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)
            races = [
                {
                    'race_id': 'incomplete_race',
                    # Missing 'drivers' field
                }
            ]

            with pytest.raises((KeyError, AttributeError)):
                generator.generate_batch(races)


class TestSpeedTraceGeneratorOutput:
    """Test output file properties."""

    def test_output_is_valid_png(self):
        """Test that generated file is a valid PNG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)
            lap_data = [200, 205, 210, 215, 220]

            path = generator.generate_trace(
                race_id='png_test',
                driver_id='driver_1',
                lap_data=lap_data
            )

            # Check PNG magic number
            with open(path, 'rb') as f:
                magic = f.read(4)
                assert magic == b'\x89PNG', "File is not a valid PNG"

    def test_output_file_is_reasonable_size(self):
        """Test that PNG file has reasonable size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)
            lap_data = [200, 205, 210] * 10  # 30 laps

            path = generator.generate_trace(
                race_id='size_test',
                driver_id='driver_1',
                lap_data=lap_data
            )

            file_size = Path(path).stat().st_size
            # PNG should be at least 1KB and less than 10MB
            assert 1024 < file_size < 10 * 1024 * 1024

    def test_multiple_races_create_separate_directories(self):
        """Test that different races get separate directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)

            generator.generate_trace('race_a', 'driver_1', [200, 205])
            generator.generate_trace('race_b', 'driver_1', [200, 205])

            assert (Path(tmpdir) / 'race_a').exists()
            assert (Path(tmpdir) / 'race_b').exists()

    def test_same_driver_different_races_different_files(self):
        """Test same driver in different races creates different files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)

            path_a = generator.generate_trace('race_a', 'driver_1', [200, 205])
            path_b = generator.generate_trace('race_b', 'driver_1', [210, 215])

            assert path_a != path_b
            assert Path(path_a).parent != Path(path_b).parent
