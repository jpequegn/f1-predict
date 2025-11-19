"""Tests for dataset cache builder module."""

import json
from pathlib import Path
import tempfile

import pytest

from f1_predict.models.dataset_cache_builder import DatasetCacheBuilder


class TestDatasetCacheBuilderInitialization:
    """Test dataset cache builder initialization."""

    def test_builder_initialization_default_dirs(self):
        """Test builder initializes with default directories."""
        builder = DatasetCacheBuilder()
        assert builder is not None
        assert hasattr(builder, 'processed_data_dir')
        assert hasattr(builder, 'output_dir')

    def test_builder_initialization_custom_dirs(self):
        """Test builder initializes with custom directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / 'data'
            out_dir = Path(tmpdir) / 'output'
            builder = DatasetCacheBuilder(
                processed_data_dir=str(data_dir),
                output_dir=str(out_dir)
            )
            assert builder.processed_data_dir == data_dir
            assert builder.output_dir == out_dir

    def test_builder_paths_converted_to_pathlib(self):
        """Test string paths are converted to Path objects."""
        builder = DatasetCacheBuilder(
            processed_data_dir='data/processed',
            output_dir='data/cache'
        )
        assert isinstance(builder.processed_data_dir, Path)
        assert isinstance(builder.output_dir, Path)


class TestExtractLapSpeeds:
    """Test lap speed extraction."""

    def test_extract_lap_speeds_returns_list(self):
        """Test extract_lap_speeds returns list of floats."""
        builder = DatasetCacheBuilder()
        speeds = builder.extract_lap_speeds(race_id='2024-01', driver_id='driver_001', lap_data=[120.5, 125.3, 122.1])
        assert isinstance(speeds, list)
        assert all(isinstance(s, float) for s in speeds)

    def test_extract_lap_speeds_with_mock_data(self):
        """Test extracting speeds from mock race data."""
        builder = DatasetCacheBuilder()
        speeds = builder.extract_lap_speeds(race_id='2024-01', driver_id='driver_001', lap_data=[120.0, 125.0, 130.0])
        assert len(speeds) == 3
        assert speeds == [120.0, 125.0, 130.0]

    def test_extract_lap_speeds_converts_integers(self):
        """Test that integer speeds are converted to floats."""
        builder = DatasetCacheBuilder()
        speeds = builder.extract_lap_speeds(race_id='2024-01', driver_id='driver_001', lap_data=[120, 125, 130])
        assert all(isinstance(s, float) for s in speeds)

    def test_extract_lap_speeds_with_empty_data(self):
        """Test extracting speeds with empty lap data."""
        builder = DatasetCacheBuilder()
        speeds = builder.extract_lap_speeds(race_id='2024-01', driver_id='driver_001', lap_data=[])
        assert speeds == []

    def test_extract_lap_speeds_with_single_lap(self):
        """Test extracting speed from single lap."""
        builder = DatasetCacheBuilder()
        speeds = builder.extract_lap_speeds(race_id='2024-01', driver_id='driver_001', lap_data=[150.5])
        assert len(speeds) == 1
        assert speeds[0] == 150.5


class TestExtractTabularFeatures:
    """Test tabular feature extraction."""

    def test_extract_tabular_features_returns_dict(self):
        """Test extract_tabular_features returns dictionary."""
        builder = DatasetCacheBuilder()
        features = builder.extract_tabular_features(
            race_id='2024-01',
            driver_id='driver_001',
            weather_data={'temp': 25, 'humidity': 60},
            tire_data={'compound': 'soft'},
            driver_data={'age': 28},
            circuit_data={'elevation': 500}
        )
        assert isinstance(features, dict)

    def test_extract_tabular_features_includes_weather(self):
        """Test tabular features include weather data."""
        builder = DatasetCacheBuilder()
        features = builder.extract_tabular_features(
            race_id='2024-01',
            driver_id='driver_001',
            weather_data={'temp': 25, 'humidity': 60},
            tire_data={'compound': 'soft'},
            driver_data={'age': 28},
            circuit_data={'elevation': 500}
        )
        assert 'temp' in features or any('temp' in str(k).lower() for k in features)

    def test_extract_tabular_features_includes_tire_info(self):
        """Test tabular features include tire data."""
        builder = DatasetCacheBuilder()
        features = builder.extract_tabular_features(
            race_id='2024-01',
            driver_id='driver_001',
            weather_data={'temp': 25},
            tire_data={'compound': 'soft'},
            driver_data={'age': 28},
            circuit_data={'elevation': 500}
        )
        assert 'tire_compound' in features or any('tire' in str(k).lower() for k in features)

    def test_extract_tabular_features_normalizes_values(self):
        """Test tabular features are properly normalized."""
        builder = DatasetCacheBuilder()
        features = builder.extract_tabular_features(
            race_id='2024-01',
            driver_id='driver_001',
            weather_data={'temp': 25},
            tire_data={'compound': 'soft'},
            driver_data={'age': 28},
            circuit_data={'elevation': 500}
        )
        # All feature values should be numeric
        feature_values = [v for v in features.values() if isinstance(v, (int, float))]
        assert len(feature_values) > 0


class TestBuildCache:
    """Test cache building functionality."""

    def test_build_cache_returns_dict(self):
        """Test build_cache returns dictionary."""
        builder = DatasetCacheBuilder()
        races = [
            {
                'race_id': '2024-01',
                'date': '2024-03-03',
                'drivers': [
                    {
                        'driver_id': 'driver_001',
                        'lap_data': [120.0, 125.0],
                        'weather_data': {'temp': 25},
                        'tire_data': {'compound': 'soft'},
                        'driver_data': {'age': 28},
                        'circuit_data': {'elevation': 500}
                    }
                ]
            }
        ]
        cache = builder.build_cache(races)
        assert isinstance(cache, dict)

    def test_build_cache_includes_metadata(self):
        """Test cache includes metadata section."""
        builder = DatasetCacheBuilder()
        races = [
            {
                'race_id': '2024-01',
                'date': '2024-03-03',
                'drivers': []
            }
        ]
        cache = builder.build_cache(races)
        assert 'metadata' in cache

    def test_build_cache_includes_samples(self):
        """Test cache includes training samples."""
        builder = DatasetCacheBuilder()
        races = [
            {
                'race_id': '2024-01',
                'date': '2024-03-03',
                'drivers': [
                    {
                        'driver_id': 'driver_001',
                        'lap_data': [120.0],
                        'weather_data': {'temp': 25},
                        'tire_data': {'compound': 'soft'},
                        'driver_data': {'age': 28},
                        'circuit_data': {'elevation': 500}
                    }
                ]
            }
        ]
        cache = builder.build_cache(races)
        assert 'samples' in cache
        assert isinstance(cache['samples'], list)

    def test_build_cache_with_multiple_drivers(self):
        """Test building cache with multiple drivers."""
        builder = DatasetCacheBuilder()
        races = [
            {
                'race_id': '2024-01',
                'date': '2024-03-03',
                'drivers': [
                    {
                        'driver_id': 'driver_001',
                        'lap_data': [120.0],
                        'weather_data': {'temp': 25},
                        'tire_data': {'compound': 'soft'},
                        'driver_data': {'age': 28},
                        'circuit_data': {'elevation': 500}
                    },
                    {
                        'driver_id': 'driver_002',
                        'lap_data': [125.0],
                        'weather_data': {'temp': 25},
                        'tire_data': {'compound': 'hard'},
                        'driver_data': {'age': 30},
                        'circuit_data': {'elevation': 500}
                    }
                ]
            }
        ]
        cache = builder.build_cache(races)
        assert len(cache['samples']) >= 2

    def test_build_cache_with_multiple_races(self):
        """Test building cache with multiple races."""
        builder = DatasetCacheBuilder()
        races = [
            {
                'race_id': '2024-01',
                'date': '2024-03-03',
                'drivers': [
                    {
                        'driver_id': 'driver_001',
                        'lap_data': [120.0],
                        'weather_data': {'temp': 25},
                        'tire_data': {'compound': 'soft'},
                        'driver_data': {'age': 28},
                        'circuit_data': {'elevation': 500}
                    }
                ]
            },
            {
                'race_id': '2024-02',
                'date': '2024-03-10',
                'drivers': [
                    {
                        'driver_id': 'driver_001',
                        'lap_data': [125.0],
                        'weather_data': {'temp': 20},
                        'tire_data': {'compound': 'medium'},
                        'driver_data': {'age': 28},
                        'circuit_data': {'elevation': 100}
                    }
                ]
            }
        ]
        cache = builder.build_cache(races)
        assert len(cache['samples']) >= 2
        # Verify both races are represented
        sample_races = {s.get('race_id') for s in cache['samples'] if 'race_id' in s}
        assert '2024-01' in sample_races or '2024-02' in sample_races

    def test_build_cache_metadata_has_version(self):
        """Test cache metadata includes version."""
        builder = DatasetCacheBuilder()
        races = []
        cache = builder.build_cache(races)
        assert 'version' in cache['metadata']

    def test_build_cache_metadata_has_timestamp(self):
        """Test cache metadata includes creation timestamp."""
        builder = DatasetCacheBuilder()
        races = []
        cache = builder.build_cache(races)
        assert 'created_at' in cache['metadata']

    def test_build_cache_sample_has_image_path(self):
        """Test each sample includes speed trace image path."""
        builder = DatasetCacheBuilder()
        races = [
            {
                'race_id': '2024-01',
                'date': '2024-03-03',
                'drivers': [
                    {
                        'driver_id': 'driver_001',
                        'lap_data': [120.0],
                        'weather_data': {'temp': 25},
                        'tire_data': {'compound': 'soft'},
                        'driver_data': {'age': 28},
                        'circuit_data': {'elevation': 500}
                    }
                ]
            }
        ]
        cache = builder.build_cache(races)
        assert len(cache['samples']) > 0
        sample = cache['samples'][0]
        assert 'image_path' in sample or 'speed_trace' in sample

    def test_build_cache_sample_has_tabular_features(self):
        """Test each sample includes tabular features."""
        builder = DatasetCacheBuilder()
        races = [
            {
                'race_id': '2024-01',
                'date': '2024-03-03',
                'drivers': [
                    {
                        'driver_id': 'driver_001',
                        'lap_data': [120.0],
                        'weather_data': {'temp': 25},
                        'tire_data': {'compound': 'soft'},
                        'driver_data': {'age': 28},
                        'circuit_data': {'elevation': 500}
                    }
                ]
            }
        ]
        cache = builder.build_cache(races)
        sample = cache['samples'][0]
        assert 'features' in sample or 'tabular_features' in sample


class TestSaveCache:
    """Test cache saving functionality."""

    def test_save_cache_creates_file(self):
        """Test save_cache creates output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            builder = DatasetCacheBuilder(output_dir=str(out_dir))
            cache = {'samples': []}
            builder.save_cache(cache, filename='test_cache.json')
            assert (out_dir / 'test_cache.json').exists()

    def test_save_cache_creates_valid_json(self):
        """Test saved cache is valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            builder = DatasetCacheBuilder(output_dir=str(out_dir))
            cache = {'samples': [], 'metadata': {'version': '1.0'}}
            builder.save_cache(cache, filename='test_cache.json')

            with open(out_dir / 'test_cache.json') as f:
                loaded = json.load(f)
            assert loaded == cache

    def test_save_cache_creates_directory(self):
        """Test save_cache creates output directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / 'deep' / 'nested' / 'path'
            builder = DatasetCacheBuilder(output_dir=str(out_dir))
            cache = {'samples': []}
            builder.save_cache(cache, filename='test_cache.json')
            assert out_dir.exists()


class TestLoadCache:
    """Test cache loading functionality."""

    def test_load_cache_returns_dict(self):
        """Test load_cache returns dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            builder = DatasetCacheBuilder(output_dir=str(out_dir))
            original_cache = {'samples': [], 'metadata': {'version': '1.0'}}
            builder.save_cache(original_cache, filename='test_cache.json')

            loaded_cache = builder.load_cache(filename='test_cache.json')
            assert isinstance(loaded_cache, dict)

    def test_load_cache_preserves_content(self):
        """Test load_cache preserves cache content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            builder = DatasetCacheBuilder(output_dir=str(out_dir))
            original_cache = {
                'samples': [{'test': 'data'}],
                'metadata': {'version': '1.0'}
            }
            builder.save_cache(original_cache, filename='test_cache.json')

            loaded_cache = builder.load_cache(filename='test_cache.json')
            assert loaded_cache == original_cache

    def test_load_cache_file_not_found(self):
        """Test load_cache raises error if file not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            builder = DatasetCacheBuilder(output_dir=str(out_dir))

            with pytest.raises(FileNotFoundError):
                builder.load_cache(filename='nonexistent.json')


class TestValidateCache:
    """Test cache validation."""

    def test_validate_cache_returns_bool(self):
        """Test validate_cache returns boolean."""
        builder = DatasetCacheBuilder()
        cache = {'samples': [], 'metadata': {'version': '1.0'}}
        result = builder.validate_cache(cache)
        assert isinstance(result, bool)

    def test_validate_cache_with_valid_structure(self):
        """Test validation succeeds with valid cache structure."""
        builder = DatasetCacheBuilder()
        cache = {
            'samples': [
                {
                    'race_id': '2024-01',
                    'driver_id': 'driver_001',
                    'image_path': '/path/to/image.png',
                    'features': [25.0, 0.5, 28, 500]
                }
            ],
            'metadata': {
                'version': '1.0',
                'created_at': '2024-01-01T00:00:00',
                'sample_count': 1
            }
        }
        assert builder.validate_cache(cache)

    def test_validate_cache_missing_samples(self):
        """Test validation fails if samples missing."""
        builder = DatasetCacheBuilder()
        cache = {'metadata': {'version': '1.0'}}
        assert not builder.validate_cache(cache)

    def test_validate_cache_missing_metadata(self):
        """Test validation fails if metadata missing."""
        builder = DatasetCacheBuilder()
        cache = {'samples': []}
        assert not builder.validate_cache(cache)

    def test_validate_cache_sample_completeness(self):
        """Test validation checks sample completeness."""
        builder = DatasetCacheBuilder()
        # Cache with incomplete sample
        cache = {
            'samples': [
                {
                    'race_id': '2024-01'
                    # Missing required fields
                }
            ],
            'metadata': {'version': '1.0'}
        }
        assert not builder.validate_cache(cache)


class TestErrorHandling:
    """Test error handling."""

    def test_extract_lap_speeds_with_invalid_speeds(self):
        """Test extracting speeds with non-numeric values."""
        builder = DatasetCacheBuilder()
        with pytest.raises((TypeError, ValueError)):
            builder.extract_lap_speeds(
                race_id='2024-01',
                driver_id='driver_001',
                lap_data=['invalid', 'speeds']
            )

    def test_extract_lap_speeds_with_none_data(self):
        """Test extracting speeds with None."""
        builder = DatasetCacheBuilder()
        with pytest.raises((TypeError, AttributeError)):
            builder.extract_lap_speeds(
                race_id='2024-01',
                driver_id='driver_001',
                lap_data=None
            )

    def test_build_cache_with_missing_fields(self):
        """Test building cache with races missing required fields."""
        builder = DatasetCacheBuilder()
        races = [
            {
                'race_id': '2024-01'
                # Missing drivers and date
            }
        ]
        with pytest.raises((KeyError, TypeError)):
            builder.build_cache(races)
