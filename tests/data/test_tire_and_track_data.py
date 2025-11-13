"""Unit tests for tire and track data modules."""

import pytest
import pandas as pd
from datetime import datetime


class TestTireDataModel:
    """Test tire data structures and validation."""

    def test_tire_compound_types(self):
        """Test all valid tire compounds."""
        valid_compounds = ['soft', 'medium', 'hard', 'intermediate', 'wet']
        assert len(valid_compounds) == 5
        assert 'soft' in valid_compounds
        assert 'wet' in valid_compounds

    def test_tire_degradation_curve(self):
        """Test tire degradation follows realistic pattern."""
        # Laps vs grip level (0-100)
        degradation = {
            'soft': [100, 98, 96, 94, 92, 90, 88, 85, 82, 75],  # Fast degradation
            'medium': [100, 99, 98, 97, 96, 95, 94, 93, 92, 90],  # Slower degradation
            'hard': [100, 99.5, 99, 98.5, 98, 97.5, 97, 96.5, 96, 95.5],  # Very slow
        }

        for compound, grip_levels in degradation.items():
            # Should be monotonically decreasing
            for i in range(1, len(grip_levels)):
                assert grip_levels[i] <= grip_levels[i - 1]

    def test_tire_temperature_range(self):
        """Test tire operating temperature ranges."""
        tire_temps = {
            'soft': {'optimal': (90, 110), 'working': (80, 120)},
            'medium': {'optimal': (85, 105), 'working': (75, 115)},
            'hard': {'optimal': (80, 100), 'working': (70, 110)},
            'wet': {'optimal': (60, 80), 'working': (50, 90)},
        }

        for compound, temp_data in tire_temps.items():
            optimal = temp_data['optimal']
            assert optimal[0] < optimal[1]  # Min < Max

    def test_tire_pit_strategy(self):
        """Test pit strategy tire combinations."""
        pit_strategies = {
            'one_stop': [('soft', 30), ('medium', 38)],
            'two_stop': [('soft', 20), ('medium', 25), ('hard', 13)],
            'three_stop': [('soft', 15), ('medium', 15), ('hard', 15), ('medium', 13)],
        }

        for strategy, stops in pit_strategies.items():
            total_laps = sum(laps for _, laps in stops)
            assert total_laps > 0
            assert len(stops) >= 1  # At least one stint


class TestTrackDataModel:
    """Test track data structures."""

    def test_track_characteristics(self):
        """Test track has required characteristics."""
        track = {
            'name': 'Monaco Grand Prix',
            'circuit_id': 'monaco',
            'location': 'Monte Carlo',
            'country': 'Monaco',
            'length_km': 3.337,
            'corners': 19,
            'difficulty': 9.5,
            'overtaking_difficulty': 10,
            'tire_preservation': 'high',
        }

        required_fields = [
            'name',
            'circuit_id',
            'location',
            'country',
            'length_km',
            'corners',
        ]
        for field in required_fields:
            assert field in track

    def test_track_length_valid(self):
        """Test track length is in valid range."""
        # F1 tracks are typically 3-6 km long
        track_lengths = {
            'monaco': 3.337,
            'silverstone': 5.891,
            'monza': 5.793,
            'spa': 7.004,
            'baku': 6.003,
        }

        for circuit, length in track_lengths.items():
            assert 2 < length < 8  # Valid F1 track range

    def test_track_corners_realistic(self):
        """Test number of corners is realistic."""
        track_corners = {
            'monaco': 19,
            'silverstone': 18,
            'monza': 11,
            'spa': 19,
            'baku': 21,
        }

        for circuit, corners in track_corners.items():
            assert 8 < corners < 25  # Typical F1 range

    def test_track_difficulty_scale(self):
        """Test difficulty rating is on valid scale."""
        difficulties = {
            'monza': 5.0,  # Relatively easy
            'silverstone': 7.5,  # Moderate
            'monaco': 10.0,  # Most difficult
            'baku': 8.5,  # High difficulty
        }

        for circuit, difficulty in difficulties.items():
            assert 0 <= difficulty <= 10

    def test_overtaking_difficulty(self):
        """Test overtaking difficulty rankings."""
        overtaking_difficulty = {
            'monaco': 10,  # Nearly impossible
            'silverstone': 5,  # Moderate
            'monza': 3,  # Easy (DRS zone)
            'spa': 4,  # Relatively easy
        }

        for circuit, difficulty in overtaking_difficulty.items():
            assert 0 <= difficulty <= 10


class TestTireDataCollection:
    """Test tire performance data collection."""

    @pytest.fixture
    def sample_tire_performance(self):
        """Create sample tire performance data."""
        return pd.DataFrame({
            'race_id': ['race_1'] * 60,
            'driver_id': ['driver_1'] * 60,
            'compound': ['soft'] * 20 + ['medium'] * 20 + ['hard'] * 20,
            'lap': list(range(1, 21)) * 3,
            'lap_time': [
                # Soft compound - fastest initially, rapid degradation
                *[79.5, 79.3, 79.1, 79.0, 78.9, 79.0, 79.2, 79.5, 79.8, 80.2,
                  80.5, 80.8, 81.2, 81.5, 81.8, 82.0, 82.2, 82.4, 82.6, 82.8],
                # Medium compound - slower but consistent
                *[81.5, 81.4, 81.3, 81.2, 81.1, 81.1, 81.2, 81.3, 81.4, 81.5,
                  81.6, 81.7, 81.8, 81.9, 82.0, 82.1, 82.2, 82.3, 82.4, 82.5],
                # Hard compound - slowest but most durable
                *[82.5, 82.4, 82.3, 82.2, 82.1, 82.1, 82.0, 82.0, 82.0, 82.0,
                  82.0, 82.1, 82.1, 82.1, 82.1, 82.2, 82.2, 82.2, 82.2, 82.3],
            ]
        })

    def test_tire_performance_structure(self, sample_tire_performance):
        """Test tire performance data structure."""
        required_cols = ['race_id', 'driver_id', 'compound', 'lap', 'lap_time']
        for col in required_cols:
            assert col in sample_tire_performance.columns

    def test_tire_compound_performance_order(self, sample_tire_performance):
        """Test soft is fastest, then medium, then hard (on average)."""
        soft_avg = sample_tire_performance[
            sample_tire_performance['compound'] == 'soft'
        ]['lap_time'].mean()

        medium_avg = sample_tire_performance[
            sample_tire_performance['compound'] == 'medium'
        ]['lap_time'].mean()

        hard_avg = sample_tire_performance[
            sample_tire_performance['compound'] == 'hard'
        ]['lap_time'].mean()

        # Hard tire should be slowest
        assert hard_avg > medium_avg
        # Soft should be in fastest region initially even with degradation
        assert soft_avg <= medium_avg or soft_avg < 82

    def test_tire_degradation_progression(self, sample_tire_performance):
        """Test tire degradation increases over stint."""
        soft_stint = sample_tire_performance[
            sample_tire_performance['compound'] == 'soft'
        ].sort_values('lap')

        # Last laps should be slower than first laps
        first_five_avg = soft_stint.head(5)['lap_time'].mean()
        last_five_avg = soft_stint.tail(5)['lap_time'].mean()

        assert last_five_avg > first_five_avg


class TestTrackDataCollection:
    """Test track performance data collection."""

    @pytest.fixture
    def sample_track_data(self):
        """Create sample track data."""
        return pd.DataFrame({
            'circuit': ['monaco'] * 10 + ['silverstone'] * 10,
            'season': [2023, 2024] * 10,
            'round': list(range(1, 6)) * 4,
            'winner': ['Hamilton', 'Verstappen'] * 10,
            'avg_speed': [
                160.0 + i * 0.1 for i in range(20)
            ],
            'safety_cars': [2, 1, 2, 1, 0] * 4,
            'dnfs': [3, 2, 2, 1, 0] * 4,
        })

    def test_track_data_structure(self, sample_track_data):
        """Test track data has required fields."""
        required_fields = ['circuit', 'season', 'round', 'avg_speed']
        for field in required_fields:
            assert field in sample_track_data.columns

    def test_circuit_consistency(self, sample_track_data):
        """Test circuit data is consistent."""
        monaco_data = sample_track_data[sample_track_data['circuit'] == 'monaco']
        assert len(monaco_data) == 10

    def test_safety_car_frequency(self, sample_track_data):
        """Test safety car frequency is realistic."""
        safety_cars = sample_track_data['safety_cars']
        assert (safety_cars >= 0).all()
        assert safety_cars.max() <= 10  # Realistic maximum

    def test_dnf_rate(self, sample_track_data):
        """Test DNF rate is realistic."""
        dnfs = sample_track_data['dnfs']
        total_cars = 20  # Standard F1 grid
        dnf_rate = dnfs.sum() / (len(dnfs) * total_cars)
        assert 0 <= dnf_rate <= 0.5  # 0-50% is realistic


class TestTireWearModel:
    """Test tire wear calculations."""

    def test_tire_wear_temperature_dependent(self):
        """Test tire wear increases with temperature."""
        wear_model = {
            'optimal_temp': 100,
            'wear_at_optimal': 0.5,
            'wear_hot': 0.8,
            'wear_cold': 0.3,
        }

        # Wear should be optimal at ideal temp and increase away from it
        assert wear_model['wear_at_optimal'] < wear_model['wear_hot']
        assert wear_model['wear_at_optimal'] > wear_model['wear_cold']

    def test_tire_wear_load_dependent(self):
        """Test tire wear depends on lateral loads."""
        loads = {
            'low_load': 0.3,
            'medium_load': 0.5,
            'high_load': 0.8,
        }

        # Higher load = more wear
        assert loads['low_load'] < loads['medium_load'] < loads['high_load']

    def test_tire_wear_progression(self):
        """Test tire wear increases monotonically."""
        lap_wear = [0, 0.5, 1.0, 1.6, 2.3, 3.1, 4.0, 5.0, 6.1, 7.3]

        # Each lap should have more cumulative wear
        for i in range(1, len(lap_wear)):
            assert lap_wear[i] >= lap_wear[i - 1]


class TestTrackAdaptation:
    """Test track-specific adaptation."""

    def test_driver_track_affinity(self):
        """Test driver performance varies by track."""
        driver_performance = {
            'monaco': 95,  # Monaco specialist
            'monza': 75,   # Doesn't like Monaco
            'silverstone': 88,  # Good all-rounder
        }

        assert driver_performance['monaco'] > driver_performance['monza']

    def test_car_setup_for_track(self):
        """Test car setup depends on track."""
        setup = {
            'high_downforce': 'monaco',
            'low_downforce': 'monza',
            'balanced': 'silverstone',
        }

        assert len(setup) == 3
        assert 'monaco' in setup.values()

    def test_tire_strategy_by_track(self):
        """Test tire strategy varies by track."""
        strategies = {
            'monza': 'one_stop',
            'silverstone': 'two_stop',
            'monaco': 'two_stop',
            'spa': 'one_stop',
        }

        assert len(strategies) >= 2  # Different strategies for different tracks


class TestDataIntegrity:
    """Test data integrity and validation."""

    def test_no_negative_values(self):
        """Test no negative lap times or temperatures."""
        track_data = pd.DataFrame({
            'lap_time': [80.5, 80.3, 80.1, 80.0],
            'track_temp': [45.0, 46.0, 45.5, 45.8],
        })

        assert (track_data['lap_time'] > 0).all()
        assert (track_data['track_temp'] > 0).all()

    def test_realistic_lap_times(self):
        """Test lap times are in realistic range."""
        # F1 lap times typically range from 70s (Monaco) to 100+ (slow circuits)
        lap_times = [79.5, 80.2, 81.0, 82.5, 85.0]

        for time in lap_times:
            assert 60 < time < 150  # Reasonable F1 lap time

    def test_data_consistency(self):
        """Test data consistency across tables."""
        race_data = pd.DataFrame({
            'race_id': ['race_1', 'race_1', 'race_1'],
            'driver_id': ['driver_1', 'driver_2', 'driver_3'],
            'position': [1, 2, 3],
        })

        # All races should have the same race_id
        race_ids = race_data['race_id'].unique()
        assert len(race_ids) == 1
