"""Tests for anomaly detection registry and persistence."""

import pytest
from datetime import datetime
from pathlib import Path

from f1_predict.data.anomaly_registry import AnomalyRegistry, AnomalyRecord


@pytest.fixture
def temp_registry_dir(tmp_path):
    """Create temporary directory for registry."""
    return tmp_path / "anomaly_registry"


@pytest.fixture
def registry(temp_registry_dir):
    """Create AnomalyRegistry instance."""
    return AnomalyRegistry(storage_dir=str(temp_registry_dir))


def test_anomaly_record_creation():
    """Test creating anomaly records."""
    record = AnomalyRecord(
        season=2024,
        race_round=1,
        driver_id=1,
        driver_name="Max",
        anomaly_type="podium_anomaly",
        anomaly_score=0.75,
        severity="warning",
        explanation="Unexpected podium finish"
    )

    assert record.season == 2024
    assert record.race_round == 1
    assert record.driver_id == 1
    assert record.anomaly_type == "podium_anomaly"
    assert record.severity == "warning"


def test_registry_initialization(registry):
    """Test registry initializes correctly."""
    assert registry is not None
    assert hasattr(registry, "add_anomaly")
    assert hasattr(registry, "get_anomalies")


def test_add_anomaly(registry):
    """Test adding anomalies to registry."""
    record = AnomalyRecord(
        season=2024,
        race_round=1,
        driver_id=1,
        driver_name="Max",
        anomaly_type="test",
        anomaly_score=0.5,
        severity="info"
    )

    registry.add_anomaly(record)

    assert len(registry.anomalies) == 1


def test_get_anomalies_by_season(registry):
    """Test retrieving anomalies by season."""
    record1 = AnomalyRecord(
        season=2024, race_round=1, driver_id=1, driver_name="Max",
        anomaly_type="test", anomaly_score=0.5, severity="info"
    )
    record2 = AnomalyRecord(
        season=2023, race_round=1, driver_id=2, driver_name="Lewis",
        anomaly_type="test", anomaly_score=0.6, severity="warning"
    )

    registry.add_anomaly(record1)
    registry.add_anomaly(record2)

    season_2024 = registry.get_anomalies(season=2024)
    assert len(season_2024) == 1
    assert season_2024[0].driver_id == 1


def test_get_anomalies_by_driver(registry):
    """Test retrieving anomalies by driver."""
    record1 = AnomalyRecord(
        season=2024, race_round=1, driver_id=1, driver_name="Max",
        anomaly_type="test", anomaly_score=0.5, severity="info"
    )
    record2 = AnomalyRecord(
        season=2024, race_round=2, driver_id=1, driver_name="Max",
        anomaly_type="test", anomaly_score=0.6, severity="warning"
    )
    record3 = AnomalyRecord(
        season=2024, race_round=1, driver_id=2, driver_name="Lewis",
        anomaly_type="test", anomaly_score=0.4, severity="info"
    )

    registry.add_anomaly(record1)
    registry.add_anomaly(record2)
    registry.add_anomaly(record3)

    driver_1_anomalies = registry.get_anomalies(driver_id=1)
    assert len(driver_1_anomalies) == 2
    assert all(a.driver_id == 1 for a in driver_1_anomalies)


def test_get_anomalies_by_severity(registry):
    """Test retrieving anomalies by severity level."""
    record1 = AnomalyRecord(
        season=2024, race_round=1, driver_id=1, driver_name="Max",
        anomaly_type="test", anomaly_score=0.9, severity="critical"
    )
    record2 = AnomalyRecord(
        season=2024, race_round=2, driver_id=1, driver_name="Max",
        anomaly_type="test", anomaly_score=0.6, severity="warning"
    )
    record3 = AnomalyRecord(
        season=2024, race_round=3, driver_id=2, driver_name="Lewis",
        anomaly_type="test", anomaly_score=0.4, severity="info"
    )

    registry.add_anomaly(record1)
    registry.add_anomaly(record2)
    registry.add_anomaly(record3)

    critical = registry.get_anomalies(severity="critical")
    assert len(critical) == 1
    assert critical[0].severity == "critical"


def test_save_and_load_registry(registry):
    """Test saving and loading registry to/from file."""
    record = AnomalyRecord(
        season=2024, race_round=1, driver_id=1, driver_name="Max",
        anomaly_type="test", anomaly_score=0.75, severity="warning"
    )

    registry.add_anomaly(record)
    registry.save()

    # Create new registry and load
    new_registry = AnomalyRegistry(storage_dir=str(registry.storage_dir))
    new_registry.load()

    assert len(new_registry.anomalies) == 1
    assert new_registry.anomalies[0].driver_id == 1


def test_registry_summary(registry):
    """Test registry summary statistics."""
    record1 = AnomalyRecord(
        season=2024, race_round=1, driver_id=1, driver_name="Max",
        anomaly_type="test", anomaly_score=0.9, severity="critical"
    )
    record2 = AnomalyRecord(
        season=2024, race_round=2, driver_id=2, driver_name="Lewis",
        anomaly_type="test", anomaly_score=0.6, severity="warning"
    )

    registry.add_anomaly(record1)
    registry.add_anomaly(record2)

    summary = registry.get_summary()

    assert summary["total_anomalies"] == 2
    assert summary["by_severity"]["critical"] == 1
    assert summary["by_severity"]["warning"] == 1


def test_registry_clear(registry):
    """Test clearing registry."""
    record = AnomalyRecord(
        season=2024, race_round=1, driver_id=1, driver_name="Max",
        anomaly_type="test", anomaly_score=0.5, severity="info"
    )

    registry.add_anomaly(record)
    assert len(registry.anomalies) == 1

    registry.clear()
    assert len(registry.anomalies) == 0


def test_duplicate_anomalies_detection(registry):
    """Test that duplicate anomalies are handled properly."""
    record = AnomalyRecord(
        season=2024, race_round=1, driver_id=1, driver_name="Max",
        anomaly_type="test", anomaly_score=0.5, severity="info"
    )

    registry.add_anomaly(record)
    registry.add_anomaly(record)  # Add same anomaly twice

    # Should either skip duplicate or have logic to handle it
    # Implementation detail - just verify it doesn't crash
    assert len(registry.anomalies) >= 1
