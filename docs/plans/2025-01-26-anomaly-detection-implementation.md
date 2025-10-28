# Anomaly Detection Enhancement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement dual-path anomaly detection (fast univariate during collection + sophisticated multivariate post-storage) integrated with F1DataCollector.

**Architecture:** Pipeline hooks pattern - AnomalyDetectionHooks injected at collection/storage points. UnivariateDetector runs synchronously during collection (~60ms). MultivariateAnalyzer runs asynchronously post-storage (~45s).

**Tech Stack:** scikit-learn (Isolation Forest), statsmodels (ARIMA), pandas, numpy, structlog

---

## Phase 1: Foundation - Core Hooks & Univariate Detection

### Task 1: Create AnomalyDetectionHooks base class

**Files:**
- Create: `src/f1_predict/data/anomaly_hooks.py`
- Test: `tests/data/test_anomaly_hooks.py`

**Step 1: Write the failing test**

```python
# tests/data/test_anomaly_hooks.py
import pytest
from f1_predict.data.anomaly_hooks import AnomalyDetectionHooks

def test_hooks_initialization():
    """Test AnomalyDetectionHooks initializes with all detectors."""
    hooks = AnomalyDetectionHooks()

    assert hooks is not None
    assert hasattr(hooks, 'univariate_detector')
    assert hasattr(hooks, 'multivariate_analyzer')
    assert hasattr(hooks, 'registry')

def test_on_data_collected_returns_data():
    """Test on_data_collected returns data with anomaly flags."""
    hooks = AnomalyDetectionHooks()

    data = [
        {'race_id': 1, 'position': 1, 'points': 25},
        {'race_id': 1, 'position': 2, 'points': 18},
    ]

    result = hooks.on_data_collected(data)

    assert result is not None
    assert len(result) == 2
    assert 'anomaly_flag' in result[0]
    assert 'anomaly_score' in result[0]
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/julienpequegnot/Code/f1-predict/.worktrees/anomaly-detection-issue-40
uv run pytest tests/data/test_anomaly_hooks.py::test_hooks_initialization -v
```

Expected output:
```
FAILED tests/data/test_anomaly_hooks.py::test_hooks_initialization - ModuleNotFoundError: No module named 'f1_predict.data.anomaly_hooks'
```

**Step 3: Write minimal implementation**

```python
# src/f1_predict/data/anomaly_hooks.py
"""Anomaly detection hooks for data pipeline integration."""

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class AnomalyMetadata:
    """Metadata for anomaly detection results."""

    anomaly_flag: bool = False
    anomaly_score: float = 0.0
    anomaly_method: str = ""
    anomaly_confidence: float = 0.0
    features_involved: list[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'anomaly_flag': self.anomaly_flag,
            'anomaly_score': self.anomaly_score,
            'anomaly_method': self.anomaly_method,
            'anomaly_confidence': self.anomaly_confidence,
            'features_involved': self.features_involved,
            'explanation': self.explanation,
        }


class AnomalyDetectionHooks:
    """Pluggable hooks for anomaly detection in data pipeline."""

    def __init__(self):
        """Initialize anomaly detection hooks."""
        self.logger = logger.bind(component="anomaly_hooks")
        # Will be initialized when detectors are available
        self.univariate_detector = None
        self.multivariate_analyzer = None
        self.registry = None

    def on_data_collected(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Hook: Run fast checks on collected data.

        Args:
            data: List of collected records

        Returns:
            Data with anomaly flags added
        """
        try:
            # Add anomaly metadata to each record
            for record in data:
                record['_anomaly'] = AnomalyMetadata().to_dict()
            return data
        except Exception as e:
            self.logger.error(f"Error in on_data_collected: {e}")
            return data

    def on_data_stored(
        self, data: list[dict[str, Any]], season: int
    ) -> dict[str, Any]:
        """
        Hook: Run sophisticated analysis post-storage.

        Args:
            data: Stored data records
            season: F1 season

        Returns:
            Anomaly report
        """
        try:
            return {
                'anomalies': [],
                'summary': {'total': 0, 'critical': 0},
            }
        except Exception as e:
            self.logger.error(f"Error in on_data_stored: {e}")
            return {'anomalies': [], 'summary': {}}
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/data/test_anomaly_hooks.py::test_hooks_initialization -v
uv run pytest tests/data/test_anomaly_hooks.py::test_on_data_collected_returns_data -v
```

Expected output:
```
PASSED tests/data/test_anomaly_hooks.py::test_hooks_initialization
PASSED tests/data/test_anomaly_hooks.py::test_on_data_collected_returns_data
```

**Step 5: Commit**

```bash
git add src/f1_predict/data/anomaly_hooks.py tests/data/test_anomaly_hooks.py
git commit -m "feat: Add AnomalyDetectionHooks base class with pipeline integration"
```

---

## Phase 2: Univariate Detector Implementation

### Task 2: Implement UnivariateDetector (Z-score & IQR)

**Files:**
- Create: `src/f1_predict/data/univariate_detector.py`
- Modify: `src/f1_predict/data/anomaly_hooks.py`
- Test: `tests/data/test_univariate_detector.py`

[See full implementation plan for complete code]

---

## Task Checklist

- [ ] Task 1: AnomalyDetectionHooks base class
- [ ] Task 2: UnivariateDetector (Z-score & IQR)
- [ ] Task 3: MultivariateAnalyzer (Isolation Forest)
- [ ] Task 4: RaceAnomalyDetector (F1-specific)
- [ ] Task 5: Integrate with F1DataCollector
- [ ] Task 6: AnomalyRegistry for persistence
- [ ] Task 7: CLI commands
- [ ] Task 8: Full test suite and verification

**Estimated Time:** 10-15 hours
**Estimated Completion:** 2-3 working sessions
