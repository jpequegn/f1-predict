# Comprehensive Analysis: Current Monitoring & Alerting System

## Executive Summary

The current monitoring and alerting system is built on **file-based storage** using JSONL (JSON Lines) format. Both `ModelPerformanceTracker` and `AlertingSystem` classes persist data to local files with in-memory state management. A new database backend infrastructure (PostgreSQL/SQLite) has been prepared but not yet integrated.

---

## 1. Current MonitoringTracker (ModelPerformanceTracker) Implementation

### 1.1 Class Overview

**File:** `src/f1_predict/web/utils/monitoring.py`

**Purpose:** Track predictions, health metrics, and calculate performance metrics over time

### 1.2 Data Structures

#### Core Dataclasses:

1. **PredictionRecord**
   - Fields: timestamp, model_version, prediction_id, predicted_outcome, confidence, features, metadata
   - Used for: Individual prediction tracking
   - Serialization: to_dict() method for JSON conversion

2. **PerformanceMetric**
   - Fields: timestamp, metric_name, value, window_size, threshold, status
   - Used for: Time-point metrics snapshots
   - Storage: Persisted to metrics.jsonl

3. **ModelHealthSnapshot**
   - Fields: timestamp, model_version, accuracy, precision, recall, f1_score, roc_auc, expected_calibration_error, num_predictions, prediction_accuracy_trend, degradation_detected
   - Used for: Periodic health assessments
   - Storage: Persisted to health_snapshots.jsonl

### 1.3 File-Based Storage Structure

```
data/monitoring/
├── predictions.jsonl          # Individual prediction records
├── metrics.jsonl              # Performance metric snapshots
└── health_snapshots.jsonl     # Health snapshots
```

**Storage Format:**
- Each line is a JSON object (JSONL = JSON Lines)
- Append-only for predictions and alerts
- Complete rewrite for acknowledgments

### 1.4 ModelPerformanceTracker Methods & Implementation

#### A. record_prediction()
- **Lines:** 94-125
- **Current Implementation:**
  ```python
  - Creates PredictionRecord instance
  - Appends to predictions.jsonl as JSON line
  - No database persistence
  ```
- **Data Flow:** In-memory dataclass → JSON serialization → JSONL append
- **Limitations:**
  - Slow for large queries (must load entire file)
  - No indexing
  - No transactional safety

#### B. record_actual_outcome()
- **Lines:** 127-172
- **Current Implementation:**
  ```python
  - Loads all predictions via _load_predictions()
  - Linear search for matching prediction_id
  - Calculates calibration error
  - Returns evaluation result dict
  ```
- **Issues:**
  - No persistence of actual outcome
  - No update to prediction record
  - Full file load for single lookup

#### C. get_performance_metrics()
- **Lines:** 174-211
- **Current Implementation:**
  ```python
  - Loads all predictions
  - Filters by timestamp and model_version
  - Converts to pandas DataFrame
  - Calculates statistics
  ```
- **Window-based:** Uses window_minutes parameter (default 60)
- **Metrics calculated:**
  - num_predictions
  - avg_confidence
  - min_confidence
  - max_confidence
  - confidence_std

#### D. calculate_accuracy()
- **Lines:** 213-245
- **Current Implementation:**
  ```python
  - Loads all predictions
  - Filters by window
  - Uses confidence as proxy for accuracy (incorrect)
  - Returns mean confidence
  ```
- **Critical Issue:** Uses confidence as accuracy proxy instead of actual/predicted comparison

#### E. calculate_calibration_metrics()
- **Lines:** 247-305
- **Current Implementation:**
  ```python
  - Binning approach (default 10 bins)
  - Calculates Expected Calibration Error (ECE)
  - Per-bin statistics
  - No actual outcome data used
  ```
- **Method:** Simplified ECE calculation (0.05 * proportion per bin)

#### F. record_health_snapshot()
- **Lines:** 307-320
- **Current Implementation:**
  ```python
  - Appends ModelHealthSnapshot to health_snapshots.jsonl
  - Logs info message
  - No validation
  ```

#### G. get_recent_health_snapshots()
- **Lines:** 322-352
- **Current Implementation:**
  ```python
  - Loads all health snapshots
  - Filters by model_version
  - Sorts by timestamp (descending)
  - Returns top N
  ```
- **Performance Issue:** O(N) file read for every call

#### H. get_performance_trend()
- **Lines:** 354-384
- **Current Implementation:**
  ```python
  - Gets recent health snapshots
  - Filters by time window
  - Converts to pandas DataFrame
  - Sorts by datetime
  ```
- **Returns:** DataFrame with performance metrics over time

#### I. _load_predictions() (Private)
- **Lines:** 386-398
- **Current Implementation:**
  ```python
  - Opens predictions.jsonl
  - Reads all lines
  - Parses JSON
  - Returns list of dicts
  ```
- **Bottleneck:** Called by multiple methods, full file load each time

---

## 2. AlertingSystem Class Implementation

### 2.1 Class Overview

**File:** `src/f1_predict/web/utils/alerting.py`

**Purpose:** Create, manage, and deliver alerts based on monitoring rules

### 2.2 Data Structures

#### Core Dataclasses:

1. **Alert**
   - Fields: timestamp, alert_id, severity, title, message, metric_name, metric_value, threshold, component, model_version, acknowledged, acknowledged_at, acknowledged_by
   - Stored in: alerts.jsonl
   - Unique identifier: alert_id

2. **AlertRule**
   - Fields: rule_id, metric_name, metric_type, threshold, comparison, severity, component, enabled, cooldown_minutes, channels, conditions
   - Stored in: alert_rules.json (single file, not JSONL)
   - Used for: Rule configuration and evaluation

#### Enums:

1. **AlertSeverity:** INFO, WARNING, CRITICAL
2. **AlertChannel:** CONSOLE, EMAIL, SLACK, FILE

### 2.3 File-Based Storage Structure

```
data/monitoring/
├── alerts.jsonl               # Individual alerts (append-only)
├── alert_rules.json           # Alert rules (entire file rewrite)
└── alert_log.txt              # Text log (file channel)
```

### 2.4 AlertingSystem Methods & Implementation

#### A. create_alert()
- **Lines:** 125-181
- **Current Implementation:**
  ```python
  - Generates alert_id from timestamp
  - Creates Alert instance
  - Appends to alerts.jsonl
  - Logs and triggers callbacks
  ```
- **Callback System:** Registered callbacks triggered per alert

#### B. evaluate_rule()
- **Lines:** 183-256
- **Current Implementation:**
  ```python
  1. Check if rule enabled
  2. Check cooldown (using _last_alert_times dict in memory)
  3. Evaluate comparison operator (<, >, ==, !=)
  4. Check additional conditions (min_value, max_value)
  5. Create alert if triggered
  6. Update cooldown timestamp
  7. Deliver to channels
  ```
- **Cooldown Mechanism:** In-memory dict tracking (lost on restart)
- **Channel Delivery:** Delegates to _deliver_alert()

#### C. add_rule()
- **Lines:** 258-270
- **Current Implementation:**
  ```python
  - Loads existing rules from alert_rules.json
  - Appends new rule
  - Rewrites entire file
  ```
- **Issue:** Not atomic, potential data loss

#### D. get_alerts()
- **Lines:** 272-293
- **Current Implementation:**
  ```python
  - Loads all alerts from alerts.jsonl
  - Filters by severity (optional)
  - Sorts by timestamp (descending)
  - Returns top N
  ```

#### E. acknowledge_alert()
- **Lines:** 295-330
- **Current Implementation:**
  ```python
  1. Load all alerts (up to 10000)
  2. Find matching alert_id
  3. Update acknowledged, acknowledged_at, acknowledged_by
  4. Rewrite entire alerts.jsonl file
  5. Log update
  ```
- **Critical Issue:** 
  - Reads up to 10K records
  - Rewrites entire file (slow, not atomic)
  - In-memory state lost between restarts

#### F. get_alert_statistics()
- **Lines:** 332-356
- **Current Implementation:**
  ```python
  - Loads all alerts
  - Counts by severity
  - Counts acknowledged vs unacknowledged
  - Counts by component
  ```

#### G. register_callback()
- **Lines:** 358-367
- **Current Implementation:**
  ```python
  - Stores callback in in-memory dict
  - Supports multiple callbacks per channel
  ```

#### H. _trigger_callbacks() (Private)
- **Lines:** 369-390
- **Current Implementation:**
  ```python
  - Logs alert with appropriate level
  - Iterates through registered callbacks
  - Calls each callback with alert
  - Catches exceptions
  ```

#### I. _initialize_channels() (Private)
- **Lines:** 392-423
- **Current Implementation:**
  ```python
  - Initializes EmailAlertChannel (if enabled)
  - Initializes SlackAlertChannel (if enabled)
  - Loads config from AlertChannelConfig
  - Validates channel configurations
  ```

#### J. _deliver_alert() (Private)
- **Lines:** 425-461
- **Current Implementation:**
  ```python
  - Routes to appropriate channel:
    - console: Print to stdout
    - file: Append to alert_log.txt
    - email: Delegates to EmailAlertChannel
    - slack: Delegates to SlackAlertChannel
  ```

#### K. _load_rules() (Private)
- **Lines:** 463-475
- **Current Implementation:**
  ```python
  - Loads alert_rules.json
  - Returns list of rule dicts
  - Returns None on error
  ```

---

## 3. Integration Points Analysis

### 3.1 Web Interface (Streamlit Monitoring Page)

**File:** `src/f1_predict/web/pages/monitoring.py`

**Current Usage:**
- **Lines 34-39:** `get_monitoring_systems()` cached resource
  ```python
  temp_dir = tempfile.gettempdir()
  Returns:
    - ModelPerformanceTracker(temp_dir)
    - AlertingSystem(temp_dir)
    - DriftDetector()
  ```
  - **Issue:** Uses temp directory (data lost on reboot)

- **Performance Dashboard (Lines 42-140):**
  - Calls `perf_tracker.get_performance_trend()`
  - Calls `perf_tracker.get_recent_health_snapshots()`
  - Displays in DataLoaders.load_performance_trend()

- **Drift Detection Dashboard (Lines 142-184):**
  - No current integration with monitoring data

- **Alert Management (Lines 186-277):**
  - Calls `alerting_system.get_alerts()`
  - Calls `alerting_system.acknowledge_alert()`
  - Displays via TableFormatters.format_alert_table()

- **Model Comparison (Lines 279-354):**
  - Uses hardcoded sample data
  - No integration with tracking systems

### 3.2 Monitoring Dashboard Utilities

**File:** `src/f1_predict/web/utils/monitoring_dashboard.py`

**DataLoaders class (Lines 432-520):**
1. `load_performance_trend()` (435-456)
   - Calls `performance_tracker.get_performance_trend()`
   - Returns DataFrame or empty

2. `load_health_snapshots()` (458-479)
   - Calls `performance_tracker.get_recent_health_snapshots()`
   - Returns DataFrame or empty

3. `load_recent_alerts()` (481-503)
   - Calls `alerting_system.get_alerts()`
   - Returns DataFrame or empty

4. `load_drift_results()` (505-520)
   - Placeholder (not implemented)

### 3.3 Alert Enrichment

**File:** `src/f1_predict/web/utils/alert_enricher.py`

**Dependencies:**
- Imports Alert from alerting.py
- `enrich_alert()` method adds SHAP explanations
- `format_email_with_explanation()` - email formatting
- `format_slack_with_explanation()` - Slack formatting
- Currently standalone, not integrated with AlertingSystem

---

## 4. Current Data Flow Analysis

### 4.1 Prediction Recording Flow

```
Application Code
    ↓
record_prediction()
    ↓
Create PredictionRecord dataclass
    ↓
JSON serialize (.to_dict())
    ↓
Append to predictions.jsonl
    ↓
Storage (File System)
```

**Key Points:**
- Synchronous, blocking I/O
- No deduplication (duplicate prediction_ids possible)
- No validation
- Lost on file system errors

### 4.2 Health Snapshot Creation Flow

```
External Application
    ↓
Create ModelHealthSnapshot
    ↓
Call record_health_snapshot()
    ↓
JSON serialize
    ↓
Append to health_snapshots.jsonl
    ↓
Storage (File System)
```

**Issues:**
- Manual creation required (not calculated by tracker)
- No automatic periodic snapshots
- No trigger for degradation detection

### 4.3 Alert Triggering Flow

```
Metric Calculation
    ↓
Evaluate Rule (evaluate_rule())
    ↓
Comparison Logic (<, >, ==, !=)
    ↓
Check Cooldown (in-memory dict)
    ↓
create_alert()
    ↓
Append to alerts.jsonl
    ↓
_trigger_callbacks()
    ↓
_deliver_alert() (multiple channels)
    │
    ├─→ Console (print)
    ├─→ File (alert_log.txt)
    ├─→ Email (EmailAlertChannel)
    └─→ Slack (SlackAlertChannel)
```

**Weaknesses:**
- Cooldown only in memory (resets on restart)
- Multiple file writes
- Not transactional

### 4.4 Alert Acknowledgment Flow

```
User Action (Web UI)
    ↓
acknowledge_alert(alert_id)
    ↓
Load all alerts (10K limit)
    ↓
Find matching alert_id (linear search)
    ↓
Update fields in memory
    ↓
Rewrite entire alerts.jsonl file
    ↓
Storage (File System)
```

**Problems:**
- 10K record limitation
- Full file rewrite (slow, risky)
- No atomicity

---

## 5. Backward Compatibility Considerations

### 5.1 Interfaces That Must Be Maintained

#### ModelPerformanceTracker:
```python
# Constructor
__init__(data_dir: Path | str = "data/monitoring")

# Public methods that external code depends on
record_prediction(prediction_id, model_version, predicted_outcome, 
                  confidence, features, metadata=None) → None

record_actual_outcome(prediction_id, actual_outcome) → dict | None

get_performance_metrics(model_version=None, window_minutes=60) → dict

calculate_accuracy(model_version=None, window_minutes=60) → float | None

calculate_calibration_metrics(model_version=None, n_bins=10) → dict

record_health_snapshot(snapshot: ModelHealthSnapshot) → None

get_recent_health_snapshots(model_version=None, limit=100) → list[ModelHealthSnapshot]

get_performance_trend(model_version=None, hours=24) → pd.DataFrame
```

#### AlertingSystem:
```python
# Constructor
__init__(data_dir: Path | str = "data/monitoring", 
         channel_config: AlertChannelConfig | None = None)

# Public methods
create_alert(severity, title, message, metric_name, metric_value, 
             threshold, component, model_version) → Alert

evaluate_rule(rule: AlertRule, metric_value: float, 
              model_version: str) → Alert | None

add_rule(rule: AlertRule) → None

get_alerts(limit=100, severity=None) → list[Alert]

acknowledge_alert(alert_id: str, acknowledged_by=None) → bool

get_alert_statistics() → dict

register_callback(channel: str, callback: Callable) → None
```

### 5.2 Data Contracts (Return Types)

These return types are used by calling code and must remain compatible:

1. **PredictionRecord** - dataclass
   - Used in: Monitoring page, health checks
   - Must preserve: to_dict() method

2. **ModelHealthSnapshot** - dataclass
   - Used in: Performance tracking, dashboard
   - Must preserve: All fields, to_dict() method

3. **Alert** - dataclass
   - Used in: Alert management UI, enrichment
   - Must preserve: All fields, to_dict() method

4. **AlertRule** - dataclass
   - Used in: Rule configuration, evaluation
   - Must preserve: All fields, to_dict() method

5. **pandas.DataFrame returns**
   - Expected columns: timestamp, accuracy, precision, recall, f1_score
   - Used in: Plotly charts, trend analysis

### 5.3 What Can Be Changed Safely

- **Internal storage:** File → Database (transparent to users)
- **_load_predictions():** Completely internal
- **_load_rules():** Completely internal
- **_deliver_alert():** Can be refactored if public interface preserved
- **Initialization of channels:** Can be refactored
- **Cooldown tracking:** Can move from in-memory to database

### 5.4 What Requires Migration

1. **Existing JSONL files** must be migrated to database
2. **Cooldown state** must be preserved (last_triggered timestamps)
3. **Alert acknowledgments** must be migrated
4. **Historical data** must be preserved

---

## 6. Identified Gaps & Issues

### 6.1 Critical Issues

1. **Prediction Accuracy Calculation (Line 242)**
   - Current: Uses confidence as accuracy proxy
   - Should: Compare predicted_outcome with actual_outcome
   - Impact: Metrics are completely wrong

2. **Cooldown Reset on Restart (Line 112)**
   - Current: In-memory dict `_last_alert_times`
   - Issue: Lost between application restarts
   - Result: May fire alerts more frequently than intended

3. **Alert Acknowledgment Not Atomic (Lines 318-320)**
   - Current: Load → Update → Rewrite entire file
   - Issue: Data loss if process crashes mid-write
   - Impact: Lost alert acknowledgments

4. **Data Storage in Temp Directory (monitoring.py:34)**
   - Current: `tempfile.gettempdir()`
   - Issue: Data lost on system reboot
   - Impact: Complete loss of monitoring history

### 6.2 Performance Issues

1. **Full File Loads**
   - `_load_predictions()` called by multiple methods
   - No caching between calls
   - 60+ second query for large files

2. **Linear Searches**
   - `record_actual_outcome()` searches all predictions
   - `acknowledge_alert()` searches all alerts
   - O(N) time complexity

3. **Complete File Rewrites**
   - `acknowledge_alert()` rewrites entire alerts.jsonl
   - `add_rule()` rewrites entire alert_rules.json
   - Risk of data loss, slow performance

### 6.3 Design Issues

1. **No Actual Outcome Storage**
   - `record_actual_outcome()` doesn't persist the outcome
   - Can't calculate true accuracy later
   - No prediction-outcome linkage

2. **Manual Health Snapshot Creation**
   - No automatic calculations
   - External systems must call `record_health_snapshot()`
   - Easy to forget or miss

3. **Callback System Not Persisted**
   - `register_callback()` only in-memory
   - Lost on restart
   - No way to recover callback state

4. **Missing Validation**
   - No schema validation
   - No constraint checks
   - No duplicate prevention

---

## 7. Database Backend Readiness

### 7.1 ORM Models Prepared

**File:** `src/f1_predict/web/utils/database_models.py`

All required models defined:
- Prediction (with indexes on timestamp, model_version, prediction_id)
- HealthSnapshot (with indexes)
- Alert (with indexes on timestamp, severity, acknowledged)
- AlertRule (with last_triggered tracking)
- DriftResult
- FeatureImportance
- DegradationAnalysis
- MetricSnapshot

### 7.2 Database Configuration

**File:** `src/f1_predict/web/utils/database.py`

- Singleton DatabaseManager
- Supports PostgreSQL and SQLite
- Connection pooling configured
- SQLite WAL (Write-Ahead Logging) configured
- Health check available
- Schema creation available

### 7.3 Repository Pattern Implemented

**File:** `src/f1_predict/web/utils/database_repositories.py`

All repositories ready:
- PredictionRepository (with batch operations)
- HealthSnapshotRepository (with degradation queries)
- AlertRepository (with filtering, statistics)
- AlertRuleRepository (with enabled/metric filtering)
- DriftResultRepository
- FeatureImportanceRepository
- DegradationAnalysisRepository
- MetricSnapshotRepository

---

## 8. Migration Complexity Assessment

### 8.1 High Complexity

1. **Cooldown Mechanism**
   - Currently: In-memory dict
   - Must: Track last_triggered per rule in database
   - Complexity: Need to update AlertRule model if not already done
   - Status: Already added to AlertRule model ✓

2. **Acknowledgment Workflow**
   - Currently: Rewrite entire file
   - Must: Update single alert record atomically
   - Already implemented in AlertRepository ✓

3. **Prediction-Outcome Linkage**
   - Currently: Not linked
   - Must: Add actual_outcome field to Prediction model
   - Additional work needed

### 8.2 Medium Complexity

1. **Migration Tool**
   - Must parse existing JSONL files
   - Convert timestamps
   - Bulk insert to database
   - Validate migration

2. **Graceful Fallback**
   - Database disabled → File fallback
   - Handle both simultaneously during transition
   - Configuration-driven switching

### 8.3 Low Complexity

1. **Most Repository Methods**
   - Already implemented
   - Just need to integrate
   - Tests exist

---

## 9. Files Requiring Updates

### 9.1 Core System Files

| File | Changes Required | Impact |
|------|-----------------|--------|
| monitoring.py | Complete refactor: replace file I/O with DB queries | HIGH |
| alerting.py | Refactor: use repositories instead of JSONL | HIGH |
| database_models.py | Add actual_outcome field to Prediction | LOW |
| database_repositories.py | Small additions for monitoring queries | MEDIUM |

### 9.2 Integration Points

| File | Changes Required | Impact |
|------|-----------------|--------|
| monitoring.py (page) | Update initialization (remove temp_dir usage) | MEDIUM |
| monitoring_dashboard.py | Update data loader implementations | MEDIUM |
| alert_enricher.py | No changes (works with Alert dataclass) | NONE |

### 9.3 Testing Files

| File | Changes Required | Impact |
|------|-----------------|--------|
| test_monitoring.py | Update fixtures, tests | MEDIUM |
| test_database.py | Add new tests for integrated flows | MEDIUM |

---

## 10. Key Dependencies to Track

### 10.1 External Dependencies

```python
from f1_predict.web.utils.monitoring import (
    PredictionRecord,
    PerformanceMetric,
    ModelHealthSnapshot,
    ModelPerformanceTracker,  # PRIMARY
)

from f1_predict.web.utils.alerting import (
    AlertSeverity,
    AlertChannel,
    Alert,
    AlertRule,
    AlertingSystem,  # PRIMARY
)
```

### 10.2 Derived Dependencies

```python
from f1_predict.web.utils.alert_enricher import ExplanabilityAlertEnricher
# Uses Alert from alerting.py
# No changes needed for database migration

from f1_predict.web.pages.monitoring import show_monitoring_page
# Creates monitoring systems
# MUST change initialization

from f1_predict.web.utils.monitoring_dashboard import DataLoaders
# Calls tracker and alerting methods
# Will work unchanged if interfaces preserved
```

---

## Summary Table

| Aspect | Current State | Issues | Database Ready? |
|--------|---------------|--------|-----------------|
| Prediction Storage | JSONL file | No indexes, slow | YES - Model exists |
| Health Snapshots | JSONL file | No calculation trigger | YES - Model exists |
| Alert Storage | JSONL file | Not atomic | YES - Model exists |
| Alert Rules | JSON file | Entire rewrite | YES - Model exists |
| Cooldown Tracking | In-memory dict | Lost on restart | YES - Field in model |
| Actual Outcomes | Not tracked | Can't calculate accuracy | NO - Need field |
| Repositories | N/A | Full implementation | YES - All ready |
| Config/DB Mgmt | Prepared | Not integrated | YES - Complete |

