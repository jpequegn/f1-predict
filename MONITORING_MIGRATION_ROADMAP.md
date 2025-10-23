# Database Migration Roadmap for Monitoring System

## Quick Reference

### Current Architecture (File-Based)

```
Application Layer
    ├─ ModelPerformanceTracker
    │  └─ JSONL Files (predictions.jsonl, health_snapshots.jsonl, metrics.jsonl)
    │
    ├─ AlertingSystem
    │  ├─ JSONL File (alerts.jsonl)
    │  ├─ JSON File (alert_rules.json)
    │  └─ Text Log (alert_log.txt)
    │
    └─ Web Interface (Streamlit)
       ├─ monitoring.py page
       └─ monitoring_dashboard.py utils
```

### Target Architecture (Database-Backed)

```
Application Layer
    ├─ ModelPerformanceTracker (refactored)
    │  └─ PredictionRepository ─┐
    │                            │
    ├─ AlertingSystem (refactored)
    │  ├─ AlertRepository       ├─→ SQLAlchemy ORM
    │  └─ AlertRuleRepository   │
    │                            │
    ├─ HealthSnapshotRepository ┤
    │                            │
    └─ Web Interface (unchanged) │
       ├─ monitoring.py page     │
       └─ monitoring_dashboard.py│
                                 │
                    ┌────────────┘
                    ↓
            Database Layer
         ┌──────────────────┐
         │ PostgreSQL/SQLite │
         │  + Connection Pool│
         │  + Transactions   │
         │  + Indexes        │
         └──────────────────┘
```

---

## Phase 1: Database Integration (Immediate)

### 1.1 Prediction Model Update

**File:** `src/f1_predict/web/utils/database_models.py`

**Change Required:**
```python
# Add to Prediction model
actual_outcome = Column(Integer, nullable=True)  # Can be None until outcome recorded
actual_outcome_timestamp = Column(DateTime(timezone=True), nullable=True)
```

**Impact:** Allows storing actual outcomes with predictions

### 1.2 DatabaseManager Initialization

**File:** `src/f1_predict/web/app.py` (or appropriate startup location)

**Add:**
```python
from f1_predict.web.utils.database import DatabaseManager, DatabaseConfig

# In application startup
config = DatabaseConfig()
DatabaseManager.initialize(config)
DatabaseManager.create_all_tables()
```

**Environment Variables to Set:**
```bash
MONITORING_DB_ENABLED=true
MONITORING_DB_TYPE=sqlite  # or postgresql
MONITORING_DB_PATH=data/monitoring/monitoring.db
```

---

## Phase 2: ModelPerformanceTracker Refactoring

### 2.1 Update record_prediction()

**Before:**
```python
def record_prediction(self, prediction_id, model_version, predicted_outcome, 
                     confidence, features, metadata=None):
    record = PredictionRecord(...)
    with open(self.predictions_file, "a") as f:
        f.write(json.dumps(record.to_dict()) + "\n")
```

**After:**
```python
def record_prediction(self, prediction_id, model_version, predicted_outcome, 
                     confidence, features, metadata=None):
    from f1_predict.web.utils.database import DatabaseManager
    from f1_predict.web.utils.database_repositories import PredictionRepository
    
    with DatabaseManager.session_scope() as session:
        repo = PredictionRepository(session)
        repo.create(
            timestamp=datetime.now(timezone.utc),
            model_version=model_version,
            prediction_id=prediction_id,
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            features=features,
            extra_metadata=metadata
        )
```

### 2.2 Update record_actual_outcome()

**Before:**
```python
def record_actual_outcome(self, prediction_id, actual_outcome):
    predictions = self._load_predictions()
    matching = [p for p in predictions if p["prediction_id"] == prediction_id]
    # ... calculations but no persistence
```

**After:**
```python
def record_actual_outcome(self, prediction_id, actual_outcome):
    from f1_predict.web.utils.database import DatabaseManager
    from f1_predict.web.utils.database_repositories import PredictionRepository
    
    with DatabaseManager.session_scope() as session:
        repo = PredictionRepository(session)
        pred = repo.get_by_prediction_id(prediction_id)
        
        if not pred:
            return None
        
        # Update actual outcome
        pred.actual_outcome = actual_outcome
        pred.actual_outcome_timestamp = datetime.now(timezone.utc)
        session.flush()
        
        # Calculate metrics
        is_correct = pred.predicted_outcome == actual_outcome
        calibration_error = abs(pred.confidence - (1.0 if is_correct else 0.0))
        
        return {
            "prediction_id": prediction_id,
            "model_version": pred.model_version,
            "predicted": pred.predicted_outcome,
            "actual": actual_outcome,
            "correct": is_correct,
            "confidence": pred.confidence,
            "calibration_error": calibration_error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
```

### 2.3 Update get_performance_metrics()

**Before:**
```python
def get_performance_metrics(self, model_version=None, window_minutes=60):
    predictions = self._load_predictions()
    cutoff_time = time.time() - (window_minutes * 60)
    recent = [p for p in predictions if p["timestamp"] >= cutoff_time ...]
    df = pd.DataFrame(recent)
    # ... return metrics
```

**After:**
```python
def get_performance_metrics(self, model_version=None, window_minutes=60):
    from f1_predict.web.utils.database import DatabaseManager
    from f1_predict.web.utils.database_repositories import PredictionRepository
    
    hours = window_minutes / 60
    
    with DatabaseManager.session_scope() as session:
        repo = PredictionRepository(session)
        predictions = repo.get_recent(
            model_version=model_version or "all",
            hours=int(hours) + 1,
            limit=10000
        )
        
        if not predictions:
            return {}
        
        # Convert to dicts for DataFrame
        data = [p.to_dict() for p in predictions]
        df = pd.DataFrame(data)
        
        metrics = {
            "num_predictions": len(df),
            "avg_confidence": float(df["confidence"].mean()),
            "min_confidence": float(df["confidence"].min()),
            "max_confidence": float(df["confidence"].max()),
            "confidence_std": float(df["confidence"].std()),
        }
        
        return metrics
```

### 2.4 Update calculate_accuracy() - CRITICAL FIX

**Before:**
```python
def calculate_accuracy(self, model_version=None, window_minutes=60):
    # WRONG: Uses confidence as accuracy
    return float(np.mean([p["confidence"] for p in recent]))
```

**After:**
```python
def calculate_accuracy(self, model_version=None, window_minutes=60):
    from f1_predict.web.utils.database import DatabaseManager
    from f1_predict.web.utils.database_repositories import PredictionRepository
    
    hours = window_minutes / 60
    
    with DatabaseManager.session_scope() as session:
        repo = PredictionRepository(session)
        predictions = repo.get_recent(
            model_version=model_version or "all",
            hours=int(hours) + 1,
            limit=10000
        )
        
        # Filter to only predictions with actual outcomes
        with_outcomes = [p for p in predictions if p.actual_outcome is not None]
        
        if not with_outcomes:
            return None
        
        # Calculate TRUE accuracy
        correct = sum(1 for p in with_outcomes 
                     if p.predicted_outcome == p.actual_outcome)
        
        return correct / len(with_outcomes)
```

### 2.5 Update remaining methods similarly

For each of these methods, follow the pattern:
1. Open database session
2. Use appropriate repository
3. Query with indexes (timestamp, model_version)
4. Convert results to DataFrame if needed
5. Perform calculations
6. Return in same format as before

Methods to update:
- `calculate_calibration_metrics()` - use predictions with actual_outcome
- `get_recent_health_snapshots()` - delegate to HealthSnapshotRepository
- `get_performance_trend()` - use HealthSnapshotRepository with time queries
- Remove `_load_predictions()` - no longer needed

---

## Phase 3: AlertingSystem Refactoring

### 3.1 Update initialize

**Before:**
```python
def __init__(self, data_dir, channel_config=None):
    self.data_dir = Path(data_dir)
    self.data_dir.mkdir(parents=True, exist_ok=True)
    self.alerts_file = self.data_dir / "alerts.jsonl"
    self.rules_file = self.data_dir / "alert_rules.json"
    self.alert_callbacks = {}
    self._last_alert_times = {}  # IN-MEMORY - LOST ON RESTART
```

**After:**
```python
def __init__(self, data_dir=None, channel_config=None):
    # data_dir still accepted for backward compatibility, but not used
    if data_dir:
        logger.warning("data_dir parameter deprecated, using database")
    
    from f1_predict.web.utils.database import DatabaseManager
    
    self.alert_callbacks = {}
    self.logger = logger.bind(component="alerting_system")
    
    if channel_config is None:
        channel_config = load_alert_config_from_env()
    
    self.channel_config = channel_config
    self._initialize_channels()
    
    # Cache rules on initialization for performance
    self._refresh_rules_cache()
```

### 3.2 Add _refresh_rules_cache()

**New Method:**
```python
def _refresh_rules_cache(self):
    """Load and cache all enabled alert rules from database."""
    from f1_predict.web.utils.database import DatabaseManager
    from f1_predict.web.utils.database_repositories import AlertRuleRepository
    
    try:
        with DatabaseManager.session_scope() as session:
            repo = AlertRuleRepository(session)
            db_rules = repo.get_enabled()
            
            # Convert to AlertRule objects
            self.rules_cache = {
                rule.rule_id: AlertRule(
                    rule_id=rule.rule_id,
                    metric_name=rule.metric_name,
                    metric_type=rule.metric_type,
                    threshold=rule.threshold,
                    comparison=rule.comparison,
                    severity=rule.severity,
                    component=rule.component,
                    enabled=rule.enabled,
                    cooldown_minutes=rule.cooldown_minutes,
                    channels=rule.channels or [],
                    conditions=rule.conditions or {},
                )
                for rule in db_rules
            }
    except Exception as e:
        logger.error("failed_to_load_rules_cache", error=str(e))
        self.rules_cache = {}
```

### 3.3 Update evaluate_rule() - Cooldown Fix

**Before:**
```python
def evaluate_rule(self, rule, metric_value, model_version):
    if not rule.enabled:
        return None
    
    # Check in-memory cooldown (LOST ON RESTART)
    rule_key = rule.rule_id
    if rule_key in self._last_alert_times:
        cooldown_seconds = rule.cooldown_minutes * 60
        if time.time() - self._last_alert_times[rule_key] < cooldown_seconds:
            return None
    
    # ... evaluation logic
    
    # Update in-memory tracking
    self._last_alert_times[rule_key] = time.time()
```

**After:**
```python
def evaluate_rule(self, rule, metric_value, model_version):
    from f1_predict.web.utils.database import DatabaseManager
    from f1_predict.web.utils.database_repositories import AlertRuleRepository
    from datetime import datetime, timezone
    
    if not rule.enabled:
        return None
    
    # Check cooldown in database
    with DatabaseManager.session_scope() as session:
        repo = AlertRuleRepository(session)
        db_rule = repo.get_by_rule_id(rule.rule_id)
        
        if db_rule and db_rule.last_triggered:
            cooldown_seconds = rule.cooldown_minutes * 60
            elapsed = (datetime.now(timezone.utc) - db_rule.last_triggered).total_seconds()
            
            if elapsed < cooldown_seconds:
                return None
    
    # ... evaluation logic
    
    # Update last_triggered in database
    with DatabaseManager.session_scope() as session:
        repo = AlertRuleRepository(session)
        repo.update_last_triggered(rule.rule_id)
    
    # Deliver to channels as before
    for channel in rule.channels:
        self._deliver_alert(alert, channel)
```

### 3.4 Update create_alert()

**Before:**
```python
def create_alert(self, severity, title, message, ...):
    alert_id = f"alert_{int(time.time() * 1000)}"
    alert = Alert(timestamp=time.time(), ...)
    
    with open(self.alerts_file, "a") as f:
        f.write(json.dumps(alert.to_dict()) + "\n")
```

**After:**
```python
def create_alert(self, severity, title, message, metric_name, metric_value,
                threshold, component, model_version):
    from f1_predict.web.utils.database import DatabaseManager
    from f1_predict.web.utils.database_repositories import AlertRepository
    from datetime import datetime, timezone
    
    alert_id = f"alert_{int(time.time() * 1000)}"
    
    with DatabaseManager.session_scope() as session:
        repo = AlertRepository(session)
        db_alert = repo.create(
            timestamp=datetime.now(timezone.utc),
            alert_id=alert_id,
            severity=severity,
            title=title,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            component=component,
            model_version=model_version,
            acknowledged=False,
        )
    
    # Create dataclass for return and callbacks
    alert = Alert(
        timestamp=db_alert.timestamp.timestamp(),
        alert_id=alert_id,
        severity=severity,
        title=title,
        message=message,
        metric_name=metric_name,
        metric_value=metric_value,
        threshold=threshold,
        component=component,
        model_version=model_version,
        acknowledged=False,
    )
    
    self.logger.info("alert_created", alert_id=alert_id, severity=severity)
    self._trigger_callbacks(alert)
    
    return alert
```

### 3.5 Update acknowledge_alert() - Atomicity Fix

**Before:**
```python
def acknowledge_alert(self, alert_id, acknowledged_by=None):
    alerts = self.get_alerts(limit=10000)
    
    for alert in alerts:
        if alert.alert_id == alert_id:
            alert.acknowledged = True
            alert.acknowledged_at = time.time()
            alert.acknowledged_by = acknowledged_by
            
            # Rewrite entire file (NOT ATOMIC)
            with open(self.alerts_file, "w") as f:
                for a in sorted(alerts, key=lambda x: x.timestamp):
                    f.write(json.dumps(a.to_dict()) + "\n")
```

**After:**
```python
def acknowledge_alert(self, alert_id, acknowledged_by=None):
    from f1_predict.web.utils.database import DatabaseManager
    from f1_predict.web.utils.database_repositories import AlertRepository
    
    with DatabaseManager.session_scope() as session:
        repo = AlertRepository(session)
        result = repo.acknowledge(alert_id, acknowledged_by)
        
        if result:
            self.logger.info("alert_acknowledged", alert_id=alert_id, by=acknowledged_by)
            return True
        
        return False
```

### 3.6 Update get_alerts()

**Before:**
```python
def get_alerts(self, limit=100, severity=None):
    alerts = []
    with open(self.alerts_file) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if severity is None or data["severity"] == severity:
                    alerts.append(Alert(**data))
    
    return sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:limit]
```

**After:**
```python
def get_alerts(self, limit=100, severity=None):
    from f1_predict.web.utils.database import DatabaseManager
    from f1_predict.web.utils.database_repositories import AlertRepository
    
    with DatabaseManager.session_scope() as session:
        repo = AlertRepository(session)
        db_alerts = repo.get_recent(limit=limit, severity=severity)
        
        # Convert to Alert dataclasses
        alerts = [
            Alert(
                timestamp=a.timestamp.timestamp(),
                alert_id=a.alert_id,
                severity=a.severity,
                title=a.title,
                message=a.message,
                metric_name=a.metric_name,
                metric_value=a.metric_value,
                threshold=a.threshold,
                component=a.component,
                model_version=a.model_version,
                acknowledged=a.acknowledged,
                acknowledged_at=a.acknowledged_at.timestamp() if a.acknowledged_at else None,
                acknowledged_by=a.acknowledged_by,
            )
            for a in db_alerts
        ]
        
        return alerts
```

### 3.7 Update add_rule()

**Before:**
```python
def add_rule(self, rule):
    rules = self._load_rules() or []
    rules.append(rule.to_dict())
    
    with open(self.rules_file, "w") as f:
        json.dump(rules, f, indent=2)
```

**After:**
```python
def add_rule(self, rule):
    from f1_predict.web.utils.database import DatabaseManager
    from f1_predict.web.utils.database_repositories import AlertRuleRepository
    
    with DatabaseManager.session_scope() as session:
        repo = AlertRuleRepository(session)
        repo.create(
            rule_id=rule.rule_id,
            metric_name=rule.metric_name,
            metric_type=rule.metric_type,
            threshold=rule.threshold,
            comparison=rule.comparison,
            severity=rule.severity,
            component=rule.component,
            enabled=rule.enabled,
            cooldown_minutes=rule.cooldown_minutes,
            channels=rule.channels,
            conditions=rule.conditions,
        )
    
    # Refresh cache
    self._refresh_rules_cache()
    
    self.logger.info("alert_rule_added", rule_id=rule.rule_id)
```

### 3.8 Update get_alert_statistics()

**Before:**
```python
def get_alert_statistics(self):
    alerts = self.get_alerts(limit=10000)
    
    stats = {
        "total_alerts": len(alerts),
        "unacknowledged": len([a for a in alerts if not a.acknowledged]),
        "by_severity": {...},
        "by_component": {...},
    }
```

**After:**
```python
def get_alert_statistics(self):
    from f1_predict.web.utils.database import DatabaseManager
    from f1_predict.web.utils.database_repositories import AlertRepository
    
    with DatabaseManager.session_scope() as session:
        repo = AlertRepository(session)
        stats = repo.get_statistics()
        
        return stats
```

### 3.9 Remove methods no longer needed

- `_load_rules()` - no longer needed
- Remove file-based alert delivery from `_deliver_alert()` (optional, keep for backward compat)

---

## Phase 4: Testing Updates

### 4.1 Update test_monitoring.py

**Before:**
```python
def performance_tracker(temp_dir):
    return ModelPerformanceTracker(temp_dir)

def alerting_system(temp_dir):
    return AlertingSystem(temp_dir)
```

**After:**
```python
@pytest.fixture(autouse=True)
def setup_database(temp_dir, monkeypatch):
    """Setup test database before each test."""
    monkeypatch.setenv("MONITORING_DB_ENABLED", "true")
    monkeypatch.setenv("MONITORING_DB_TYPE", "sqlite")
    monkeypatch.setenv("MONITORING_DB_PATH", str(temp_dir / "test.db"))
    
    from f1_predict.web.utils.database import DatabaseManager, DatabaseConfig
    
    config = DatabaseConfig()
    DatabaseManager.initialize(config)
    DatabaseManager.create_all_tables()
    
    yield
    
    DatabaseManager.drop_all_tables()

def performance_tracker():
    return ModelPerformanceTracker()  # No data_dir needed

def alerting_system():
    return AlertingSystem()  # No data_dir needed
```

---

## Phase 5: Deployment Update

### 5.1 Update monitoring.py page

**Before:**
```python
@st.cache_resource
def get_monitoring_systems():
    temp_dir = tempfile.gettempdir()
    return {
        "performance_tracker": ModelPerformanceTracker(temp_dir),
        "alerting_system": AlertingSystem(temp_dir),
        "drift_detector": DriftDetector(),
    }
```

**After:**
```python
@st.cache_resource
def get_monitoring_systems():
    from f1_predict.web.utils.database import DatabaseManager
    
    # Ensure database is initialized
    DatabaseManager.initialize()
    
    return {
        "performance_tracker": ModelPerformanceTracker(),
        "alerting_system": AlertingSystem(),
        "drift_detector": DriftDetector(),
    }
```

### 5.2 Environment Configuration

Add to `.env` or deployment config:
```bash
# Enable database monitoring
MONITORING_DB_ENABLED=true

# Choose SQLite for local dev, PostgreSQL for production
MONITORING_DB_TYPE=sqlite
# MONITORING_DB_TYPE=postgresql

# SQLite path
MONITORING_DB_PATH=data/monitoring/monitoring.db

# PostgreSQL (if using)
# MONITORING_DB_HOST=localhost
# MONITORING_DB_PORT=5432
# MONITORING_DB_USER=postgres
# MONITORING_DB_PASSWORD=your_password
# MONITORING_DB_NAME=f1_predict_monitoring
```

---

## Rollback Plan

If issues occur during migration:

1. **Keep JSONL files** as backup during initial deployment
2. **Disable database** by setting `MONITORING_DB_ENABLED=false`
3. **Implement graceful fallback** in both classes:
   ```python
   def __init__(self, ...):
       self.use_database = DatabaseManager.is_enabled()
   
   def record_prediction(self, ...):
       if self.use_database:
           # Use database path
       else:
           # Use JSONL fallback
   ```

---

## Testing Checklist

- [ ] Create test database successfully
- [ ] Insert predictions and verify indexing
- [ ] Query with time windows (< 1 second)
- [ ] Calculate accuracy correctly with actual outcomes
- [ ] Alert rules cache refreshes
- [ ] Cooldown persists across restarts
- [ ] Alert acknowledgment is atomic
- [ ] Web interface displays data correctly
- [ ] No memory leaks in long-running sessions
- [ ] PostgreSQL configuration works
- [ ] SQLite configuration works
- [ ] Backward compatibility with existing API

---

## Success Criteria

1. All predictions indexed and queryable in <100ms
2. Alerts atomic and persistent
3. Cooldown tracking survives restarts
4. Accuracy calculation correct (not confidence proxy)
5. Web interface unchanged from user perspective
6. All existing tests pass
7. No data loss during migration
8. Performance improved for large datasets

