# F1 Predict Codebase Analysis
## ML Model Architecture & Data Flow

**Project**: F1 Predict (Formula 1 Race Outcome Prediction)  
**Status**: Production-ready with model monitoring (Issue #36 on feature/model-monitoring-issue-36)  
**Test Coverage**: 80%+  
**Python**: 3.9-3.12  
**Last Updated**: October 2024

---

## 1. CURRENT ML MODEL IMPLEMENTATIONS

### 1.1 Model Hierarchy & Architecture

```
src/f1_predict/models/
├── __init__.py
├── baseline.py              [149 lines] - RuleBasedPredictor
├── logistic.py              [296 lines] - LogisticRacePredictor
├── random_forest.py         [352 lines] - RandomForestRacePredictor
├── xgboost_model.py         [424 lines] - XGBoostRacePredictor
├── lightgbm_model.py        [429 lines] - LightGBMRacePredictor
├── ensemble.py              [259 lines] - EnsemblePredictor
└── evaluation.py            [331 lines] - ModelEvaluator
```

### 1.2 Available Models

| Model | Type | Target Predictions | Key Features |
|-------|------|-------------------|--------------|
| **RuleBasedPredictor** | Heuristic | Position/Podium | Qualifying weight 60%, Form 20%, Reliability 10%, Circuit 10% |
| **LogisticRacePredictor** | Linear | Binary classification | Fast, interpretable, baseline model |
| **RandomForestRacePredictor** | Ensemble (sklearn) | Binary outcomes | 100 trees, feature importance, OOB scoring |
| **XGBoostRacePredictor** | Gradient Boosting | Binary outcomes | 100 rounds, GPU support, early stopping, fast |
| **LightGBMRacePredictor** | Gradient Boosting | Binary outcomes | Leaf-based trees, efficient, parallel-friendly |
| **EnsemblePredictor** | Meta-learner | Combined | Soft/hard voting, weighted averaging, model agreement |

### 1.3 Prediction Targets

All models support three prediction targets:
- **"podium"**: Position <= 3 (top 3 finishers)
- **"points"**: Position <= 10 (points-scoring finishers)
- **"win"**: Position == 1 (race winner)

---

## 2. FEATURE ENGINEERING PIPELINE

### 2.1 Architecture

```
src/f1_predict/features/engineering.py [633 lines]
├── DriverFormCalculator
├── TeamReliabilityCalculator
├── TrackPerformanceCalculator
├── QualifyingRaceGapCalculator
├── WeatherFeatureCalculator
└── FeatureEngineer (orchestrator)
```

### 2.2 Feature Calculators

#### DriverFormCalculator
- **Input**: Race results + driver_id + date window
- **Output**: form_score (0-100)
- **Calculation**:
  - Weighted position scores (recency-weighted, default window=5)
  - Consistency penalty (std dev of positions * 2)
  - DNF penalty (up to 30 points)
  - Formula: 60% position + 25% consistency + 15% reliability

#### TeamReliabilityCalculator
- **Input**: Race results + constructor_id
- **Output**: Dictionary with metrics
  - finish_rate: % of races completed
  - avg_position: Average finishing position
  - mechanical_failure_rate: DNF rate
  - points_consistency: Std dev of points
  - reliability_score: Weighted composite (40% finish + 30% position + 30% consistency)

#### TrackPerformanceCalculator
- **Input**: Race results + driver_id + circuit_id
- **Output**: Track-specific performance metrics
  - avg_position at circuit
  - avg_points at circuit
  - best_position
  - races_at_track
  - track_performance_score (60% position + 40% points)
- **Requirement**: Minimum 2 races at circuit

#### QualifyingRaceGapCalculator
- **Input**: Race + qualifying results + driver_id
- **Output**: Racecraft metrics
  - avg_quali_position vs avg_race_position
  - avg_position_gain (positive = improved in race)
  - position_gain_consistency
  - racecraft_score (50 + avg_position_gain * 5, bounded 0-100)

#### WeatherFeatureCalculator
- **Status**: Placeholder for future integration
- **Current Output**: Placeholder features (all 50.0 neutral score)
- **Future**: Wet/variable/temperature performance

### 2.3 FeatureEngineer Orchestrator

**Method**: `generate_features(race_results, qualifying_results, circuit_id, up_to_date)`

**Generated Features**:
1. Driver form score
2. Quali-race gap metrics (5 features)
3. Track-specific features (optional, 5 features if circuit_id provided)
4. Weather features (3 placeholder features)

**Output**: DataFrame with merged features for all drivers

---

## 3. DATA MODELS & DATA FLOW

### 3.1 Data Models (Pydantic)

```
src/f1_predict/data/models.py
├── Circuit: circuit_id, name, location (lat/long/country)
├── Location: geographic information
├── Constructor: team_id, name, nationality
├── Driver: driver_id, name, DOB, nationality, number
├── Race: season, round, date, circuit, sessions
├── Session: practice/qualifying/sprint with times
└── [More race-related models...]
```

### 3.2 Data Collection Pipeline

```
src/f1_predict/data/collector.py [F1DataCollector]
├── collect_all_data()
│   ├── Calls Ergast API (rate limited: 4 req/sec)
│   ├── Seasons: 2020-2024 (configurable)
│   ├── Data types:
│   │   ├── Race results (position, points, status)
│   │   ├── Qualifying results
│   │   ├── Race schedules
│   │   └── Driver/constructor/circuit metadata
│   ├── Storage: data/raw/ (CSV + JSON)
│   └── Error handling: Continues on individual race failures

├── collect_and_clean_all_data()
│   ├── Orchestrates collection + cleaning in pipeline
│   ├── Quality validation (thresholds configurable)
│   └── Generates quality reports

└── Error recovery: retry logic, skip mechanisms
```

### 3.3 Data Cleaning Pipeline

```
src/f1_predict/data/cleaning.py
├── DataCleaner
│   ├── Handle missing data (configurable defaults)
│   ├── Standardize names (driver/team/circuit mappings)
│   ├── Type conversions (strings → numbers, date formats)
│   ├── Business rule validation
│   └── Quality report generation

└── DataQualityValidator
    ├── Duplicate detection (within races)
    ├── Position consistency checks
    ├── Data completeness scoring
    └── Comprehensive quality reports
```

### 3.4 Data Flow Diagram

```
┌─────────────────┐
│   Ergast API    │ (F1 Historical Data 2020-2024)
└────────┬────────┘
         │ [Rate Limited: 4 req/sec]
         ↓
┌──────────────────────────────┐
│   F1DataCollector            │
│   - Raw API calls            │
│   - Season/round iteration   │
│   - CSV/JSON export          │
└────────┬─────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│   data/raw/                  │
│   - race_results_*.csv       │
│   - qualifying_results_*.csv │
│   - race_schedules_*.csv     │
└────────┬─────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│   DataCleaner                │
│   - Missing value handling   │
│   - Name standardization     │
│   - Type conversion          │
│   - Validation               │
└────────┬─────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│   data/processed/            │
│   - Cleaned CSV/JSON         │
│   - Quality reports          │
└────────┬─────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│   FeatureEngineer            │
│   - Form scores              │
│   - Team reliability         │
│   - Track performance        │
│   - Quali-race gaps          │
│   - Weather features         │
└────────┬─────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│   Feature Matrix             │
│   (rows: drivers × races)    │
│   (cols: engineered features)│
└────────┬─────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│   ML Models                  │
│   - Training phase: fit()    │
│   - Prediction: predict()    │
│   - Evaluation: metrics      │
└──────────────────────────────┘
```

---

## 4. MODEL TRAINING & EVALUATION FRAMEWORK

### 4.1 ModelEvaluator Class

```python
src/f1_predict/models/evaluation.py [331 lines]

Methods:
├── evaluate(y_true, y_pred, y_proba)
│   └── Returns: accuracy, precision, recall, f1_score, roc_auc
│
├── cross_validate(model, features, race_results)
│   ├── K-fold cross-validation (default: 5 splits)
│   ├── Per-fold metrics + aggregates
│   ├── Feature importance averaging
│   └── Returns: mean_metrics, std_metrics, fold_metrics
│
├── evaluate_confidence_calibration(y_true, y_proba, n_bins=10)
│   ├── Binning predictions [0, 1]
│   ├── Per-bin metrics (predicted vs actual)
│   └── Expected Calibration Error (ECE)
│
└── compare_models(models_dict, features, race_results)
    └── Returns: DataFrame comparing all models
```

### 4.2 Model Training Pattern (All Models)

All models follow identical training interface:

```python
# 1. Initialize
model = XGBoostRacePredictor(target="podium")

# 2. Fit
model.fit(features_df, race_results_df)

# 3. Predict (probabilities)
probs = model.predict_proba(test_features)

# 4. Predict (with threshold)
predictions = model.predict(test_features, threshold=0.5)
# Returns: DataFrame with [driver_id, predicted_outcome, confidence]

# 5. Feature importance
importances = model.get_feature_importance()

# 6. Persistence
model.save("path/to/model.pkl")
loaded_model = XGBoostRacePredictor.load("path/to/model.pkl")
```

### 4.3 Model Configuration

All models support:
- **Feature Scaling**: StandardScaler (fitted on training set)
- **Early Stopping**: Validation-based convergence
- **Hyperparameter Tuning**: All params configurable
- **Feature Names Tracking**: For prediction alignment
- **Binary Classification**: Configurable targets (podium/points/win)

---

## 5. ENSEMBLE SYSTEM

### 5.1 EnsemblePredictor Architecture

```python
src/f1_predict/models/ensemble.py [259 lines]

Components:
├── Individual Models (list[Any])
│   └── Any model with predict() or predict_proba()
│
├── Voting Strategies
│   ├── Soft (default): Weighted average of probabilities
│   └── Hard: Majority vote with weighted voting
│
└── Weights (list[float])
    └── Normalized automatically, default: equal weights
```

### 5.2 Ensemble Methods

```python
ensemble = EnsemblePredictor(
    models=[xgb_model, lgb_model, rf_model],
    weights=[0.4, 0.3, 0.3],  # XGB weighted higher
    voting="soft"
)

# Soft voting: weighted average of probabilities
probs = ensemble.predict_proba(features)  # Returns: np.ndarray

# Predictions with thresholding
predictions = ensemble.predict(features, threshold=0.5)
# Returns: DataFrame[driver_id, predicted_outcome, confidence]

# Model agreement score (0-1)
agreement = ensemble.get_model_agreement(features)
# High agreement = high confidence in ensemble
```

### 5.3 Persistence

- Individual models saved to: `{ensemble_path}_models/model_0.pkl`, etc.
- Ensemble metadata: weights, voting strategy
- Load reconstructs ensemble from saved models

---

## 6. WEB INTERFACE INTEGRATION

### 6.1 Prediction Manager (Web Utilities)

```
src/f1_predict/web/utils/prediction.py [286 lines]

PredictionManager:
├── load_model(model_type)
│   ├── Types: 'ensemble', 'xgboost', 'lightgbm', 'random_forest'
│   ├── Creates model instances
│   └── Returns: (model, metadata_dict)
│
├── get_upcoming_races()
│   └── Returns: list[dict] with race metadata
│
├── prepare_race_features(race_id, season)
│   ├── Loads race results
│   ├── Calculates driver form scores
│   └── Returns: Feature matrix for prediction
│
├── generate_prediction(model, features, race_name)
│   ├── Executes model predictions
│   ├── Sorts by confidence
│   └── Returns: dict with podium predictions
│
├── format_prediction_results(prediction, drivers_info)
│   └── Returns: DataFrame formatted for Streamlit display
│
└── export_prediction(prediction, format)
    └── CSV or JSON export
```

### 6.2 Web Pages Using Models

- `web/pages/predict.py`: Race outcome prediction UI
- `web/pages/explainability.py`: SHAP-based model explanation
- `web/pages/analytics.py`: Performance metrics & analysis
- `web/pages/compare.py`: Model comparison interface

---

## 7. TESTING INFRASTRUCTURE

### 7.1 Test Organization

```
tests/
├── models/
│   ├── test_baseline.py          [Mocked]
│   ├── test_logistic.py          [Fixture data]
│   ├── test_random_forest.py     [Mocked API]
│   ├── test_xgboost.py
│   ├── test_lightgbm.py
│   └── test_evaluation.py        [K-fold CV tests]
│
├── features/
│   └── test_engineering.py       [Form/reliability/gap calcs]
│
├── data/
│   └── test_models.py            [Pydantic validation]
│
└── web/
    ├── test_prediction.py        [PredictionManager]
    └── conftest.py               [Shared fixtures]
```

### 7.2 Key Testing Patterns

- **Mocking**: All external API calls (Ergast) mocked
- **Fixtures**: Sample race/driver data in `tests/fixtures/`
- **Markers**: `@pytest.mark.slow`, `@pytest.mark.integration`
- **Coverage**: pytest-cov with 80%+ target
- **Parallel**: pytest-xdist enabled

---

## 8. KEY DATA STRUCTURES

### 8.1 Feature Matrix Structure

Input features for ML models:

```
Features DataFrame (rows: drivers, columns: features):
├── driver_id: unique identifier
├── qualifying_position: Q position for race
├── form_score: driver form (0-100) calculated from last 5 races
├── driver_form_score: normalized form
├── avg_quali_position: average Q position last 10 races
├── avg_race_position: average race position last 10 races
├── avg_position_gain: qualifying → race improvement
├── position_gain_consistency: std dev of gains
├── racecraft_score: ability to gain positions in race (0-100)
├── track_performance_score: historical performance at circuit
├── wet_performance_score: placeholder (currently 50.0)
├── variable_conditions_score: placeholder
└── temperature_adaptation_score: placeholder
```

### 8.2 Prediction Output Structure

```python
# Model predictions return:
DataFrame {
    'driver_id': str,
    'predicted_outcome': int (0 or 1),
    'confidence': float (0-100%)
}

# Ensemble/complex predictions:
Dict {
    'race': str,
    'predictions': [
        {
            'position': int,
            'driver_id': str,
            'confidence': float
        },
        ...
    ],
    'podium': [(driver_id, confidence), ...]
}
```

---

## 9. TIME SERIES & HISTORICAL HANDLING

### 9.1 Current Approach

Models use **sliding window** for form calculation:

```python
# Driver form (default window_size=5)
recent_races = race_results[race_results['driver_id'] == driver_id]
recent_races = recent_races.sort_values('date').tail(5)

# Recency weighting (exponential decay)
weights = [0.7**i for i in range(n-1, -1, -1)]
```

### 9.2 Temporal Features

- **Driver Form**: Last 5 races with exponential recency weighting
- **Team Reliability**: Last 10 races
- **Track Performance**: All races at circuit (min 2 required)
- **Quali-Race Gap**: Last 10 races

### 9.3 No Explicit Time Series Models

- ⚠️ **Gap**: No LSTM, ARIMA, or other sequence models
- 🎯 **Opportunity**: Time series forecasting for season progression

---

## 10. CURRENT INTEGRATION POINTS

### 10.1 Model Loading Flow

```
Web App (Streamlit)
    ↓
prediction.py:PredictionManager
    ↓
models/__init__.py [Imports all 6 models]
    ↓
Specific model (e.g., XGBoostRacePredictor)
    ↓
predict() / predict_proba()
    ↓
Results to web UI
```

### 10.2 Data Flow to Models

```
Raw F1 Data (2020-2024)
    ↓ [F1DataCollector]
data/raw/ (CSV/JSON)
    ↓ [DataCleaner]
data/processed/
    ↓ [FeatureEngineer]
Feature Matrix
    ↓ [Model.fit()]
Trained Model
    ↓ [Model.predict()]
Predictions → Web UI / Export
```

### 10.3 Configuration Points

- Feature engineer windows (form=5, team=10, gap=10)
- Model hyperparameters (learning_rate, n_estimators, depth)
- Prediction thresholds (default 0.5)
- Ensemble weights (default equal)

---

## 11. PERFORMANCE CHARACTERISTICS

### 11.1 Model Accuracy (README)

- **Overall**: 75-87% (varies by target)
- **Podium**: Typically 78-85%
- **Points**: Typically 76-82%
- **Win**: Typically 72-79% (hardest to predict)

### 11.2 Training Time (on modern hardware)

- **Logistic**: <1 second
- **Random Forest**: 1-3 seconds
- **XGBoost**: 2-5 seconds
- **LightGBM**: 1-4 seconds
- **Ensemble**: 5-15 seconds (includes model creation)

### 11.3 Feature Importance Ranking

(Typical from model explainability)
1. form_score (strongest signal)
2. racecraft_score (quali→race improvement)
3. qualifying_position (direct correlation)
4. track_performance_score
5. position_gain_consistency

---

## 12. ARCHITECTURAL PATTERNS

### 12.1 Common Patterns

**Composition Over Inheritance**:
- BasePredictor abstraction (implicit via duck typing)
- All models implement: fit(), predict(), predict_proba(), save(), load()
- EnsemblePredictor combines models without hierarchy

**Factory Pattern**:
- PredictionManager.load_model(type) creates instances
- EnsemblePredictor.load() deserializes and reconstructs

**Strategy Pattern**:
- Voting strategies (soft/hard) in EnsemblePredictor
- Target strategies (podium/points/win) in all models

**Scaler Persistence**:
- StandardScaler fitted on training, saved with model
- Applied consistently during prediction (no data leakage)

### 12.2 Error Handling

All models validate:
- Empty DataFrames before operations
- Feature alignment (features.columns vs model.feature_names)
- Target validity (podium/points/win)
- Model state (fitted before predict)

---

## 13. EXTERNAL DEPENDENCIES

### Core ML Libraries
- **sklearn**: StandardScaler, preprocessing, metrics, RandomForest
- **xgboost**: 1.7.0+, binary classification, feature importance
- **lightgbm**: 4.0.0+, leaf-based trees, parallel training
- **pandas**: 2.0.0+, DataFrames, merging, I/O
- **numpy**: 1.24.0+, array operations

### Data/Serialization
- **pickle**: Model persistence (native)
- **pydantic**: 2.0.0+, data validation
- **pandera**: 0.17.0+, schema validation (optional)

### Web Integration
- **streamlit**: Web UI (web/pages/*.py)
- **plotly**: 5.15.0+, interactive visualization
- **shap**: 0.43.0+, model explainability

---

## 14. MONITORING & VERSIONING (Issue #36)

### Current Status
- Feature branch: `feature/model-monitoring-issue-36`
- Commit: 8d86357 "Implement comprehensive model monitoring & drift detection"

### New Utilities
```
src/f1_predict/web/utils/
├── monitoring.py           [Model performance tracking]
├── drift_detection.py      [Data/prediction drift]
├── model_versioning.py     [Version control]
├── ab_testing.py          [A/B test framework]
└── alerting.py            [Threshold-based alerts]
```

### Monitoring Capabilities
- Real-time performance metrics
- Data drift detection (feature distributions)
- Prediction drift detection (confidence shifts)
- Model versioning & comparison
- A/B testing support
- Alert thresholds and notifications

---

## 15. QUICK REFERENCE: KEY FILES

| Component | File | LOC | Purpose |
|-----------|------|-----|---------|
| **Models** | models/__init__.py | 31 | Exports all 6 models |
| | ensemble.py | 259 | Voting & weighted ensemble |
| | evaluation.py | 331 | Cross-validation, metrics |
| **Features** | engineering.py | 633 | Form, reliability, gaps |
| **Data** | collector.py | - | Ergast API integration |
| | cleaning.py | - | Data validation pipeline |
| | models.py | - | Pydantic data structures |
| **Web** | prediction.py | 286 | PredictionManager |
| | app.py | - | Streamlit main app |
| | pages/*.py | - | Prediction, analytics, explainability |
| **Tests** | models/test_*.py | - | Unit & integration tests |

---

## SUMMARY

**F1 Predict** is a well-architected ML system with:
- ✅ 6 production-ready models (baseline to ensemble)
- ✅ Comprehensive feature engineering (form, reliability, gaps)
- ✅ Clean data flow (Ergast → clean → features → models)
- ✅ Strong testing (80%+ coverage)
- ✅ Web integration (Streamlit predictions, explainability)
- ✅ Model persistence (save/load with metadata)
- 🔄 New monitoring infrastructure (Issue #36 in progress)

**Opportunities for Enhancement**:
- Time series models (LSTM for season progression)
- Real-time drift detection automation
- Multi-objective optimization (podium + consistency)
- Advanced feature engineering (weather integration)
- Model retraining scheduler
- Prediction confidence calibration

