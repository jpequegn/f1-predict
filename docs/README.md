# F1 Predict Documentation

This directory contains comprehensive documentation for the F1 Predict ML system.

## Quick Navigation

### For New Developers
Start here to understand the codebase structure and architecture:

1. **[ARCHITECTURE_SUMMARY.md](./ARCHITECTURE_SUMMARY.md)** - Visual overview of the system
   - Directory structure
   - Data flow pipeline diagram
   - Model comparison matrix
   - Feature engineering breakdown
   - Integration points

2. **[CODEBASE_ANALYSIS.md](./CODEBASE_ANALYSIS.md)** - Deep technical analysis (15 sections)
   - All 6 ML models in detail
   - Feature engineering pipeline
   - Data models and flow
   - Training & evaluation framework
   - Ensemble system
   - Web integration
   - Testing infrastructure
   - Time series handling
   - Performance characteristics
   - Monitoring & versioning (Issue #36)

### Quick Reference Tables
All available in both documents:
- Model comparison (type, speed, accuracy)
- Feature calculator specifications
- Performance metrics (accuracy, latency)
- Configuration parameters
- Testing structure

## System Overview

**F1 Predict** is a production-ready ML system that predicts Formula 1 race outcomes with 75-87% accuracy.

### Key Components
```
Data Collection   →  Data Cleaning  →  Feature Engineering  →  ML Models  →  Web UI
(Ergast API)         (Validation)        (Form, Reliability)     (6 models)   (Streamlit)
                                                                  (Ensemble)
```

### 6 Available Models
1. **RuleBasedPredictor** - Heuristic baseline (instant, very interpretable)
2. **LogisticRacePredictor** - Linear model (fast, interpretable)
3. **RandomForestRacePredictor** - Ensemble of decision trees
4. **XGBoostRacePredictor** - Gradient boosting (fast, accurate)
5. **LightGBMRacePredictor** - Leaf-based boosting (efficient)
6. **EnsemblePredictor** - Meta-learner combining multiple models

### Prediction Types
All models support:
- **Podium**: Position ≤ 3 (top 3 finishers)
- **Points**: Position ≤ 10 (points-scoring)
- **Win**: Position = 1 (race winner)

## Feature Engineering

The system engineers 14+ features per driver-race combination:

1. **Driver Form** (1 feature)
   - Last 5 races with recency weighting
   - Includes consistency penalty and DNF rate

2. **Quali-Race Gap** (5 features)
   - Qualifying vs race position comparison
   - Racecraft score (ability to gain positions)

3. **Track Performance** (5 features, optional)
   - Historical performance at specific circuit
   - Average position and points at track

4. **Weather** (3 features, placeholder)
   - Future integration planned

5. **Team Reliability** (5 features, optional)
   - Finish rate, mechanical failure rate
   - Points consistency

## Data Flow

```
Ergast API (2020-2024)
    ↓ [F1DataCollector - rate limited 4 req/sec]
data/raw/ (CSV/JSON)
    ↓ [DataCleaner - validation + quality scoring]
data/processed/ (Cleaned data)
    ↓ [FeatureEngineer - orchestrator]
Feature Matrix (drivers × features)
    ↓ [Model.fit() / predict()]
Predictions + Confidence
    ↓ [Web UI, Export, Monitoring]
```

## Testing

- **Coverage**: 80%+ with pytest-cov
- **Approach**: Mocked APIs, fixture data, parametrized tests
- **Organization**: Unit, integration, and visual tests
- **Speed**: Parallel execution with pytest-xdist

## Web Interface (Streamlit)

- **Home**: Project overview
- **Predict**: Race outcome predictions
- **Explainability**: SHAP feature importance
- **Analytics**: Model performance metrics
- **Settings**: Configuration management
- **Chat**: AI-powered insights (Claude/OpenAI/Local)

## Monitoring (Issue #36)

Recently implemented comprehensive monitoring:
- Performance tracking
- Data drift detection
- Prediction drift detection
- A/B testing framework
- Alerting system

## File Structure

```
docs/
├── README.md                  [This file]
├── ARCHITECTURE_SUMMARY.md    [Visual overview]
├── CODEBASE_ANALYSIS.md       [Deep technical analysis]
├── schemas/                   [Data schema documentation]
├── tutorials/                 [How-to guides]
└── architecture/              [Detailed architecture docs]
```

## Key Metrics

- **Model Accuracy**: 75-87% (varies by target)
- **Training Time**: 1-15 seconds (model-dependent)
- **Prediction Latency**: <100ms per race
- **Data Coverage**: 2020-2024 (5 seasons, ~105 races)
- **Samples**: ~2,100 driver-race combinations

## Common Tasks

### Load a Model
```python
from f1_predict.models import XGBoostRacePredictor
model = XGBoostRacePredictor.load("path/to/model.pkl")
predictions = model.predict(features)
```

### Generate Features
```python
from f1_predict.features.engineering import FeatureEngineer
engineer = FeatureEngineer()
features = engineer.generate_features(race_results, qualifying_results)
```

### Evaluate Models
```python
from f1_predict.models import ModelEvaluator
evaluator = ModelEvaluator()
cv_results = evaluator.cross_validate(model, features, results)
```

### Use Ensemble
```python
from f1_predict.models import EnsemblePredictor
ensemble = EnsemblePredictor(
    models=[xgb, lgb, rf],
    weights=[0.4, 0.3, 0.3],
    voting="soft"
)
predictions = ensemble.predict(features)
```

## Configuration

Key parameters you can customize:

**Feature Engineering**:
- `driver_form_window`: 5 (races to look back)
- `team_reliability_window`: 10
- `quali_race_window`: 10

**Model Training**:
- `target`: "podium"/"points"/"win"
- `learning_rate`: 0.1
- `n_estimators`: 100
- `early_stopping_rounds`: 10

**Prediction**:
- `threshold`: 0.5 (binary decision boundary)
- `voting`: "soft"/"hard" (ensemble strategy)
- `weights`: [0.4, 0.3, 0.3] (model weights)

## For More Details

- **Component Details**: See [CODEBASE_ANALYSIS.md](./CODEBASE_ANALYSIS.md)
- **Visual Overview**: See [ARCHITECTURE_SUMMARY.md](./ARCHITECTURE_SUMMARY.md)
- **Source Code**: `src/f1_predict/`
- **Tests**: `tests/`

## Recent Updates

**Issue #36**: Comprehensive model monitoring & drift detection
- Real-time performance tracking
- Data/prediction drift detection
- A/B testing support
- Alerting framework

---

Last updated: October 2024
Python: 3.9-3.12
Test Coverage: 80%+
