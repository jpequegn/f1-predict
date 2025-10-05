# API Reference

Complete API reference for the F1 Prediction System.

## Data Collection

### F1DataCollector

Main interface for collecting historical F1 data.

**Location**: `src/f1_predict/data/collector.py`

```python
from f1_predict.data.collector import F1DataCollector

collector = F1DataCollector(data_dir="data")
```

#### Methods

##### `collect_race_results(seasons=None, force_refresh=False)`

Collect historical race results.

```python
results_path = collector.collect_race_results(
    seasons=[2023, 2024],
    force_refresh=False
)
# Returns: Path to CSV file
```

**Parameters**:
- `seasons` (list[int], optional): Seasons to collect (default: 2020-2024)
- `force_refresh` (bool): Re-download existing data (default: False)

**Returns**: `str` - Path to race results CSV file

---

##### `collect_and_clean_all_data(force_refresh=False, enable_enrichment=False)`

Collect and clean all data types in one operation.

```python
results = collector.collect_and_clean_all_data(
    force_refresh=False,
    enable_enrichment=True
)
```

**Parameters**:
- `force_refresh` (bool): Re-download existing data
- `enable_enrichment` (bool): Include external data (weather, tracks)

**Returns**: `dict` - Collection status report

---

## Models

### RandomForestRacePredictor

Random Forest model for race predictions.

**Location**: `src/f1_predict/models/random_forest.py`

```python
from f1_predict.models.random_forest import RandomForestRacePredictor

model = RandomForestRacePredictor(
    target_type='podium',
    n_estimators=100,
    max_depth=None
)
```

#### Parameters

- `target_type` (str): 'podium', 'points', or 'win'
- `n_estimators` (int): Number of trees (default: 100)
- `max_depth` (int, optional): Maximum tree depth
- `random_state` (int): Random seed (default: 42)

#### Methods

##### `fit(features)`

Train the model on historical data.

```python
model.fit(train_features)
```

**Parameters**:
- `features` (pd.DataFrame): Training data with engineered features

**Returns**: `RandomForestRacePredictor` - Self for chaining

---

##### `predict(features)`

Generate predictions.

```python
predictions = model.predict(test_features)
```

**Parameters**:
- `features` (pd.DataFrame): Test data

**Returns**: `np.ndarray` - Binary predictions (0/1)

---

##### `predict_proba(features)`

Generate probability estimates.

```python
probabilities = model.predict_proba(test_features)
```

**Parameters**:
- `features` (pd.DataFrame): Test data

**Returns**: `np.ndarray` - Probability estimates (shape: n_samples x 2)

---

##### `save(path)` / `load(path)`

Persist and load trained models.

```python
# Save
model.save('models/random_forest_podium.pkl')

# Load
loaded_model = RandomForestRacePredictor.load('models/random_forest_podium.pkl')
```

---

### XGBoostRacePredictor

XGBoost gradient boosting model.

**Location**: `src/f1_predict/models/xgboost_model.py`

```python
from f1_predict.models.xgboost_model import XGBoostRacePredictor

model = XGBoostRacePredictor(
    target_type='podium',
    n_estimators=100,
    learning_rate=0.1
)
```

Interface matches `RandomForestRacePredictor`.

---

### EnsemblePredictor

Combines multiple models with weighted voting.

**Location**: `src/f1_predict/models/ensemble.py`

```python
from f1_predict.models.ensemble import EnsemblePredictor

ensemble = EnsemblePredictor(voting='soft')
ensemble.add_model(LogisticRacePredictor(), weight=0.15)
ensemble.add_model(RandomForestRacePredictor(), weight=0.3)
ensemble.fit(train_data)
```

---

## Feature Engineering

### FeatureEngineer

Creates features from raw race data.

**Location**: `src/f1_predict/features/engineering.py`

```python
from f1_predict.features.engineering import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.create_basic_features(raw_data)
```

#### Methods

##### `create_basic_features(data)`

Generate standard feature set.

```python
features = engineer.create_basic_features(race_results_df)
```

**Features Created**:
- `grid_position`: Qualifying position
- `driver_form_*`: Recent performance metrics
- `constructor_points`: Team championship position
- `circuit_experience`: Driver history at circuit
- `season_progress`: Position in season

**Returns**: `pd.DataFrame` - Engineered features

---

## Model Evaluation

### ModelEvaluator

Evaluate model performance.

**Location**: `src/f1_predict/models/evaluation.py`

```python
from f1_predict.models.evaluation import ModelEvaluator

evaluator = ModelEvaluator(model)
metrics = evaluator.evaluate(test_data, y_true)
```

#### Methods

##### `evaluate(features, y_true)`

Calculate performance metrics.

```python
metrics = evaluator.evaluate(test_features, test_labels)
```

**Returns**: `dict` with keys:
- `accuracy`: Overall accuracy
- `precision`: Precision score
- `recall`: Recall score
- `f1_score`: F1 score
- `roc_auc`: ROC AUC score

---

## CLI Commands

### Data Collection

```bash
# Collect all data types
f1-predict collect --type all

# Collect specific type
f1-predict collect --type race-results

# With external data enrichment
f1-predict collect --type all --enrich

# Force refresh existing data
f1-predict collect --type all --refresh
```

### Data Cleaning

```bash
# Clean all data with strict validation
f1-predict clean --type all --strict

# Clean specific data type
f1-predict clean --type race-results

# Validate without cleaning
f1-predict validate --type all
```

### Status

```bash
# View collection status
f1-predict status
```

---

## API Clients

### ErgastAPIClient

Client for Ergast F1 API.

**Location**: `src/f1_predict/api/ergast.py`

```python
from f1_predict.api.ergast import ErgastAPIClient

with ErgastAPIClient() as client:
    races = client.get_race_results(season=2024, round=1)
```

Built-in rate limiting: 4 requests/second

---

### WeatherAPIClient

Client for OpenWeatherMap API.

**Location**: `src/f1_predict/api/weather.py`

```python
from f1_predict.api.weather import WeatherAPIClient

client = WeatherAPIClient(api_key="your_key")
weather = client.get_current_weather(lat=43.7347, lon=7.4206)
```

Free tier: 60 requests/minute

---

## Type Definitions

### Common Types

```python
from typing import TypeAlias

# Prediction targets
TargetType: TypeAlias = Literal["podium", "points", "win"]

# Model paths
ModelPath: TypeAlias = str | Path

# Season years
SeasonYear: TypeAlias = int  # 1950-2100
```

### Protocols

```python
from typing import Protocol
import pandas as pd

class PredictorProtocol(Protocol):
    """Interface all models must implement."""

    def fit(self, data: pd.DataFrame) -> "PredictorProtocol":
        """Train model."""
        ...

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate predictions."""
        ...

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Generate probability estimates."""
        ...
```

---

## Examples

### Complete Prediction Workflow

```python
from f1_predict.data.collector import F1DataCollector
from f1_predict.features.engineering import FeatureEngineer
from f1_predict.models.ensemble import EnsemblePredictor
from f1_predict.models.evaluation import ModelEvaluator
import pandas as pd

# 1. Collect data
collector = F1DataCollector()
collector.collect_and_clean_all_data()

# 2. Load data
data = pd.read_csv('data/processed/race_results_cleaned.csv')

# 3. Engineer features
engineer = FeatureEngineer()
features = engineer.create_basic_features(data)

# 4. Split data
train = features[features['season'] < 2024]
test = features[features['season'] == 2024]

# 5. Train model
model = EnsemblePredictor()
model.fit(train)

# 6. Predict
predictions = model.predict(test)
probabilities = model.predict_proba(test)

# 7. Evaluate
evaluator = ModelEvaluator(model)
metrics = evaluator.evaluate(test, test['podium'])
print(f"Accuracy: {metrics['accuracy']:.1%}")

# 8. Save model
model.save('models/ensemble_podium.pkl')
```

---

## Error Handling

### Common Exceptions

```python
from f1_predict.exceptions import (
    DataCollectionError,
    ModelNotFittedError,
    InvalidTargetError
)

try:
    model.predict(features)
except ModelNotFittedError:
    print("Model must be fitted before prediction")
except InvalidTargetError as e:
    print(f"Invalid target type: {e}")
```

---

## References

- [Architecture Overview](architecture/overview.md)
- [Data Schemas](schemas/data_models.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Tutorials](tutorials/)
