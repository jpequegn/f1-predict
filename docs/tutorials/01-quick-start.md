# Quick Start Guide

**Estimated time**: 10-15 minutes

This tutorial will guide you through making your first F1 race prediction using the f1-predict package.

## Prerequisites

- Python 3.9 or higher installed
- Basic familiarity with command line
- 10-15 minutes of time

## Step 1: Installation

### Install uv (Recommended)

```bash
# On macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/jpequegn/f1-predict.git
cd f1-predict

# Install dependencies
uv sync --dev

# Verify installation
uv run pytest tests/models/test_baseline.py -v
```

## Step 2: Collect Sample Data

The package includes data collection tools to fetch historical F1 data:

```bash
# Collect race data for 2020-2024
uv run python -c "
from f1_predict.data.collector import F1DataCollector

collector = F1DataCollector(data_dir='data')
results = collector.collect_race_results()
print(f'Collected data saved to: {results}')
"
```

**Expected output**: `Collected data saved to: data/raw/race_results_2020_2024.csv`

## Step 3: Train Your First Model

Let's train a simple baseline model to get familiar with the workflow:

```python
# save as: train_model.py
from f1_predict.models.baseline import RuleBasedPredictor
from f1_predict.data.collector import F1DataCollector
import pandas as pd

# Load data
data = pd.read_csv('data/raw/race_results_2020_2024.csv')

# Prepare features
recent_data = data[data['season'].astype(int) >= 2023].copy()

# Initialize and use model
model = RuleBasedPredictor()

# Get recent race for prediction
latest_race = recent_data.groupby(['season', 'round']).first().tail(1)

predictions = model.predict(
    drivers=recent_data['driver_id'].unique()[:10],  # Top 10 drivers
    circuit_id=latest_race.iloc[0]['circuit_id']
)

print("\nPredicted Race Results:")
print("-" * 50)
for i, driver in enumerate(predictions[:5], 1):
    print(f"{i}. {driver}")
```

Run it:

```bash
uv run python train_model.py
```

**Expected output**:
```
Predicted Race Results:
--------------------------------------------------
1. max_verstappen
2. perez
3. leclerc
4. sainz
5. hamilton
```

## Step 4: Use a Machine Learning Model

Now let's try a real ML model - Logistic Regression:

```python
# save as: train_ml_model.py
from f1_predict.models.logistic import LogisticRacePredictor
from f1_predict.features.engineering import FeatureEngineer
import pandas as pd

# Load and prepare data
data = pd.read_csv('data/raw/race_results_2020_2024.csv')

# Create features
engineer = FeatureEngineer()
features_df = engineer.create_basic_features(data)

# Split data
train_data = features_df[features_df['season'].astype(int) < 2024]
test_data = features_df[features_df['season'].astype(int) == 2024]

# Train model
model = LogisticRacePredictor(target_type='podium')
model.fit(train_data)

# Make predictions
if len(test_data) > 0:
    predictions = model.predict(test_data.head(20))
    probabilities = model.predict_proba(test_data.head(20))

    print("\nPodium Predictions:")
    print("-" * 60)
    print(f"{'Driver':<20} {'Predicted':<12} {'Probability':<12}")
    print("-" * 60)

    for idx, (pred, prob) in enumerate(zip(predictions[:10], probabilities[:10])):
        driver = test_data.iloc[idx]['driver_id']
        print(f"{driver:<20} {'Yes' if pred == 1 else 'No':<12} {prob[1]:.1%}")

# Evaluate model
from f1_predict.models.evaluation import ModelEvaluator

evaluator = ModelEvaluator(model)
metrics = evaluator.evaluate(test_data, test_data['podium'])

print(f"\nModel Performance:")
print(f"Accuracy: {metrics['accuracy']:.1%}")
print(f"Precision: {metrics['precision']:.1%}")
print(f"Recall: {metrics['recall']:.1%}")
```

Run it:

```bash
uv run python train_ml_model.py
```

## Step 5: Compare Models

The package includes multiple models. Let's compare them:

```python
# save as: compare_models.py
from f1_predict.models.baseline import RuleBasedPredictor
from f1_predict.models.logistic import LogisticRacePredictor
from f1_predict.models.random_forest import RandomForestRacePredictor
from f1_predict.models.evaluation import ModelEvaluator
from f1_predict.features.engineering import FeatureEngineer
import pandas as pd

# Load data
data = pd.read_csv('data/raw/race_results_2020_2024.csv')
engineer = FeatureEngineer()
features_df = engineer.create_basic_features(data)

# Split data
train_data = features_df[features_df['season'].astype(int) < 2024]
test_data = features_df[features_df['season'].astype(int) == 2024]

# Train multiple models
models = {
    'Logistic Regression': LogisticRacePredictor(target_type='podium'),
    'Random Forest': RandomForestRacePredictor(target_type='podium', n_estimators=100),
}

print("Model Comparison")
print("=" * 70)
print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12}")
print("=" * 70)

for name, model in models.items():
    model.fit(train_data)
    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate(test_data, test_data['podium'])

    print(f"{name:<25} {metrics['accuracy']:<12.1%} {metrics['precision']:<12.1%} {metrics['recall']:<12.1%}")

print("=" * 70)
```

Run it:

```bash
uv run python compare_models.py
```

## Next Steps

Congratulations! You've made your first F1 predictions. Here's what to explore next:

### Learn More

1. **Model Documentation**: See `docs/models/` for detailed model documentation
2. **Advanced Tutorials**:
   - Data pipeline and cleaning
   - Hyperparameter tuning
   - Ensemble models
   - Feature engineering

3. **API Reference**: See `docs/api-reference.md` for complete API documentation

### Try Advanced Features

```python
# Use XGBoost for better accuracy
from f1_predict.models.xgboost_model import XGBoostRacePredictor

model = XGBoostRacePredictor(
    target_type='win',
    n_estimators=200,
    learning_rate=0.05
)

# Create an ensemble for maximum accuracy
from f1_predict.models.ensemble import EnsemblePredictor

ensemble = EnsemblePredictor(voting='soft')
ensemble.add_model(LogisticRacePredictor())
ensemble.add_model(RandomForestRacePredictor(n_estimators=100))
ensemble.add_model(XGBoostRacePredictor())
ensemble.fit(train_data)
```

### Get Help

- **Troubleshooting**: See `docs/troubleshooting.md`
- **FAQ**: See `docs/faq.md`
- **Issues**: https://github.com/jpequegn/f1-predict/issues

## Summary

You've learned to:
- ✅ Install and setup f1-predict
- ✅ Collect historical F1 data
- ✅ Train baseline and ML models
- ✅ Make race predictions
- ✅ Evaluate model performance
- ✅ Compare multiple models

**Total time**: ~10-15 minutes

Happy predicting! 🏎️
