# Hyperparameter Optimization Pipeline - Design Document

**Date:** 2025-10-29
**Issue:** #39
**Status:** Design Complete - Ready for Implementation

## Overview

Implement automated hyperparameter tuning using Optuna to systematically optimize tree-based model configurations (XGBoost, LightGBM, RandomForest) and improve prediction accuracy by 3%+.

**Success Criteria:**
- Achieve >3% accuracy improvement over default parameters
- Complete optimization run within 1 hour on available hardware
- Support all tree-based model types with unified interface
- Automatically save and use best configurations in production pipeline
- Track all trials and results in MLflow

## Architecture & Core Components

### File Structure

```
src/f1_predict/optimization/
├── __init__.py
├── hyperparameter_optimizer.py    # Main HyperparameterOptimizer class
├── search_spaces.py                # Model-specific search space definitions
├── objectives.py                   # Trial objective functions
└── config.py                       # Configuration and constants
```

### Core Components

**1. HyperparameterOptimizer** - Main orchestration class
- `__init__(model_type, study_name, n_trials, timeout_seconds)`
- `optimize(X_train, y_train, X_val, y_val)` → runs Optuna study, returns best params + best model
- `load_best_params(model_type)` → retrieves best params from MLflow registry
- Handles trial execution, MLflow logging, early stopping
- Supports: XGBoost, LightGBM, RandomForest

**2. SearchSpaceRegistry** - Parameter space definitions
- `get_search_space(model_type)` → returns Optuna search space definition
- Tree-based parameters: learning_rate, max_depth, n_estimators, min_child_weight, subsample, colsample, reg_alpha, reg_lambda
- Returns consistent structure across all models
- Configurable per model to allow future customization

**3. ObjectiveFunction** - Trial evaluation logic
- `optimize_tree_model(trial, X_train, y_train, X_val, y_val, model_type)`
- Trains model with trial's suggested hyperparameters
- Evaluates on validation set (accuracy for classification, RMSE for regression)
- Implements early stopping (prune unpromising trials after 3 evaluations)
- Returns metric to maximize
- Handles exceptions gracefully (OOM, timeouts, NaN values)

**4. MLflowIntegration** - Experiment tracking
- Log each trial: hyperparameters, metrics (accuracy, training time), model details
- Log best trial: full model artifact, best params JSON, improvement metrics
- Track study metadata: model type, dataset used, total optimization time
- Register best model in MLflow model registry

## Data Flow & Integration

### High-Level Optimization Flow

```
1. Input Stage:
   - Load historical training data (2020-2024 F1 races)
   - Pre-compute splits: Train (60%), Validation (20%), Test (20%)
   - Apply feature preprocessing (scaling, encoding)
   - Store splits in memory for trial reuse

2. Optimization Stage (Bayesian with Early Stopping):
   - Initialize Optuna Study with SuccessiveHalvingPruner
   - For each trial (target: 50-100 trials in 1 hour):
     a. Sample hyperparameters from search space
     b. Train model on training set with those params
     c. Evaluate on validation set
     d. Log params, metrics, duration to MLflow
     e. Pruner evaluates: discard if confidence score too low
   - Continue until timeout (1 hour) or max trials reached

3. Best Model Selection:
   - Optuna identifies best trial (highest validation accuracy)
   - Extract best hyperparameters
   - Retrain best model on train + validation set (maximize data)
   - Evaluate on held-out test set for final performance validation
   - Save best params to JSON config file

4. Integration with Production Pipeline:
   - Best hyperparameters automatically loaded on next model instantiation
   - Update model configuration files (or parameter registry)
   - Next training run uses optimized params by default
   - MLflow tracks: "optimized" vs "default" baseline comparison
```

### Data Handling

- **Pre-computed splits:** Splits computed once, reused across all trials (faster)
- **Single validation split:** Not K-fold (saves time while still reliable)
- **In-memory storage:** Use numpy arrays for fast repeated access
- **Consistent random seed:** Ensures reproducibility across optimization runs

### MLflow Integration Points

```python
# Per-trial logging
mlflow.log_params(trial.params)                    # Log hyperparameters
mlflow.log_metrics({
    "validation_accuracy": val_acc,
    "test_accuracy": test_acc,
    "training_time_seconds": duration,
})
mlflow.log_artifact("trial_model.pkl")             # Save trial model

# Best trial logging
mlflow.log_artifact("best_params.json")            # Save best config
mlflow.register_model(model_uri, "xgboost-optimized-v1")  # Register model
mlflow.set_tag("optimization_study", study_name)  # Tag for filtering
```

### Integration with Existing Training

- **Before optimization:** `model = XGBoost(DEFAULT_PARAMS)`
- **After optimization:** `model = XGBoost(load_optimized_params())`
- No changes to existing training logic
- Config loader handles fallback to defaults if optimized params don't exist
- Gradual rollout possible: test optimized models before full deployment

## Implementation Details & Error Handling

### Search Space Definition

Example for XGBoost (with LightGBM and RandomForest following same pattern):

```python
SEARCH_SPACES = {
    'xgboost': {
        'n_estimators': (100, 500),               # int: num boosting rounds
        'max_depth': (3, 10),                     # int: tree depth limit
        'learning_rate': (0.001, 0.3, 'log'),   # float: shrinkage
        'subsample': (0.5, 1.0),                 # float: row subsampling
        'colsample_bytree': (0.5, 1.0),          # float: column subsampling
        'min_child_weight': (1, 10),             # int: min leaf instances
        'reg_alpha': (0.0, 1.0),                 # float: L1 regularization
        'reg_lambda': (0.0, 1.0),                # float: L2 regularization
    },
    'lightgbm': {...},    # Similar structure with LightGBM-specific params
    'random_forest': {...},  # Similar with RF-specific params
}
```

### Early Stopping Strategy

- **Primary:** Optuna's `SuccessiveHalvingPruner`
  - Aggressively removes bottom 50% of trials after 3 evaluations
  - Saves ~30-40% computation time with minimal accuracy loss
- **Fallback:** Simple threshold pruning
  - If `val_acc < best_so_far - 0.05`, prune trial (likely unstable)

### Parallel Execution

- Optuna with `n_jobs=-1`: Use all available CPU cores
- M1/M2 Mac: ~4-6 parallel trials simultaneously
- Memory-safe: Each trial uses separate model instance
- Shared data: Train/val splits kept in memory (not copied per trial)

### Error Handling

**Data Issues:**
- Missing features → Log warning, prune trial
- Train/val shape mismatch → Raise explicit error with dimensions
- NaN in metrics → Prune trial (indicates instability/overfitting)

**Model Training Issues:**
- Out of memory → Catch OOM, reduce data batch size, prune trial
- Training timeout → Catch timeout, log issue, prune trial
- Invalid hyperparams → Pre-validate before training, skip problematic ranges

**MLflow Integration Issues:**
- Connection failure → Log locally to fallback file, retry on next trial
- Artifact upload failure → Continue optimization, warn user
- Model registration failure → Still save params locally, warn about registry

**Optuna Issues:**
- Study load failure → Create fresh study with warning
- Sampler initialization failure → Fallback to RandomSampler
- Invalid objective value → Prune trial, log exception

### Configuration Format (Saved After Optimization)

```json
{
  "optimization": {
    "timestamp": "2025-10-29T12:00:00Z",
    "model_type": "xgboost",
    "n_trials": 87,
    "optimization_duration_seconds": 3600,
    "best_validation_accuracy": 0.856,
    "test_accuracy": 0.851,
    "improvement_over_baseline": 0.032,
    "baseline_accuracy": 0.824,
    "optimization_framework": "optuna",
    "pruner": "SuccessiveHalvingPruner"
  },
  "best_hyperparameters": {
    "n_estimators": 342,
    "max_depth": 7,
    "learning_rate": 0.087,
    "subsample": 0.92,
    "colsample_bytree": 0.78,
    "min_child_weight": 3,
    "reg_alpha": 0.15,
    "reg_lambda": 0.42
  }
}
```

### Testing Strategy

- **Unit tests:** SearchSpaceRegistry validation (all spaces syntactically correct)
- **Integration tests:** Mini-optimization with 5 trials, 1 model, <30 seconds
- **Fixture data:** Use 1000-row subset of training data for fast test runs
- **Mock MLflow:** Avoid side effects and external dependencies in tests
- **Performance tests:** Verify 50-100 trials complete in <1 hour

## Constraints & Trade-offs

| Constraint | Impact | Mitigation |
|-----------|--------|-----------|
| <1 hour optimization time | Limits trial count, may miss global optimum | Use Bayesian search (smart) not grid search (exhaustive) |
| Single validation split | Less robust than K-fold | Accept slightly higher variance for speed trade-off |
| Tree-based only initially | Doesn't optimize LSTM/ARIMA | Design extensible; can add later |
| MLflow optional | Need fallback if MLflow unavailable | Save params to JSON regardless |
| Memory constraints | Can't afford to keep many full models in memory | Only save best model and trial configs, not all models |

## Future Extensions

1. **Multi-objective optimization:** Optimize for accuracy + speed (inference time)
2. **Neural network support:** LSTM and ARIMA hyperparameter tuning (phase 2)
3. **Warm-starting:** Use previous optimization results as starting point for next run
4. **Distributed optimization:** Scale across multiple machines using Optuna's PostgreSQL backend
5. **AutoML features:** Automatic feature engineering combined with hyperparameter search

## Success Metrics

- ✅ All tree-based models achieve >3% accuracy improvement
- ✅ Full optimization cycle completes within 1 hour
- ✅ Best params automatically loaded and used in production
- ✅ All trials logged to MLflow with full reproducibility
- ✅ Fallback gracefully if MLflow unavailable
- ✅ Early stopping reduces unnecessary trials by 30%+
- ✅ Unit test coverage ≥80% for optimization module
