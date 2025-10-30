# Hyperparameter Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement automated hyperparameter tuning for tree-based models (XGBoost, LightGBM, RandomForest) using Optuna with <1 hour optimization time.

**Architecture:** Modular optimizer framework with SearchSpaceRegistry defining parameter ranges, ObjectiveFunction handling trial evaluation with early stopping, HyperparameterOptimizer orchestrating Optuna study, and MLflow integration.

**Tech Stack:** Optuna (Bayesian optimization), MLflow (experiment tracking), scikit-learn (evaluation), Pydantic (config)

---

## Task 1: Create Optimization Module Structure

**Files:**
- Create: `src/f1_predict/optimization/__init__.py`
- Create: `src/f1_predict/optimization/config.py`
- Create: `src/f1_predict/optimization/search_spaces.py`
- Create: `src/f1_predict/optimization/objectives.py`
- Create: `src/f1_predict/optimization/hyperparameter_optimizer.py`
- Create: `tests/optimization/__init__.py`
- Create: `tests/optimization/test_hyperparameter_optimizer.py`

**Step 1: Create module files**

```bash
mkdir -p src/f1_predict/optimization tests/optimization
touch src/f1_predict/optimization/__init__.py
touch tests/optimization/__init__.py
```

**Step 2: Write __init__.py**

`src/f1_predict/optimization/__init__.py`:
```python
"""Hyperparameter optimization module using Optuna."""

from f1_predict.optimization.hyperparameter_optimizer import (
    HyperparameterOptimizer,
)
from f1_predict.optimization.search_spaces import SearchSpaceRegistry

__all__ = [
    "HyperparameterOptimizer",
    "SearchSpaceRegistry",
]
```

**Step 3: Commit**

```bash
git add src/f1_predict/optimization/__init__.py tests/optimization/__init__.py
git commit -m "chore: create optimization module structure"
```

---

## Task 2: Implement SearchSpaceRegistry

**Files:**
- Create: `src/f1_predict/optimization/search_spaces.py`
- Create: `tests/optimization/test_search_spaces.py`

**Implementation:** Define model-specific hyperparameter ranges for XGBoost, LightGBM, and RandomForest in `SearchSpaceRegistry` class with `get_search_space(model_type)` method.

**Test Coverage:** Test valid spaces for all 3 models, invalid model type error handling.

---

## Task 3: Write ObjectiveFunction

**Files:**
- Create: `src/f1_predict/optimization/objectives.py`
- Create: `tests/optimization/test_objectives.py`

**Implementation:** `ObjectiveFunction` class with methods:
- `optimize_xgboost(trial, X_train, y_train, X_val, y_val)` → accuracy
- `optimize_lightgbm(...)` → accuracy
- `optimize_random_forest(...)` → accuracy

Error handling for NaN values, invalid hyperparams.

---

## Task 4: Main HyperparameterOptimizer Class

**Files:**
- Create: `src/f1_predict/optimization/hyperparameter_optimizer.py`
- Create: `tests/optimization/test_hyperparameter_optimizer.py`

**Implementation:**
- `__init__(model_type, study_name, n_trials, timeout_seconds)`
- `optimize(X_train, y_train, X_val, y_val)` → returns (best_params, best_model)
- Uses Optuna with TPESampler and SuccessiveHalvingPruner
- Supports xgboost, lightgbm, random_forest

---

## Task 5: MLflow Integration

**Files:**
- Create: `src/f1_predict/optimization/mlflow_integration.py`
- Create: `tests/optimization/test_mlflow_integration.py`

**Implementation:** `MLflowTracker` class to log trials and best results to MLflow. Graceful fallback if MLflow unavailable.

---

## Task 6: Configuration Loader

**Files:**
- Create: `src/f1_predict/optimization/config_loader.py`
- Create: `tests/optimization/test_config_loader.py`

**Implementation:** `ConfigLoader` class with:
- `save_best_params(model_type, params, filepath)`
- `load_best_params(model_type, filepath)`
- `get_hyperparameters(model_type, optimized_path)` → uses optimized if available, else defaults

---

## Task 7: Integration Tests

**Files:**
- Create: `tests/optimization/test_integration.py`

**Test:** Full optimization workflow for each model type. Verify:
- Optimization completes
- Best params extracted correctly
- Study stats available
- All 3 models work

---

## Task 8: CLI Command

**Files:**
- Create: `src/f1_predict/optimization/cli.py`
- Modify: `src/f1_predict/cli_enhanced.py`

**Implementation:** Add `optimize run` command with options:
- `--model-type` (xgboost, lightgbm, random_forest)
- `--n-trials` (default 100)
- `--timeout` (default 3600)
- `--output-dir` (default data/optimized_params)

---

## Task 9: Documentation and Examples

**Files:**
- Create: `docs/HYPERPARAMETER_OPTIMIZATION.md`

**Content:**
- Quick start examples
- Supported models
- Architecture overview
- Best practices
- Troubleshooting

---

## Task 10: Final Testing & Coverage

**Run:**
```bash
uv run pytest tests/optimization/ -v --cov=src/f1_predict/optimization --cov-report=term-missing
uv run ruff check src/f1_predict/optimization/ --fix
```

**Expected:** ≥80% coverage, 0 linting errors, all tests passing

---

## Task 11: Push Feature Branch

**Run:**
```bash
git push origin feature/issue-39-hyperparameter-optimization
```

---

## Summary

**Total Tasks:** 11 implementation steps
**Expected Time:** 2-3 hours
**Test Coverage:** ≥80%
**Success Criteria:**
- ✅ 3 tree-based models supported
- ✅ <1 hour optimization time
- ✅ 3-5% accuracy improvement
- ✅ All tests passing
- ✅ Full documentation
- ✅ MLflow integration optional but supported
