# Troubleshooting Guide

This guide helps you resolve common issues when using F1 Predict.

## Installation Issues

### Python Version Conflicts

**Symptom**: Error about Python 3.9+ required

**Solution**:
```bash
# Check your Python version
python --version

# If < 3.9, install a newer version
# macOS with Homebrew:
brew install python@3.11

# Or use pyenv for version management:
pyenv install 3.11.0
pyenv local 3.11.0
```

### Dependency Installation Failures

**Symptom**: `pip install` or `uv sync` fails

**Solution**:
```bash
# Try upgrading pip/uv first
python -m pip install --upgrade pip
# or
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clear cache and retry
pip cache purge
uv cache clean

# Install with verbose logging to see exact error
uv sync --dev -v
```

### XGBoost/LightGBM Compilation Errors

#### macOS

**Symptom**: `ld: library not found for -lomp`

**Solution**:
```bash
# Install OpenMP via Homebrew
brew install libomp

# Set environment variables
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

# Reinstall XGBoost
pip install --no-cache-dir xgboost lightgbm
```

#### Windows

**Symptom**: `error: Microsoft Visual C++ 14.0 or greater is required`

**Solution**:
1. Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Or install pre-compiled wheels:
```powershell
pip install xgboost lightgbm --only-binary :all:
```

#### Linux

**Symptom**: `fatal error: omp.h: No such file or directory`

**Solution**:
```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install libgomp1

# RedHat/CentOS
sudo yum install libgomp

# Arch
sudo pacman -S openmp
```

### Virtual Environment Problems

**Symptom**: Packages installed but not found

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows

# Verify you're in the venv
which python  # Should show path in venv/

# Reinstall package in development mode
pip install -e .
```

### Permission Errors

**Symptom**: `PermissionError` when installing

**Solution**:
```bash
# Don't use sudo with pip! Instead:

# Option 1: Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install -e .

# Option 2: User install (not recommended)
pip install --user -e .
```

## Data Collection Issues

### API Connection Failures

**Symptom**: `ConnectionError` or `Timeout` when collecting data

**Solution**:
```python
# Increase timeout and add retry logic
from f1_predict.api.ergast import ErgastAPIClient

client = ErgastAPIClient(
    timeout=60.0,  # Increase from default 30s
    max_retries=5   # Increase retries
)
```

**Check network**:
```bash
# Test API connectivity
curl -I http://ergast.com/api/f1/current/last/results.json

# If fails, check firewall/proxy settings
```

### Rate Limiting Errors

**Symptom**: `HTTP 429` or "Too Many Requests"

**Solution**:
```python
# Built-in rate limiting (4 req/sec) should prevent this
# If you still hit limits, increase delay:

from f1_predict.api.base import BaseAPIClient

client = BaseAPIClient(
    base_url="...",
    rate_limit_requests=2,  # Reduce from 4
    rate_limit_window=1.0
)
```

### Data Parsing Errors

**Symptom**: `ValidationError` or `JSONDecodeError`

**Solution**:
```python
# Enable debug logging to see raw responses
import logging
logging.basicConfig(level=logging.DEBUG)

from f1_predict.data.collector import F1DataCollector

collector = F1DataCollector()
# Check logs for malformed responses
```

**If specific race fails**:
```python
# Skip problematic races
try:
    results = collector.collect_race_results()
except Exception as e:
    print(f"Error: {e}")
    # Continue with partial data
```

### Storage Permission Issues

**Symptom**: `PermissionError` when saving data

**Solution**:
```bash
# Check directory permissions
ls -la data/

# Fix permissions
chmod 755 data/
chmod 644 data/raw/*

# Or use a different directory
from f1_predict.data.collector import F1DataCollector

collector = F1DataCollector(data_dir='/tmp/f1_data')
```

## Model Training Issues

### Memory Errors

**Symptom**: `MemoryError` or system slowdown

**Solution**:
```python
# Reduce dataset size
data_subset = data.sample(frac=0.5)  # Use 50% of data

# Or reduce model complexity
model = RandomForestPredictor(
    n_estimators=50,  # Reduce from 100
    max_depth=10      # Limit tree depth
)

# For XGBoost/LightGBM, reduce memory usage:
model = XGBoostPredictor(
    tree_method='hist',  # Use histogram-based algorithm
    max_bin=128          # Reduce bins
)
```

**Monitor memory**:
```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory: {process.memory_info().rss / 1024**2:.1f} MB")
```

### Training Timeout/Slowness

**Symptom**: Training takes too long

**Solution**:
```python
# 1. Reduce data size
train_recent = train_data[train_data['season'] >= 2022]

# 2. Reduce model complexity
model = RandomForestPredictor(
    n_estimators=50,    # Fewer trees
    n_jobs=-1,          # Use all CPU cores
    max_features='sqrt' # Limit features per split
)

# 3. Use faster algorithm
from f1_predict.models.lightgbm_model import LightGBMPredictor

model = LightGBMPredictor()  # Generally faster than XGBoost
```

### Convergence Warnings

**Symptom**: `ConvergenceWarning: Maximum iterations reached`

**Solution**:
```python
# For Logistic Regression:
model = LogisticRegressionPredictor(
    max_iter=1000,  # Increase from default
    tol=1e-3        # Relax tolerance
)

# Or try different solver:
model.model.solver = 'liblinear'  # For small datasets
# or
model.model.solver = 'sag'  # For large datasets
```

### Feature Mismatch Errors

**Symptom**: `ValueError: Number of features doesn't match`

**Solution**:
```python
# Ensure consistent feature engineering
engineer = FeatureEngineer()

# Use same engineer for train and test
train_features = engineer.create_basic_features(train_data)
test_features = engineer.create_basic_features(test_data)

# Verify shapes
print(f"Train: {train_features.shape}")
print(f"Test: {test_features.shape}")
assert train_features.shape[1] == test_features.shape[1]

# Check for missing columns
missing_cols = set(train_features.columns) - set(test_features.columns)
if missing_cols:
    print(f"Missing in test: {missing_cols}")
```

## Prediction Issues

### Model Not Fitted Errors

**Symptom**: `NotFittedError: This model is not fitted yet`

**Solution**:
```python
# Always call fit() before predict()
model = RandomForestPredictor()

# ❌ Wrong:
# predictions = model.predict(data)

# ✅ Correct:
model.fit(train_data)
predictions = model.predict(test_data)
```

### Empty Prediction Results

**Symptom**: `predict()` returns empty array

**Solution**:
```python
# Check if input data is valid
print(f"Input shape: {test_data.shape}")
print(f"Input columns: {test_data.columns.tolist()}")

# Ensure data has required features
required_features = model.get_feature_names()
missing = set(required_features) - set(test_data.columns)
if missing:
    print(f"Missing features: {missing}")

# Check for NaN values
print(f"NaN values: {test_data.isnull().sum().sum()}")
```

### Probability Sum != 1

**Symptom**: `predict_proba()` probabilities don't sum to 1

**Note**: This is expected for binary classification. Each row should sum to 1:

```python
probs = model.predict_proba(test_data)
print(f"Shape: {probs.shape}")  # (n_samples, 2)
print(f"Row sums: {probs.sum(axis=1)}")  # Should all be 1.0

# Get probability of positive class:
positive_probs = probs[:, 1]
```

## Common Error Messages

### `ModuleNotFoundError: No module named 'f1_predict'`

**Cause**: Package not installed or not in Python path

**Fix**:
```bash
# In project root:
pip install -e .
# or
uv sync
```

### `ImportError: cannot import name 'X' from 'f1_predict'`

**Cause**: Trying to import something that doesn't exist

**Fix**:
```python
# Check available imports
import f1_predict
print(dir(f1_predict))

# Use correct import path
from f1_predict.models.random_forest import RandomForestPredictor
# not: from f1_predict import RandomForestPredictor
```

### `FileNotFoundError: data/raw/race_results.csv`

**Cause**: Data not collected yet

**Fix**:
```bash
# Collect data first
uv run python -c "
from f1_predict.data.collector import F1DataCollector
collector = F1DataCollector()
collector.collect_all_data()
"
```

### `KeyError: 'podium'`

**Cause**: Target column not in dataframe

**Fix**:
```python
# Create target column during feature engineering
from f1_predict.features.engineering import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.create_basic_features(data)

# Or create manually
data['podium'] = (data['position'] <= 3).astype(int)
```

### `ValueError: could not convert string to float`

**Cause**: Non-numeric data in features

**Fix**:
```python
# Check data types
print(data.dtypes)

# Convert or drop non-numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Or encode categorical variables
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data['driver_encoded'] = encoder.fit_transform(data['driver_id'])
```

## Platform-Specific Issues

### macOS Specific

**Issue**: Apple Silicon (M1/M2) compatibility

**Solution**:
```bash
# Some packages need Rosetta or native ARM builds
# Install Homebrew ARM version
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install OpenMP for XGBoost
brew install libomp

# Use Python 3.10+ for better ARM support
brew install python@3.11
```

### Windows Specific

**Issue**: Long path errors

**Solution**:
```powershell
# Enable long paths in Windows
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

**Issue**: Script execution policy

**Solution**:
```powershell
# Enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Linux Specific

**Issue**: Missing system libraries

**Solution**:
```bash
# Install common dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3-dev \
    libgomp1 \
    git

# For headless servers, may need additional packages
sudo apt-get install -y python3-pip python3-venv
```

## Performance Issues

### Slow Model Training

**Diagnostics**:
```python
import time

start = time.time()
model.fit(train_data)
print(f"Training took {time.time() - start:.1f} seconds")

# Profile with cProfile
import cProfile
cProfile.run('model.fit(train_data)')
```

**Solutions**:
1. Use parallel processing:
```python
model = RandomForestPredictor(n_jobs=-1)  # Use all cores
```

2. Reduce data size:
```python
train_subset = train_data.sample(n=1000)
```

3. Use faster model:
```python
from f1_predict.models.lightgbm_model import LightGBMPredictor
model = LightGBMPredictor()  # Faster than XGBoost
```

### Slow Predictions

**Solution**:
```python
# Batch predictions instead of row-by-row
predictions = model.predict(test_data)  # ✅ Fast

# Not:
predictions = [model.predict(row) for row in test_data.iterrows()]  # ❌ Slow
```

## Getting More Help

### Enable Debug Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Check System Info

```python
import sys
import platform
import pandas as pd
import sklearn
import xgboost
import lightgbm

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Pandas: {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"XGBoost: {xgboost.__version__}")
print(f"LightGBM: {lightgbm.__version__}")
```

### Create Minimal Reproducible Example

When reporting issues, provide:
```python
# Minimal code that reproduces the error
from f1_predict.models.random_forest import RandomForestPredictor
import pandas as pd

data = pd.DataFrame({  # Minimal example data
    'feature1': [1, 2, 3],
    'feature2': [4, 5, 6],
    'target': [0, 1, 0]
})

model = RandomForestPredictor()
model.fit(data)  # Where does it fail?
```

### File a Bug Report

If you can't resolve the issue:

1. Check [existing issues](https://github.com/jpequegn/f1-predict/issues)
2. Create new issue with:
   - **Title**: Clear, specific description
   - **Environment**: OS, Python version, package versions
   - **Steps to reproduce**: Minimal code example
   - **Expected behavior**: What should happen
   - **Actual behavior**: What actually happens
   - **Error message**: Full traceback

## Still Stuck?

- **Documentation**: See [docs/](.)
- **FAQ**: See [faq.md](faq.md)
- **Discussions**: https://github.com/jpequegn/f1-predict/discussions
- **Issues**: https://github.com/jpequegn/f1-predict/issues
