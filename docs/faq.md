# Frequently Asked Questions (FAQ)

## General Questions

### What is F1 Predict?

F1 Predict is a machine learning project that predicts Formula 1 race outcomes using historical data, driver statistics, and race conditions. It provides multiple models for predicting winners, podium finishes, and points-scoring positions.

### What can F1 Predict do?

- **Race Predictions**: Predict race winners, podium finishers, and points scorers
- **Probability Estimates**: Get confidence scores for each prediction
- **Model Comparison**: Compare different ML algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM, Ensemble)
- **Feature Analysis**: Understand which factors influence race outcomes
- **Historical Analysis**: Analyze past race performance and trends

### How accurate are the predictions?

Accuracy varies by model and prediction type:
- **Podium predictions**: 75-87% accuracy (depending on model)
- **Win predictions**: 65-80% accuracy (wins are harder to predict)
- **Points predictions**: 80-90% accuracy (more predictable)

Ensemble models typically achieve the highest accuracy (~87% for podium predictions).

### What data sources are used?

- **Ergast API**: Historical F1 data (2020-2024) including race results, qualifying, and schedules
- **OpenWeatherMap** (optional): Weather conditions for race weekends
- **Manual databases**: Track characteristics, tire data

### Can I use this for real betting?

**⚠️ Important**: This project is for educational and research purposes only.

- Past performance doesn't guarantee future results
- F1 races have inherent unpredictability (crashes, weather, mechanical failures)
- Always gamble responsibly and within your means
- Use predictions as one factor among many in your analysis

### Is this project affiliated with Formula 1?

No. This is an independent, open-source project created by F1 enthusiasts for educational purposes. It uses publicly available data and is not endorsed by Formula 1.

## Technical Questions

### Which model should I use?

It depends on your priorities:

| Priority | Recommended Model | Why |
|----------|-------------------|-----|
| **Accuracy** | Ensemble | Combines multiple models for best results (87%) |
| **Speed** | Rule-Based | Instant predictions, good baseline (70%) |
| **Interpretability** | Logistic Regression | Clear feature weights, explainable (75%) |
| **Balance** | Random Forest | Good accuracy with minimal tuning (82%) |
| **Large Data** | LightGBM | Fast training on big datasets (85%) |

See [Model Comparison](models/README.md) for detailed comparison.

### How do I improve prediction accuracy?

1. **Use more data**: Collect additional seasons of historical data
2. **Better features**: Add weather, tire strategy, track characteristics
3. **Tune hyperparameters**: Use grid search or random search
4. **Use ensemble**: Combine multiple models
5. **Handle class imbalance**: Use SMOTE or class weights for rare events (wins)
6. **Feature engineering**: Create circuit-specific and driver-specific features

### How much data do I need?

**Minimum**:
- 50-100 races for basic models
- Historical data from at least 2 seasons

**Recommended**:
- 200+ races for robust models
- 3-5 seasons of data (2020-2024 included)
- Mix of different circuits and weather conditions

**Optimal**:
- 400+ races
- 5+ seasons
- External data (weather, tire strategy)

### Can I add custom features?

Yes! The `FeatureEngineer` class is extensible:

```python
from f1_predict.features.engineering import FeatureEngineer

class CustomFeatureEngineer(FeatureEngineer):
    def create_custom_features(self, df):
        # Add your features
        df['driver_age'] = 2024 - df['driver_birth_year']
        df['home_race'] = df['driver_nationality'] == df['circuit_country']
        return df

engineer = CustomFeatureEngineer()
features = engineer.create_custom_features(data)
```

See [Feature Engineering Tutorial](tutorials/05-feature-engineering.md).

### How do I update to the latest season?

```bash
# Collect latest data
uv run python -c "
from f1_predict.data.collector import F1DataCollector

collector = F1DataCollector()
results = collector.collect_all_data()
print('Data updated successfully')
"

# Retrain your models with new data
# Models automatically use all available seasons
```

### Can I use this for other racing series?

The architecture is designed for F1 but could be adapted:

**Easy to adapt**:
- Formula E
- IndyCar
- Formula 2/3

**Would require changes**:
- NASCAR (different race format)
- Rally (different scoring, stages)
- MotoGP (different dynamics)

You'd need to:
1. Create new data collectors for your series
2. Adapt feature engineering for series-specific factors
3. Adjust models for different race formats

### What Python version do I need?

**Minimum**: Python 3.9
**Recommended**: Python 3.10 or 3.11
**Maximum tested**: Python 3.12

Python 3.13 may work but hasn't been extensively tested.

### Do I need a GPU?

No. All models train efficiently on CPU:
- Training times are <2 minutes even for ensemble models
- GPU would provide minimal benefit for these model sizes
- Focus on CPU optimization and efficient algorithms instead

## Installation Questions

### How do I install on Windows?

```powershell
# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone and setup
git clone https://github.com/jpequegn/f1-predict.git
cd f1-predict
uv sync --dev

# Run tests to verify
uv run pytest
```

Common Windows issues: See [Troubleshooting Guide](troubleshooting.md#windows-specific-issues).

### How do I install on macOS?

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (may need Homebrew for some packages)
brew install libomp  # Required for XGBoost

# Clone and setup
git clone https://github.com/jpequegn/f1-predict.git
cd f1-predict
uv sync --dev

# Run tests to verify
uv run pytest
```

### How do I install on Linux?

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/jpequegn/f1-predict.git
cd f1-predict
uv sync --dev

# Run tests to verify
uv run pytest
```

### What if uv doesn't work?

Use traditional pip:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## Contributing Questions

### How can I contribute?

We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

**Easy first contributions**:
- Fix typos or improve documentation
- Add test cases
- Report bugs with reproduction steps
- Suggest new features

**More involved**:
- Add new models
- Improve feature engineering
- Optimize performance
- Add new data sources

### What skills are needed?

**For documentation/testing**:
- Basic Python knowledge
- Git/GitHub basics
- Clear writing skills

**For model development**:
- Python programming
- Machine learning fundamentals
- Pandas and scikit-learn experience
- Understanding of F1 racing

**For data engineering**:
- Python programming
- API integration
- Data cleaning and validation
- SQL/database knowledge (helpful)

### How do I report bugs?

1. Check [existing issues](https://github.com/jpequegn/f1-predict/issues)
2. If not found, create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Your environment (OS, Python version)
   - Code example if applicable

### Can I add new models?

Yes! We encourage new model implementations. Guidelines:

1. Follow existing model structure (see `src/f1_predict/models/`)
2. Implement required methods: `fit()`, `predict()`, `predict_proba()`
3. Add comprehensive tests
4. Document mathematical foundations
5. Include usage examples
6. Submit PR with benchmarks

## Usage Questions

### How long does training take?

On a modern laptop (i7, 16GB RAM):
- Logistic Regression: ~5 seconds
- Random Forest: ~30 seconds
- XGBoost: ~45 seconds
- LightGBM: ~30 seconds
- Ensemble: ~2 minutes

### Can I run this in Google Colab?

Yes! F1 Predict works in Colab notebooks:

```python
# In Colab cell:
!git clone https://github.com/jpequegn/f1-predict.git
%cd f1-predict
!pip install -e .

# Use normally
from f1_predict.models.random_forest import RandomForestPredictor
# ...
```

### How do I save/load trained models?

```python
# Save
model = RandomForestRacePredictor(target_type='podium')
model.fit(train_data)
model.save('models/my_model.pkl')

# Load
loaded_model = RandomForestRacePredictor.load('models/my_model.pkl')
predictions = loaded_model.predict(test_data)
```

### Can I use this in a web application?

Yes. The models can be integrated into web apps:

```python
# Flask example
from flask import Flask, request, jsonify
from f1_predict.models.random_forest import RandomForestRacePredictor

app = Flask(__name__)
model = RandomForestRacePredictor.load('models/production_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    predictions = model.predict(data)
    return jsonify({'predictions': predictions.tolist()})
```

See [Web Interface Documentation](../src/f1_predict/web/README.md) for the built-in web UI.

## Error Questions

### "ModuleNotFoundError: No module named 'f1_predict'"

**Solution**: Install the package in development mode:

```bash
cd f1-predict
pip install -e .
# or with uv:
uv sync
```

### "XGBoost/LightGBM won't install"

**macOS**: Install libomp
```bash
brew install libomp
pip install xgboost lightgbm
```

**Windows**: Install Visual C++ Redistributable

**Linux**: Install OpenMP
```bash
sudo apt-get install libgomp1  # Debian/Ubuntu
sudo yum install libgomp  # RedHat/CentOS
```

### "Model not fitted error"

You must call `fit()` before `predict()`:

```python
model = RandomForestRacePredictor()
model.fit(train_data)  # ← Don't forget this!
predictions = model.predict(test_data)
```

### "ValueError: Feature mismatch"

Ensure train and test data have the same features:

```python
# Use same FeatureEngineer for both
engineer = FeatureEngineer()
train_features = engineer.create_basic_features(train_data)
test_features = engineer.create_basic_features(test_data)

# Verify feature count matches
assert train_features.shape[1] == test_features.shape[1]
```

## Still Have Questions?

- **Documentation**: Browse [docs/](.)
- **Tutorials**: See [tutorials/](tutorials/)
- **Issues**: https://github.com/jpequegn/f1-predict/issues
- **Discussions**: https://github.com/jpequegn/f1-predict/discussions
