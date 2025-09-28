# F1 Predict 🏎️

A machine learning project for predicting Formula 1 race outcomes using historical data, driver statistics, and race conditions.

## 🎯 Project Overview

This project aims to build predictive models for Formula 1 racing, including:

- **Race Winner Prediction**: Predict which driver will win a given race
- **Podium Predictions**: Forecast top 3 finishers
- **Championship Predictions**: Season-long championship standings
- **Performance Analysis**: Driver and team performance metrics

## 🏗️ Project Structure

```
f1-predict/
├── src/
│   └── f1_predict/          # Main package
│       └── __init__.py
├── data/
│   ├── raw/                 # Raw data files
│   ├── processed/           # Cleaned data
│   ├── models/              # Trained models
│   └── external/            # External data sources
├── tests/                   # Unit and integration tests
├── docs/                    # Documentation
├── config/                  # Configuration files
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/) - Ultra-fast Python package installer (recommended)
- Or pip as fallback

### Installation

#### Option 1: Using uv (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/f1-predict.git
cd f1-predict
```

2. Install uv (if not already installed):
```bash
# On macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip:
pip install uv
```

3. Install dependencies:
```bash
# Install all dependencies including development tools
uv sync --dev --all-extras

# For production only:
uv sync
```

#### Option 2: Using pip (Traditional)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/f1-predict.git
cd f1-predict
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

### Development Setup

1. Set up pre-commit hooks:
```bash
uv run pre-commit install
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Run tests to verify setup:
```bash
uv run pytest
```

## 📊 Data Sources

The project utilizes various data sources including:

- **Race Results**: Historical race outcomes and lap times
- **Driver Data**: Driver statistics, experience, and performance metrics
- **Team Data**: Constructor championships, car performance
- **Track Information**: Circuit characteristics, weather conditions
- **Seasonal Data**: Championship standings, points systems

## 🛠️ Usage

### Basic Example

```python
from f1_predict import RacePredictor

# Initialize predictor
predictor = RacePredictor()

# Load and prepare data
predictor.load_data('data/processed/race_data.csv')

# Train model
predictor.train()

# Make predictions
predictions = predictor.predict_race('2024-monaco-gp')
print(predictions)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests with coverage
uv run pytest

# Run with detailed coverage report
uv run pytest --cov-report=html

# Run specific test file
uv run pytest tests/test_predictor.py

# Run tests in parallel (faster)
uv run pytest -n auto

# Run only fast tests (skip slow integration tests)
uv run pytest -m "not slow"
```

## 📈 Model Performance

Current model metrics (to be updated as models are developed):

- **Accuracy**: TBD
- **Precision**: TBD
- **Recall**: TBD
- **F1-Score**: TBD

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Run the test suite to ensure everything passes
6. Commit your changes (`git commit -m 'Add some amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

This project uses modern Python tooling:
- **Ruff** for fast linting and formatting (replaces black, flake8, isort)
- **MyPy** for type checking
- **Bandit** for security linting
- **Pre-commit hooks** for automated quality checks

Run code quality checks:
```bash
# Format code
uv run ruff format .

# Lint code (with auto-fix)
uv run ruff check . --fix

# Type checking
uv run mypy src/

# Security scanning
uv run bandit -r src/

# Run all checks
uv run pre-commit run --all-files
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Formula 1 for providing exciting racing data
- The open-source community for excellent ML libraries
- Contributors and maintainers of this project

## 📞 Contact

- **Author**: Julien Pequegnot
- **Email**: [Your Email]
- **GitHub**: [@jpequegn](https://github.com/jpequegn)

## 🗺️ Roadmap

- [ ] Data collection and preprocessing pipeline
- [ ] Exploratory data analysis
- [ ] Feature engineering
- [ ] Baseline model implementation
- [ ] Advanced ML models (XGBoost, LightGBM)
- [ ] Model evaluation and validation
- [ ] Web API for predictions
- [ ] Dashboard for visualization
- [ ] Real-time prediction updates