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

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/f1-predict.git
cd f1-predict
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. Install dependencies:
```bash
# Install production dependencies
pip install -r requirements.txt

# For development (includes testing and linting tools)
pip install -r requirements-dev.txt
```

### Development Setup

1. Install the package in development mode:
```bash
pip install -e .
```

2. Set up pre-commit hooks (optional but recommended):
```bash
pre-commit install
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
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/f1_predict --cov-report=html

# Run specific test file
pytest tests/test_predictor.py
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

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run code quality checks:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
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