# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
# Run all tests with coverage (primary command)
uv run pytest

# Run specific test file
uv run pytest tests/data/test_collector.py

# Run with proper Python path for imports
PYTHONPATH=src python -m pytest tests/data/test_collector.py -v

# Run tests in parallel for faster execution
uv run pytest -n auto

# Skip slow tests
uv run pytest -m "not slow"

# Generate HTML coverage report
uv run pytest --cov-report=html

# Run tests excluding PyTorch-based models (avoids segfaults)
uv run pytest --ignore=tests/models/test_lstm_model.py --ignore=tests/models/test_arima_model.py -q

# Run tests for specific domains
PYTHONPATH=src uv run pytest tests/data/ -v  # Data collection tests
PYTHONPATH=src uv run pytest tests/api/ -v   # API client tests
PYTHONPATH=src uv run pytest tests/models/ --ignore=tests/models/test_lstm_model.py -v  # Model tests
PYTHONPATH=src uv run pytest tests/web/ -v   # Web interface tests
PYTHONPATH=src uv run pytest tests/features/ -v  # Feature engineering tests
```

## Testing Standards & Guidelines (Issue #17)

### Test Organization
Tests are organized by module domain with corresponding source code structure:
- `tests/api/` - API client tests (base, ergast, realtime, weather)
- `tests/data/` - Data collection & processing tests (cleaning, collector, anomaly, weather, tire, track)
- `tests/features/` - Feature engineering tests
- `tests/models/` - Model tests (baseline, ensemble, random forest, xgboost, lightgbm, arima, anomaly)
- `tests/web/` - Web interface tests (unit, integration, components)
- `tests/analysis/` - Analysis & explainability tests (SHAP, race preview)
- `tests/metrics/` - Performance metrics tests
- `conftest.py` - Shared fixtures and pytest configuration

### Test Framework Setup

**Logging Configuration**: Tests use a special structlog configuration (in `conftest.py`) to avoid issues with stdlib logger processors that expect 'name' attributes on custom logger objects.

**Fixtures**: Common fixtures are defined in `tests/conftest.py` and reused across test modules:
- `mock_http_response` - HTTP success response mock
- `mock_api_error` - HTTP error response mock
- `sample_features` - Feature DataFrame for model testing
- `sample_race_results` - Race result DataFrame
- `temp_data_dir` / `temp_model_dir` - Temporary directories
- `mock_session_state` - Streamlit session state mock
- `mock_streamlit` - Streamlit component mocks

### Test Coverage by Module

| Module | Test Coverage | Notes |
|--------|---------------|-------|
| API Clients | ✅ Comprehensive | base, ergast tested; realtime pending |
| Data Collection | ✅ Comprehensive | cleaning, collector, anomaly hooks tested |
| Weather Data | ✅ 46 tests | New (Issue #17) - covers all major functions |
| Tire/Track Data | ✅ 25 tests | New (Issue #17) - domain-specific patterns |
| Feature Engineering | ✅ Comprehensive | Driver form, team reliability, circuit performance |
| Models | ✅ Mostly complete | Exclude LSTM/ARIMA (PyTorch compatibility issues) |
| Web Interface | ✅ Growing | Prediction page, analytics, theme tested |
| Analysis | ⚠️ Partial | SHAP, explainer pending full coverage |

### Running Tests Successfully

**Preferred approach** (avoids segfaults from PyTorch):
```bash
PYTHONPATH=src uv run pytest tests/data tests/api tests/features tests/web -v
```

**Full suite** (requires stable PyTorch environment):
```bash
PYTHONPATH=src uv run pytest
```

**Quick smoke test**:
```bash
PYTHONPATH=src uv run pytest tests/data/test_cleaning.py tests/api/test_base.py -v
```

### Known Issues & Workarounds

1. **PyTorch Segmentation Faults**
   - Issue: LSTM/ARIMA tests cause segfaults in certain environments
   - Workaround: Exclude with `--ignore=tests/models/test_lstm_model.py`
   - Status: Awaiting investigation of torch/multiprocessing compatibility

2. **Structlog Test Configuration**
   - Issue: stdlib logger processors expect 'name' attribute
   - Solution: Custom test configuration in `conftest.py`
   - Status: ✅ Fixed in Issue #17

### Test Markers
Tests use pytest markers for selective execution:
```bash
uv run pytest -m "unit"           # Unit tests only
uv run pytest -m "integration"    # Integration tests only
uv run pytest -m "slow"           # Slow tests only
uv run pytest -m "not slow"       # Skip slow tests
uv run pytest -m "ml"             # ML model tests
uv run pytest -m "data"           # Data processing tests
uv run pytest -m "api"            # API tests
```

### Writing New Tests

**Pattern for unit tests**:
```python
import pytest
from unittest.mock import Mock, patch

class TestMyModule:
    """Test description."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return {"key": "value"}

    def test_basic_functionality(self, sample_data):
        """Test basic behavior."""
        assert sample_data["key"] == "value"

    @patch("module.function")
    def test_with_mock(self, mock_func, sample_data):
        """Test with mocked dependencies."""
        mock_func.return_value = "mocked"
        # Test logic here
```

**Key Guidelines**:
1. Use descriptive test names: `test_<function>_<scenario>`
2. One assertion per test when possible
3. Mock external dependencies (APIs, files, etc.)
4. Use fixtures for reusable test data
5. Add docstrings explaining what is tested
6. Keep tests fast and independent

### Coverage Goals

**Target**: ≥80% overall coverage with:
- ≥85% for critical modules (API, data collection)
- ≥75% for secondary modules (features, models)
- ≥60% for UI/presentation code

**Note**: Tests that don't import source modules will show 0% coverage (this is expected and OK).

### Code Quality
```bash
# Format code (replaces black/isort)
uv run ruff format .

# Lint with auto-fix
uv run ruff check . --fix

# Type checking
uv run mypy src/

# Security scanning
uv run bandit -r src/

# Run all pre-commit checks
uv run pre-commit run --all-files
```

### Data Collection
```bash
# Collect historical F1 data (2020-2024)
python scripts/collect_historical_data.py

# Force refresh all data
python scripts/collect_historical_data.py --refresh

# Collect specific data types
python scripts/collect_historical_data.py --race-results
python scripts/collect_historical_data.py --qualifying

# View data summary
python scripts/collect_historical_data.py --summary
```

### Data Cleaning Pipeline
```bash
# Clean all data types
f1-predict clean --type all --data-dir data --output-dir data/processed

# Clean specific data types
f1-predict clean --type race-results --data-dir data --output-dir data/processed
f1-predict clean --type qualifying --data-dir data --output-dir data/processed
f1-predict clean --type schedules --data-dir data --output-dir data/processed

# Clean with strict quality validation (fails if quality thresholds not met)
f1-predict clean --type all --data-dir data --output-dir data/processed --strict

# Validate data quality without cleaning
f1-predict validate --type all --data-dir data --output-dir data/reports

# Complete pipeline: collect and clean in one command
f1-predict collect --type all --data-dir data
f1-predict clean --type all --data-dir data --output-dir data/processed
```

## Architecture Overview

### Core Package Structure (`src/f1_predict/`)

**API Layer** (`api/`):
- `base.py`: BaseAPIClient with rate limiting (4 req/sec), retry logic, and Pydantic model parsing
- `ergast.py`: ErgastAPIClient extending BaseAPIClient for F1 data from ergast.com API
- Rate limiting built-in to prevent API abuse, configurable timeouts and retries

**Data Layer** (`data/`):
- `models.py`: Pydantic models for all F1 entities (Race, Driver, Constructor, Circuit, etc.)
- `collector.py`: F1DataCollector class for historical data collection (2020-2024)
- `cleaning.py`: DataCleaner and DataQualityValidator for data cleaning pipeline
- Uses aliases for API field mapping (e.g., `circuitId` -> `circuit_id`)

**Infrastructure**:
- `logging_config.py`: Structured logging with structlog (JSON/console formats)
- Context managers for resource cleanup throughout

### Data Flow Architecture

1. **API Client Layer**: Base HTTP client → Ergast-specific client → Rate-limited requests
2. **Model Layer**: Raw JSON → Pydantic validation → Type-safe Python objects
3. **Collection Layer**: Orchestrated API calls → Data aggregation → CSV/JSON storage
4. **Cleaning Layer**: Raw data → Data cleaning pipeline → Quality validation → Cleaned data
5. **Storage**: Raw data (`data/raw/`) and processed data (`data/processed/`)

### Data Cleaning Pipeline Components

**DataCleaner Class**:
- Handles missing or inconsistent data with configurable default values
- Standardizes driver/team/circuit names across seasons using mapping dictionaries
- Converts data types (strings to numbers, date formats, time formats)
- Validates data against business rules (position ranges, coordinate bounds, etc.)
- Generates quality reports with scoring and detailed issue tracking

**Data Quality Metrics**:
- **Missing Data**: Tracks missing values per field with percentage calculations
- **Type Issues**: Identifies and logs data type conversion failures
- **Validation Errors**: Captures business rule violations and data inconsistencies
- **Standardization**: Records name changes for consistency across seasons
- **Quality Score**: 0-100% score based on data completeness and accuracy

**Quality Thresholds** (configurable):
- Missing data: ≤5% of records
- Invalid data: ≤2% of records
- Minimum quality score: ≥85%

**DataQualityValidator Class**:
- Validates entire datasets for consistency and completeness
- Checks for duplicate records within races/qualifying sessions
- Validates position consistency (no duplicate positions per race)
- Generates comprehensive quality reports with recommendations

### Key Design Patterns

- **Composition over inheritance**: BaseAPIClient extended by ErgastAPIClient
- **Type safety**: Full Pydantic model coverage with field aliases
- **Error handling**: Graceful degradation with comprehensive logging
- **Resource management**: Context managers for API clients and data collectors
- **Rate limiting**: Built into base client to respect API constraints

## Testing Strategy

- **Unit tests**: Mock all external API calls using `unittest.mock`
- **Pytest fixtures**: Reusable sample data for consistent testing
- **Coverage requirement**: 80% minimum, configured in pyproject.toml
- **Test markers**: `slow`, `integration`, `ml`, `data`, `api` for selective test runs
- **Test isolation**: Each test uses temporary directories and mocked dependencies

## Configuration Notes

- **Package management**: Uses `uv` (preferred) with fallback to pip
- **Python version**: 3.9+ required, tested up to 3.12
- **Dependencies**: Scientific stack (pandas, numpy, scikit-learn) + F1-specific tools
- **Code style**: Ruff for linting/formatting (replaces black, flake8, isort)
- **Type checking**: MyPy with strict mode enabled

## Data Collection System

The F1DataCollector is the primary interface for historical data:

```python
from f1_predict.data.collector import F1DataCollector

collector = F1DataCollector(data_dir="data")
results = collector.collect_all_data()  # Returns status dict
race_file = collector.collect_race_results()  # Returns CSV path
```

**Seasons covered**: 2020-2024 (configurable in collector.seasons)
**Data types**: Race results, qualifying results, race schedules
**Storage formats**: Both CSV (analysis) and JSON (programmatic access)
**Error handling**: Continues collection on individual race failures

### Integrated Collection and Cleaning

```python
from f1_predict.data.collector import F1DataCollector

collector = F1DataCollector(data_dir="data")

# Collect and clean in one operation
results = collector.collect_and_clean_all_data(
    force_refresh=False,
    enable_cleaning=True
)

# Or clean previously collected data
cleaning_results = collector.clean_collected_data()
```

**Pipeline Benefits**:
- Automatic data quality validation after collection
- Standardized output formats (JSON + CSV)
- Quality reports with actionable insights
- Configurable quality thresholds with pass/fail validation
- Integration with CLI for scripted workflows

## Development Workflow

1. **Environment setup**: `uv sync --dev` or traditional pip with venv
2. **Pre-commit hooks**: Auto-installed to enforce code quality
3. **Testing**: Always run tests before commits (`uv run pytest`)
4. **Type safety**: All new code should include type hints
5. **API changes**: Update Pydantic models when adding new data sources
6. **Data collection**: Use CLI script for data operations, not direct API calls

## Common Patterns

- Import paths use `PYTHONPATH=src` for test execution
- API clients always use context managers: `with client: ...`
- Pydantic models use aliases for API compatibility
- Error logging includes context: season, round, operation type
- File operations use pathlib.Path consistently
- Rate limiting is automatic, no manual delays needed
- Data cleaning always includes quality reporting and validation
- CLI commands provide verbose logging and error handling
- Integration tests use fixture data for realistic validation