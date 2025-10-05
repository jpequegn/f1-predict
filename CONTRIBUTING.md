# Contributing to F1 Predict

Thank you for your interest in contributing to F1 Predict! This document provides guidelines and instructions for setting up your development environment and contributing to the project.

## 🛠️ Development Environment Setup

### Prerequisites

- Python 3.9 or higher
- Git
- [uv](https://docs.astral.sh/uv/) - Ultra-fast Python package installer (recommended)
- Or pip as fallback

### Step 1: Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/f1-predict.git
cd f1-predict
```

### Step 2: Virtual Environment Setup

We strongly recommend using a virtual environment to isolate project dependencies:

#### Option 1: Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Verify activation (should show path to venv)
which python  # macOS/Linux
where python   # Windows
```

#### Option 2: Using conda

```bash
# Create conda environment
conda create -n f1-predict python=3.11

# Activate environment
conda activate f1-predict
```

#### Option 3: Using pyenv + virtualenv

```bash
# Install specific Python version (if needed)
pyenv install 3.11.5

# Create virtual environment
pyenv virtualenv 3.11.5 f1-predict

# Set local environment for project
pyenv local f1-predict
```

### Step 3: Install Dependencies

#### Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# Or: pip install uv

# Install all dependencies including development tools
uv sync --dev --all-extras
```

#### Using pip (Traditional)

```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"
```

### Step 4: Verify Installation

```bash
# Test imports
python -c "import f1_predict; print('Installation successful!')"

# Run tests to ensure everything works
pytest
```

## 🔧 Development Workflow

### Code Quality Tools

This project uses modern Python tooling for code quality:

```bash
# Format code with Ruff (replaces Black)
uv run ruff format .

# Lint and fix issues with Ruff (replaces flake8 + isort)
uv run ruff check . --fix

# Type check with MyPy
uv run mypy src/

# Security scanning with Bandit
uv run bandit -r src/

# Run all quality checks
uv run pre-commit run --all-files

# Or use Make commands
make all-checks  # if Makefile is available
```

### Pre-commit Hooks (Recommended)

Set up pre-commit hooks to automatically run quality checks:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run hooks manually
uv run pre-commit run --all-files
```

### Running Tests

```bash
# Run all tests with coverage
uv run pytest

# Run with detailed coverage report
uv run pytest --cov-report=html

# Run specific test file
uv run pytest tests/test_predictor.py

# Run tests in parallel for speed
uv run pytest -n auto

# Run only fast tests (skip slow integration tests)
uv run pytest -m "not slow"

# Run tests with verbose output
uv run pytest -v
```

## 🔄 Virtual Environment Best Practices

### Using uv (Automatic Environment Management)

uv automatically manages virtual environments for you:

```bash
# uv automatically creates and uses project-specific virtual environments
uv run python --version    # Runs in project environment
uv run pytest             # Runs tests in project environment
uv sync                    # Syncs dependencies in project environment
```

### Manual Virtual Environment (Traditional)

If using pip, activate your virtual environment before working:

```bash
# Activate
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Work on project...

# Deactivate when done
deactivate
```

### Managing Dependencies

#### Using uv (Recommended)

```bash
# Add a production dependency
uv add new-package

# Add a development dependency
uv add --dev new-package

# Add an optional dependency
uv add --optional mlops new-package

# Update all dependencies
uv sync --upgrade

# Remove a dependency
uv remove new-package
```

#### Using pip (Traditional)

When adding new dependencies manually:

1. Add to `pyproject.toml` under `[project.dependencies]` or `[project.optional-dependencies]`
2. Install the updated dependencies:
```bash
pip install -e ".[dev]"
```

### Environment Verification

#### Using uv

```bash
# Check Python version
uv run python --version

# Check installed packages
uv pip list

# Verify project dependencies
uv tree
```

#### Using traditional virtual environment

```bash
# Check Python path
which python

# Check installed packages
pip list

# Check virtual environment
echo $VIRTUAL_ENV  # Should show path to your venv
```

## 🔀 Making Changes

### Step-by-Step Development Workflow

#### 1. Create Feature Branch

```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
# OR for bug fixes
git checkout -b fix/bug-description
```

### Branch Naming Convention

Use descriptive branch names:
- `feature/add-new-model` - for new features
- `fix/prediction-bug` - for bug fixes
- `docs/update-readme` - for documentation
- `refactor/data-pipeline` - for refactoring
- `test/add-integration-tests` - for test improvements
- `chore/update-dependencies` - for maintenance

#### 2. Make Changes

Follow these guidelines while coding:

**Code Style**:
- Line length: 88 characters (Black default)
- Imports: Sorted with ruff
- Docstrings: Google style (required for all public functions)
- Type hints: Required for all public functions and methods
- Naming: `snake_case` for functions/variables, `PascalCase` for classes

**Example**:
```python
from pathlib import Path
from typing import Protocol

import pandas as pd
from pydantic import BaseModel


class PredictorProtocol(Protocol):
    """Interface for prediction models."""

    def fit(self, data: pd.DataFrame) -> "PredictorProtocol":
        """Train the model on data."""
        ...


def calculate_driver_stats(
    driver_id: str,
    race_data: pd.DataFrame,
    output_path: Path | None = None
) -> dict[str, float]:
    """Calculate performance statistics for a driver.

    Args:
        driver_id: Unique driver identifier (e.g., 'verstappen')
        race_data: Historical race results DataFrame
        output_path: Optional path to save statistics

    Returns:
        Dictionary with calculated metrics:
            - win_rate: Proportion of wins
            - podium_rate: Proportion of podium finishes
            - avg_position: Average finishing position

    Raises:
        ValueError: If driver_id not found in race_data
        ValueError: If race_data is empty

    Example:
        >>> data = pd.DataFrame({'driver_id': ['verstappen'], 'position': [1]})
        >>> stats = calculate_driver_stats('verstappen', data)
        >>> stats['win_rate']
        1.0
    """
    if driver_id not in race_data["driver_id"].values:
        raise ValueError(f"Driver {driver_id} not found")

    if race_data.empty:
        raise ValueError("race_data cannot be empty")

    # Implementation...
    return {"win_rate": 0.0, "podium_rate": 0.0, "avg_position": 0.0}
```

#### 3. Run Quality Checks

```bash
# Format code
uv run ruff format .

# Lint with auto-fix
uv run ruff check . --fix

# Type checking
uv run mypy src/

# Run tests
uv run pytest

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

#### 4. Add Tests

```python
# tests/test_statistics.py
import pytest
import pandas as pd
from f1_predict.statistics import calculate_driver_stats


class TestDriverStatistics:
    """Tests for driver statistics calculations."""

    @pytest.fixture
    def sample_race_data(self):
        """Create sample race data for testing."""
        return pd.DataFrame({
            'driver_id': ['verstappen', 'verstappen', 'hamilton'],
            'position': [1, 2, 1],
            'points': [25, 18, 25]
        })

    def test_calculate_driver_stats_valid_data(self, sample_race_data):
        """Test statistics calculation with valid data."""
        stats = calculate_driver_stats('verstappen', sample_race_data)

        assert stats['win_rate'] == 0.5
        assert stats['podium_rate'] == 1.0
        assert 'avg_position' in stats

    def test_calculate_driver_stats_driver_not_found(self, sample_race_data):
        """Test error handling for unknown driver."""
        with pytest.raises(ValueError, match="not found"):
            calculate_driver_stats('unknown_driver', sample_race_data)

    def test_calculate_driver_stats_empty_data(self):
        """Test error handling for empty DataFrame."""
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_driver_stats('verstappen', empty_data)
```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

**Example**:
```
feat(models): add LightGBM predictor with early stopping

Implement LightGBMRacePredictor with:
- Categorical feature support
- Early stopping rounds for better generalization
- Configurable learning rate and max depth

The model achieves 85% accuracy on 2024 test data, matching
XGBoost performance with 2x faster training time.

Closes #42
```

#### 5. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat(models): add LightGBM predictor"

# Or use interactive commit for partial staging
git add -p
```

### Pull Request Process

#### 1. Push Branch

```bash
git push origin feature/your-feature-name
```

#### 2. Create Pull Request

Use GitHub CLI or web interface:

```bash
# Using GitHub CLI
gh pr create --title "feat: Add LightGBM predictor" --body "Description of changes"
```

**PR Title Format**: Same as commit message format

**PR Description Template**:
```markdown
## Summary
Brief description of what this PR does.

## Changes Made
- Added LightGBM predictor implementation
- Updated model evaluation to include LightGBM
- Added comprehensive tests with 95% coverage

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Documentation
- [ ] Code docstrings added/updated
- [ ] README updated if needed
- [ ] Architecture docs updated if needed

## Screenshots
If applicable, add screenshots or GIFs

## Related Issues
Closes #42
Relates to #15
```

#### 3. Code Review Process

**For PR Authors**:
- Respond to review comments promptly
- Make requested changes in new commits (don't force push)
- Mark conversations as resolved when addressed
- Request re-review after changes

**For Reviewers**:
- Be constructive and specific
- Suggest improvements with examples
- Approve when satisfied
- Use "Request Changes" for blocking issues

#### 4. Merge Requirements

PRs can be merged when:
- ✅ All checks pass (tests, linting, type checking)
- ✅ At least one approval from maintainer
- ✅ No merge conflicts
- ✅ All conversations resolved
- ✅ Documentation updated

## 📝 Code Documentation Standards

### Docstring Format

Use Google-style docstrings for all public functions, classes, and modules:

```python
def predict_race_outcome(
    circuit_id: str,
    drivers: list[str],
    weather: WeatherCondition = WeatherCondition.CLEAR,
    model_type: str = "ensemble"
) -> pd.DataFrame:
    """Predict race outcome for given parameters.

    This function uses the specified model to predict finishing positions
    for all drivers at a given circuit under specified weather conditions.

    Args:
        circuit_id: Circuit identifier (e.g., 'monaco', 'silverstone')
        drivers: List of driver IDs to include in prediction
        weather: Expected weather conditions (default: CLEAR)
        model_type: Model to use ('ensemble', 'random_forest', 'xgboost')

    Returns:
        DataFrame with columns:
            - driver_id: Driver identifier
            - predicted_position: Predicted finishing position (1-20)
            - probability: Confidence score (0-1)

    Raises:
        ValueError: If circuit_id is unknown
        ValueError: If model_type is invalid
        RuntimeError: If model not trained

    Example:
        >>> result = predict_race_outcome(
        ...     circuit_id='monaco',
        ...     drivers=['verstappen', 'leclerc'],
        ...     weather=WeatherCondition.RAIN
        ... )
        >>> result.head()
           driver_id  predicted_position  probability
        0  verstappen                   1         0.87
        1     leclerc                   2         0.72

    Note:
        Weather has significant impact on predictions. Rain conditions
        can dramatically change expected outcomes.

    See Also:
        - predict_podium_probability(): For podium-only predictions
        - predict_championship(): For championship predictions
    """
    # Implementation
```

### Type Hints

All public functions must have complete type hints:

```python
from typing import Protocol, TypeAlias
from pathlib import Path

# Type aliases for clarity
ModelPath: TypeAlias = str | Path
SeasonYear: TypeAlias = int

class ModelProtocol(Protocol):
    """Protocol defining model interface."""

    def fit(self, data: pd.DataFrame) -> "ModelProtocol":
        """Train model."""
        ...

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate predictions."""
        ...
```

## 🧪 Testing Guidelines

- Write tests for all new functionality
- Maintain or improve test coverage
- Use descriptive test names
- Include edge cases and error conditions

Example test structure:
```python
def test_race_predictor_with_valid_data():
    """Test that RacePredictor works with valid race data."""
    # Arrange
    predictor = RacePredictor()
    race_data = load_test_data()

    # Act
    result = predictor.predict(race_data)

    # Assert
    assert result is not None
    assert len(result) > 0
```

## 📁 Project Structure Guidelines

When adding new modules:

- Place source code in `src/f1_predict/`
- Add corresponding tests in `tests/`
- Update documentation as needed
- Follow existing naming conventions

## 🐛 Reporting Issues

When reporting bugs:

1. Use the issue template
2. Include Python version and OS
3. Provide minimal reproduction example
4. Include relevant error messages

## 📞 Getting Help

- Open an issue for bugs or feature requests
- Start discussions for questions or ideas
- Check existing issues and documentation first

## 🎉 Recognition

Contributors will be recognized in:
- README.md acknowledgments
- Release notes
- GitHub contributors page

Thank you for contributing to F1 Predict! 🏎️