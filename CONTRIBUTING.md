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

### Branch Naming Convention

Use descriptive branch names:
- `feature/add-new-model` - for new features
- `fix/prediction-bug` - for bug fixes
- `docs/update-readme` - for documentation
- `refactor/data-pipeline` - for refactoring

### Commit Message Format

Follow conventional commits:
```
type(scope): description

[optional body]

[optional footer]
```

Examples:
- `feat(models): add XGBoost predictor`
- `fix(data): handle missing race data`
- `docs(readme): update installation instructions`

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

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