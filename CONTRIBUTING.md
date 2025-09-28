# Contributing to F1 Predict

Thank you for your interest in contributing to F1 Predict! This document provides guidelines and instructions for setting up your development environment and contributing to the project.

## 🛠️ Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip (Python package installer)

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

```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Or install requirements separately
pip install -r requirements-dev.txt
pip install -e .
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

This project uses several tools to maintain code quality:

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type check with mypy
mypy src/

# Run all quality checks
make lint  # if Makefile is available
```

### Pre-commit Hooks (Recommended)

Set up pre-commit hooks to automatically run quality checks:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/f1_predict

# Run specific test file
pytest tests/test_predictor.py

# Run tests with verbose output
pytest -v
```

## 🔄 Virtual Environment Best Practices

### Activating and Deactivating

Always activate your virtual environment before working on the project:

```bash
# Activate
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Work on project...

# Deactivate when done
deactivate
```

### Managing Dependencies

When adding new dependencies:

1. Install the package in your activated virtual environment:
```bash
pip install new-package
```

2. Update requirements files:
```bash
# For production dependencies
echo "new-package>=version" >> requirements.txt

# For development dependencies
echo "new-package>=version" >> requirements-dev.txt
```

3. Generate exact versions (optional):
```bash
pip freeze > requirements-lock.txt
```

### Environment Verification

Ensure you're in the correct environment:

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