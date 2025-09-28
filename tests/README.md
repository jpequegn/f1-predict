# Tests

This directory contains all test files for the F1 prediction project.

## Structure

- Unit tests should be placed in files named `test_*.py`
- Integration tests should be placed in an `integration/` subdirectory
- Test data and fixtures should be placed in a `fixtures/` subdirectory

## Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src/f1_predict
```