# F1 Predict Development Makefile

.PHONY: help install install-dev clean test lint format check-format type-check all-checks setup-env

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  setup-env    Set up development environment"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run all linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  check-format Check if code is properly formatted"
	@echo "  type-check   Run mypy type checking"
	@echo "  all-checks   Run all quality checks"
	@echo "  clean        Clean up temporary files"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

setup-env:
	@echo "Setting up development environment..."
	@echo "Creating virtual environment..."
	python -m venv venv
	@echo "Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
	@echo "Then run: make install-dev"

# Testing
test:
	pytest

test-cov:
	pytest --cov=src/f1_predict --cov-report=html --cov-report=term

# Code quality
format:
	black src/ tests/
	isort src/ tests/

check-format:
	black --check src/ tests/
	isort --check-only src/ tests/

lint:
	flake8 src/ tests/

type-check:
	mypy src/

all-checks: check-format lint type-check test

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/