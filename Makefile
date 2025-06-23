# Makefile for PheTK testing and development
# Usage: make <target>

.PHONY: help test test-unit test-integration test-coverage test-fast lint format install install-dev clean docs

# Default target
help:
	@echo "PheTK Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  test-fast      - Run fast tests only (exclude slow tests)"
	@echo "  lint           - Run code quality checks"
	@echo "  format         - Format code with black"
	@echo "  install        - Install package in development mode"
	@echo "  install-dev    - Install with development dependencies"
	@echo "  clean          - Clean up generated files"
	@echo "  docs           - Generate documentation"
	@echo ""
	@echo "Examples:"
	@echo "  make test"
	@echo "  make test-coverage"
	@echo "  make lint"

# Test targets
test:
	python -m pytest tests/ -v --tb=short

test-unit:
	python -m pytest tests/ -v -m "unit or not integration" --tb=short

test-integration:
	python -m pytest tests/test_integration.py -v --tb=short

test-coverage:
	python -m pytest tests/ --cov=src/PheTK --cov-report=html --cov-report=term-missing --cov-report=xml -v
	@echo "Coverage report generated in htmlcov/index.html"

test-fast:
	python -m pytest tests/ -v -m "not slow" --tb=short

# Code quality targets
lint:
	@echo "Running flake8..."
	python -m flake8 src/PheTK/ --max-line-length=88 --extend-ignore=E203,W503 || true
	@echo "Running black check..."
	python -m black --check --diff src/PheTK/ || true
	@echo "Running mypy..."
	python -m mypy src/PheTK/ --ignore-missing-imports --no-strict-optional || true

format:
	@echo "Formatting code with black..."
	python -m black src/PheTK/
	@echo "Sorting imports with isort..."
	python -m isort src/PheTK/ || true

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e .[dev]

install-test:
	pip install -e .[test]

# Utility targets
clean:
	@echo "Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	rm -f coverage.xml bandit-report.json safety-report.json

docs:
	@echo "Documentation generation not yet implemented"
	@echo "Consider adding sphinx or mkdocs for documentation"

# Security checks
security:
	@echo "Running security checks..."
	python -m bandit -r src/PheTK/ -f json -o bandit-report.json || true
	python -m safety check --json --output safety-report.json || true
	@echo "Security reports generated: bandit-report.json, safety-report.json"

# Development workflow
dev-setup: install-dev
	@echo "Development environment setup complete"
	@echo "Run 'make test' to verify installation"

# CI simulation
ci: lint test-coverage security
	@echo "CI simulation complete"

# Quick development check
check: test-fast lint
	@echo "Quick development check complete"
