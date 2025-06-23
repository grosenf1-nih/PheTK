"""
Conftest file for PheTK tests.
Contains fixtures and common test utilities.
"""
import pytest
import polars as pl
import pandas as pd
import os
import tempfile
from unittest.mock import Mock, patch
from typing import Dict, Any


@pytest.fixture
def sample_icd_data():
    """Fixture providing sample ICD data for testing."""
    return pl.DataFrame({
        "person_id": [1, 2, 3, 4, 5, 1, 2],
        "ICD": ["250.00", "401.9", "V70.0", "250.01", "E11.9", "250.00", "401.1"],
        "vocabulary_id": ["ICD9CM", "ICD9CM", "ICD9CM", "ICD9CM", "ICD10CM", "ICD9CM", "ICD9CM"],
        "flag": [9, 9, 9, 9, 10, 9, 9]
    })


@pytest.fixture
def sample_phecode_data():
    """Fixture providing sample phecode count data."""
    return pl.DataFrame({
        "person_id": [1, 2, 3, 4, 5],
        "phecode": ["250", "401", "250", "250", "401"],
        "count": [2, 1, 3, 1, 2]
    })


@pytest.fixture
def sample_cohort_data():
    """Fixture providing sample cohort data with covariates."""
    return pl.DataFrame({
        "person_id": [1, 2, 3, 4, 5],
        "age": [65, 45, 78, 34, 56],
        "sex_at_birth": [1, 0, 1, 0, 1],
        "genotype": ["AA", "AB", "BB", "AA", "AB"],
        "pc1": [0.1, -0.2, 0.3, -0.1, 0.4],
        "pc2": [0.05, 0.15, -0.25, 0.35, -0.05]
    })


@pytest.fixture
def sample_phewas_result():
    """Fixture providing sample PheWAS results."""
    return pl.DataFrame({
        "phecode": ["250", "401", "585", "272.1"],
        "phecode_description": ["Diabetes mellitus", "Hypertension", "Chronic kidney disease", "Hyperlipidemia"],
        "phecode_category": ["endocrine/metabolic", "circulatory system", "genitourinary", "endocrine/metabolic"],
        "case_count": [120, 200, 80, 150],
        "control_count": [1000, 900, 1100, 950],
        "odds_ratio": [2.5, 1.8, 3.2, 1.6],
        "p_value": [0.001, 0.01, 0.0001, 0.05],
        "neg_log_p_value": [3.0, 2.0, 4.0, 1.3],
        "converged": [True, True, True, True],
        "beta": [0.916, 0.588, 1.163, 0.470]
    })


@pytest.fixture
def temp_csv_file():
    """Fixture providing a temporary CSV file."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    yield temp_file.name
    os.unlink(temp_file.name)


@pytest.fixture
def mock_bigquery_client():
    """Mock BigQuery client for testing."""
    with patch('google.cloud.bigquery.Client') as mock_client:
        mock_job = Mock()
        mock_result = Mock()
        mock_result.to_arrow.return_value = None
        mock_job.result.return_value = mock_result
        mock_client.return_value.query.return_value = mock_job
        yield mock_client


@pytest.fixture
def mock_os_environ():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'WORKSPACE_CDR': 'test_cdr',
        'GOOGLE_PROJECT': 'test_project'
    }):
        yield


@pytest.fixture
def sample_phecode_mapping():
    """Fixture providing sample phecode mapping data."""
    return pl.DataFrame({
        "ICD": ["250.00", "250.01", "401.9", "401.1", "E11.9"],
        "flag": [9, 9, 9, 9, 10],
        "phecode": ["250", "250", "401", "401", "250.2"],
        "phecode_description": ["Diabetes mellitus", "Diabetes mellitus", "Hypertension", "Hypertension", "Type 2 diabetes"],
        "phecode_category": ["endocrine/metabolic", "endocrine/metabolic", "circulatory system", "circulatory system", "endocrine/metabolic"],
        "phecode_exclude_range": ["", "", "", "", ""]
    })


class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def create_test_csv(data: pl.DataFrame, file_path: str) -> None:
        """Create a test CSV file from polars DataFrame."""
        data.write_csv(file_path)
    
    @staticmethod
    def create_large_dataset(n_rows: int = 1000) -> pl.DataFrame:
        """Create a large dataset for performance testing."""
        import random
        return pl.DataFrame({
            "person_id": list(range(1, n_rows + 1)),
            "phecode": [f"{random.randint(100, 999)}" for _ in range(n_rows)],
            "count": [random.randint(1, 10) for _ in range(n_rows)]
        })
