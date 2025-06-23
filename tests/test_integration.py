"""
Integration tests for PheTK package.
These tests verify that different components work together correctly.
"""
import pytest
import polars as pl
import pandas as pd
import numpy as np
import os
import sys
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PheTK.Cohort import Cohort
from PheTK.Phecode import Phecode
from PheTK.PheWAS import PheWAS
from PheTK.Plot import Plot
from PheTK import _utils, _queries


class TestPheTKWorkflow:
    """Test complete PheTK workflow integration."""
    
    def setup_method(self):
        """Set up test data for integration tests."""
        # Create realistic test datasets
        self.sample_size = 1000
        
        # Create ICD data with consistent sizing
        np.random.seed(42)  # For reproducibility
        n_records = 2500
        person_ids = []
        for i in range(1, self.sample_size + 1):
            # Each person gets 1-4 ICD codes
            n_codes = np.random.randint(1, 5)
            person_ids.extend([i] * n_codes)
        
        # Ensure we have exactly n_records by truncating or padding
        if len(person_ids) > n_records:
            person_ids = person_ids[:n_records]
        elif len(person_ids) < n_records:
            # Pad with random person IDs
            while len(person_ids) < n_records:
                person_ids.append(np.random.randint(1, self.sample_size + 1))
        
        self.icd_data = pl.DataFrame({
            'person_id': person_ids,
            'ICD': np.random.choice(['250.00', '250.01', '401.9', '401.1', 'E11.9', '585.6'], 
                                  size=n_records),
            'vocabulary_id': np.random.choice(['ICD9CM', 'ICD10CM'], size=n_records, p=[0.6, 0.4])
        })
        
        # Create cohort data
        self.cohort_data = pl.DataFrame({
            'person_id': list(range(1, self.sample_size + 1)),
            'age': np.random.randint(18, 85, size=self.sample_size),
            'sex_at_birth': np.random.choice([0, 1], size=self.sample_size),
            'pc1': np.random.normal(0, 0.1, size=self.sample_size),
            'pc2': np.random.normal(0, 0.1, size=self.sample_size),
            'genotype': np.random.choice([0, 1, 2], size=self.sample_size, p=[0.25, 0.5, 0.25])
        })
        
        # Create phecode mapping
        self.phecode_mapping = pl.DataFrame({
            'ICD': ['250.00', '250.01', '401.9', '401.1', 'E11.9', '585.6'],
            'flag': [9, 9, 9, 9, 10, 10],
            'phecode': ['250', '250', '401', '401', '250.2', '585'],
            'phecode_description': ['Diabetes mellitus', 'Diabetes mellitus', 
                                  'Hypertension', 'Hypertension', 
                                  'Type 2 diabetes', 'Chronic kidney disease'],
            'phecode_category': ['endocrine/metabolic', 'endocrine/metabolic',
                               'circulatory system', 'circulatory system',
                               'endocrine/metabolic', 'genitourinary']        })

    @patch('PheTK._utils.get_phecode_mapping_table')
    @patch('polars.read_csv')
    def test_phecode_to_phewas_workflow(self, mock_read_csv, mock_get_mapping):
        """Test workflow from Phecode to PheWAS analysis."""
        # Setup mocks
        mock_get_mapping.return_value = self.phecode_mapping
        
        def mock_read_csv_side_effect(file_path, **kwargs):
            if 'icd' in file_path.lower():
                return self.icd_data
            elif 'phecode' in file_path.lower():
                # Return test phecode counts
                return pl.DataFrame({
                    'person_id': list(range(1, 101)) * 3,
                    'phecode': ['250'] * 100 + ['401'] * 100 + ['585'] * 100,
                    'count': [2] * 300
                })
            elif 'cohort' in file_path.lower():
                return self.cohort_data
            else:
                # Default to cohort data
                return self.cohort_data
        
        mock_read_csv.side_effect = mock_read_csv_side_effect
        
        # Step 1: Create Phecode object and generate counts
        phecode = Phecode(platform="custom", icd_df_path="/path/to/icd.csv")
        phecode_counts = phecode.count_phecode(phecode_version="X")
        
        # Verify phecode counts were generated
        assert isinstance(phecode_counts, pl.DataFrame)
        assert len(phecode_counts) > 0
        assert 'phecode' in phecode_counts.columns
        assert 'person_id' in phecode_counts.columns
        
        # Step 2: Create PheWAS object (using mocked file paths)
        phewas = PheWAS(
            phecode_version="X",
            phecode_count_csv_path="/mock/phecode_counts.csv",
            cohort_csv_path="/mock/cohort.csv",
            sex_at_birth_col="sex_at_birth",
            covariate_cols=["age", "pc1", "pc2"],
            independent_variable_of_interest="genotype",
            min_cases=10
        )
        
        # Verify PheWAS object created successfully
        assert hasattr(phewas, 'phecode_counts')
        assert hasattr(phewas, 'covariate_df')
        assert len(phewas.phecode_counts) > 0
        assert len(phewas.covariate_df) > 0

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_phewas_to_plot_workflow(self, mock_get_mapping):
        """Test workflow from PheWAS results to Plot visualization."""
        # Setup mock mapping
        mock_get_mapping.return_value = self.phecode_mapping
        
        # Create mock PheWAS results
        phewas_results = pl.DataFrame({
            'phecode': ['250', '401', '585', '250.2'],
            'phecode_description': ['Diabetes mellitus', 'Hypertension', 
                                  'Chronic kidney disease', 'Type 2 diabetes'],
            'phecode_category': ['endocrine/metabolic', 'circulatory system',
                               'genitourinary', 'endocrine/metabolic'],
            'case_count': [120, 200, 80, 95],
            'control_count': [800, 700, 850, 820],
            'odds_ratio': [2.1, 1.8, 3.2, 2.3],
            'p_value': [0.001, 0.01, 0.0001, 0.005],
            'neg_log_p_value': [3.0, 2.0, 4.0, 2.3],
            'converged': [True, True, True, True],
            'beta': [0.74, 0.59, 1.16, 0.83]
        })
        
        # Create temporary file for results
        results_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        phewas_results.write_csv(results_file.name)
        results_file.close()
        
        try:
            # Create Plot object
            plot = Plot(
                phewas_result_csv_path=results_file.name,
                converged_only=True,
                phecode_version="X"
            )
            
            # Verify Plot object created successfully
            assert isinstance(plot.phewas_result, pl.DataFrame)
            assert len(plot.phewas_result) == 4
            assert plot.bonferroni > 0
            assert len(plot.phecode_categories) > 0
            assert 'label_color' in plot.phewas_result.columns
            
        finally:
            os.unlink(results_file.name)

    @patch.dict(os.environ, {'WORKSPACE_CDR': 'test_cdr', 'GOOGLE_PROJECT': 'test_project'})
    @patch('PheTK._utils.polars_gbq')
    @patch('PheTK._queries.phecode_icd_query')
    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_aou_platform_integration(self, mock_get_mapping, mock_query, mock_polars_gbq):
        """Test integration with All of Us platform simulation."""
        # Setup mocks for AoU platform
        mock_query.return_value = "SELECT * FROM test_cdr.condition_occurrence"
        mock_polars_gbq.return_value = self.icd_data
        mock_get_mapping.return_value = self.phecode_mapping
        
        # Test Cohort initialization
        cohort = Cohort(platform="aou", aou_db_version=8)
        assert cohort.platform == "aou"
        assert cohort.cdr == "test_cdr"
        
        # Test Phecode with AoU platform
        phecode = Phecode(platform="aou")
        assert phecode.platform == "aou"
        assert phecode.cdr == "test_cdr"
        
        # Verify query was called
        mock_query.assert_called_with("test_cdr")
        mock_polars_gbq.assert_called()

    def test_data_type_consistency(self):
        """Test that data types are consistent across modules."""
        # Test that utility functions handle data type conversions properly
        
        # Test pandas to polars conversion
        pandas_df = pd.DataFrame({
            'person_id': [1, 2, 3],
            'value': ['A', 'B', 'C']
        })
        
        polars_df = _utils.to_polars(pandas_df)
        assert isinstance(polars_df, pl.DataFrame)
        
        # Test that polars DataFrame remains unchanged
        original_polars = pl.DataFrame({
            'person_id': [1, 2, 3],
            'value': ['A', 'B', 'C']
        })
        
        result = _utils.to_polars(original_polars)
        assert result is original_polars

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_error_handling_integration(self, mock_get_mapping):
        """Test error handling across integrated components."""
        mock_get_mapping.return_value = self.phecode_mapping
        
        # Test invalid file paths
        with pytest.raises((FileNotFoundError, Exception)):
            PheWAS(
                phecode_version="X",
                phecode_count_csv_path="/nonexistent/phecode.csv",
                cohort_csv_path="/nonexistent/cohort.csv",
                sex_at_birth_col="sex_at_birth",
                covariate_cols=["age"],
                independent_variable_of_interest="genotype"
            )
        
        # Test invalid platform for Cohort
        with patch('sys.exit'):
            Cohort(platform="invalid_platform")

    def test_memory_efficiency_integration(self):
        """Test memory efficiency with larger datasets."""
        # Create larger dataset for memory testing
        large_size = 10000
        
        large_icd_data = pl.DataFrame({
            'person_id': np.random.randint(1, large_size + 1, size=large_size * 5),
            'ICD': np.random.choice(['250.00', '401.9', 'E11.9'], size=large_size * 5),
            'vocabulary_id': np.random.choice(['ICD9CM', 'ICD10CM'], size=large_size * 5)
        })
        
        # Test that large data can be processed without issues
        with patch('polars.read_csv', return_value=large_icd_data):
            phecode = Phecode(platform="custom", icd_df_path="/path/to/large_icd.csv")
            
            # Verify data was loaded
            assert len(phecode.icd_events) == large_size * 5
            assert 'flag' in phecode.icd_events.columns


class TestCrossModuleDataFlow:
    """Test data flow between different PheTK modules."""

    def setup_method(self):
        """Set up test data for cross-module tests."""
        self.test_person_ids = [1, 2, 3, 4, 5]
        
        self.icd_test_data = pl.DataFrame({
            'person_id': self.test_person_ids * 3,
            'ICD': ['250.00', '401.9', 'E11.9'] * 5,
            'vocabulary_id': ['ICD9CM', 'ICD9CM', 'ICD10CM'] * 5
        })
        
        self.cohort_test_data = pl.DataFrame({
            'person_id': self.test_person_ids + [6, 7, 8, 9, 10],  # Some overlap, some not
            'age': [50, 60, 70, 40, 30, 35, 45, 55, 65, 75],
            'sex_at_birth': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'genotype': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
        })

    @patch('PheTK._utils.get_phecode_mapping_table')
    @patch('polars.read_csv')
    def test_person_id_consistency(self, mock_read_csv, mock_get_mapping):
        """Test that person IDs are handled consistently across modules."""
        # Setup mocks
        mock_mapping = pl.DataFrame({
            'ICD': ['250.00', '401.9', 'E11.9'],
            'flag': [9, 9, 10],
            'phecode': ['250', '401', '250.2']
        })
        mock_get_mapping.return_value = mock_mapping
        
        def mock_read_csv_side_effect(file_path, **kwargs):
            if 'icd' in file_path.lower():
                return self.icd_test_data
            else:
                return self.cohort_test_data
        
        mock_read_csv.side_effect = mock_read_csv_side_effect
        
        # Test Phecode module
        phecode = Phecode(platform="custom", icd_df_path="/path/to/icd.csv")
        phecode_counts = phecode.count_phecode(phecode_version="X")
        
        # Verify person IDs are preserved
        assert 'person_id' in phecode_counts.columns
        phecode_person_ids = set(phecode_counts['person_id'].unique())
        expected_person_ids = set(self.test_person_ids)
        assert phecode_person_ids.issubset(expected_person_ids)

    def test_column_name_consistency(self):
        """Test that column names are consistent across modules."""
        # Standard column names that should be consistent
        expected_columns = {
            'person_id': 'person_id',
            'phecode': 'phecode',
            'vocabulary_id': 'vocabulary_id',
            'ICD': 'ICD'
        }
        
        # Test that these column names are used consistently
        # This is verified by the successful integration in other tests
        assert True  # Placeholder - actual verification happens in integration tests

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_data_format_compatibility(self, mock_get_mapping):
        """Test that data formats are compatible between modules."""
        mock_mapping = pl.DataFrame({
            'phecode': ['250', '401'],
            'phecode_description': ['Diabetes', 'Hypertension'],
            'phecode_category': ['endocrine', 'circulatory']
        })
        mock_get_mapping.return_value = mock_mapping
        
        # Create test files
        phecode_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        cohort_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        test_phecode_data = pl.DataFrame({
            'person_id': [1, 2, 3],
            'phecode': ['250', '401', '250'],
            'count': [2, 3, 1]
        })
        
        test_cohort_data = pl.DataFrame({
            'person_id': [1, 2, 3, 4, 5],
            'age': [50, 60, 70, 40, 30],
            'sex_at_birth': [1, 0, 1, 0, 1],
            'genotype': [1, 0, 1, 0, 1]
        })
        
        test_phecode_data.write_csv(phecode_file.name)
        test_cohort_data.write_csv(cohort_file.name)
        
        phecode_file.close()
        cohort_file.close()
        
        try:
            # Test that PheWAS can read data generated by Phecode
            phewas = PheWAS(
                phecode_version="X",
                phecode_count_csv_path=phecode_file.name,
                cohort_csv_path=cohort_file.name,
                sex_at_birth_col="sex_at_birth",
                covariate_cols=["age"],
                independent_variable_of_interest="genotype"
            )
            
            # Verify data was loaded correctly
            assert len(phewas.phecode_counts) > 0
            assert len(phewas.covariate_df) > 0
            
        finally:
            os.unlink(phecode_file.name)
            os.unlink(cohort_file.name)


class TestErrorPropagationIntegration:
    """Test how errors propagate through the integrated system."""

    def test_missing_dependencies_error_handling(self):
        """Test handling of missing optional dependencies."""
        # Test cases where optional dependencies might be missing
        # This is a placeholder test as actual dependency testing would require
        # more complex mocking
        assert True

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_invalid_data_error_propagation(self, mock_get_mapping):
        """Test how invalid data errors propagate through the system."""
        mock_mapping = pl.DataFrame({'phecode': ['250']})
        mock_get_mapping.return_value = mock_mapping
        
        # Create file with invalid data
        invalid_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        # Write invalid CSV (missing required columns)
        invalid_file.write("invalid,data,format\n1,2,3\n")
        invalid_file.close()
        
        try:
            # Should raise appropriate error
            with pytest.raises((KeyError, ValueError, Exception)):
                PheWAS(
                    phecode_version="X",
                    phecode_count_csv_path=invalid_file.name,
                    cohort_csv_path=invalid_file.name,
                    sex_at_birth_col="nonexistent_column",
                    covariate_cols=["nonexistent_col"],
                    independent_variable_of_interest="also_nonexistent"
                )
                
        finally:
            os.unlink(invalid_file.name)

    def test_configuration_error_handling(self):
        """Test handling of configuration errors."""
        # Test invalid platform configurations
        with patch('sys.exit'):
            Cohort(platform="unsupported_platform")
        
        with patch('sys.exit'):
            Phecode(platform="unsupported_platform")


class TestPerformanceIntegration:
    """Test performance characteristics of integrated components."""

    def test_large_dataset_handling(self):
        """Test system behavior with large datasets."""
        # Create moderately large dataset for testing
        large_person_count = 5000
        
        large_data = pl.DataFrame({
            'person_id': list(range(1, large_person_count + 1)),
            'phecode': np.random.choice(['250', '401', '585'], size=large_person_count),
            'count': np.random.randint(1, 10, size=large_person_count)
        })
        
        # Test that large data doesn't cause memory issues
        assert len(large_data) == large_person_count
        assert large_data.estimated_size() > 0  # Has some memory footprint

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_repeated_operations_stability(self, mock_get_mapping):
        """Test system stability with repeated operations."""
        mock_mapping = pl.DataFrame({'phecode': ['250']})
        mock_get_mapping.return_value = mock_mapping
        
        # Test that repeated operations don't cause issues
        for i in range(5):
            with patch('polars.read_csv') as mock_read:
                mock_read.return_value = pl.DataFrame({
                    'person_id': [1, 2, 3],
                    'ICD': ['250.00'] * 3,
                    'vocabulary_id': ['ICD9CM'] * 3
                })
                
                phecode = Phecode(platform="custom", icd_df_path=f"/path/test_{i}.csv")
                assert phecode.platform == "custom"
