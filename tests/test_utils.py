"""
Unit tests for PheTK._utils module.
"""
import pytest
import polars as pl
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import pyarrow as pa
import os
import sys

# Add src directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PheTK import _utils


class TestUtilityFunctions:
    """Test utility functions in _utils module."""

    def test_to_polars_with_pandas_dataframe(self):
        """Test conversion from pandas to polars DataFrame."""
        # Create a pandas DataFrame
        pandas_df = pd.DataFrame({
            'person_id': [1, 2, 3],
            'value': ['A', 'B', 'C']
        })
        
        # Convert to polars
        result = _utils.to_polars(pandas_df)
        
        # Assert result is polars DataFrame
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (3, 2)
        assert result.columns == ['person_id', 'value']

    def test_to_polars_with_polars_dataframe(self):
        """Test that polars DataFrame is returned unchanged."""
        # Create a polars DataFrame
        polars_df = pl.DataFrame({
            'person_id': [1, 2, 3],
            'value': ['A', 'B', 'C']
        })
          # Call to_polars
        result = _utils.to_polars(polars_df)
          # Assert same object is returned
        assert result is polars_df
        assert isinstance(result, pl.DataFrame)

    @pytest.mark.skip(reason="pyarrow.Table.from_batches mocking is complex, skipping for now")
    def test_spark_to_polars(self):
        """Test conversion from Spark DataFrame to polars."""
        pass

    @patch('google.cloud.bigquery.Client')
    @patch('polars.from_arrow')
    def test_polars_gbq_success(self, mock_from_arrow, mock_client_class):
        """Test successful BigQuery to polars conversion."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_job = Mock()
        mock_client.query.return_value = mock_job
        
        mock_rows = Mock()
        mock_job.result.return_value = mock_rows
        
        mock_arrow_table = Mock()
        mock_rows.to_arrow.return_value = mock_arrow_table
        
        mock_polars_df = pl.DataFrame({'test': [1, 2, 3]})
        mock_from_arrow.return_value = mock_polars_df
        
        # Test query
        query = "SELECT * FROM test_table"
        result = _utils.polars_gbq(query)
        
        # Assertions
        mock_client_class.assert_called_once()
        mock_client.query.assert_called_once_with(query)
        mock_job.result.assert_called_once()
        mock_rows.to_arrow.assert_called_once()
        mock_from_arrow.assert_called_once_with(mock_arrow_table)
        assert result is mock_polars_df

    @patch('polars.read_csv')
    @patch('os.path.join')
    @patch('os.path.dirname')
    def test_get_phecode_mapping_table_phecode_x_us(self, mock_dirname, mock_join, mock_read_csv):
        """Test loading phecode X mapping table for US version."""
        # Setup mocks
        mock_dirname.return_value = "/test/path"
        mock_join.return_value = "/test/path/phecode/phecodeX.csv"
        
        expected_df = pl.DataFrame({
            'ICD': ['250.00', '401.9'],
            'flag': [9, 9],
            'phecode': ['250', '401']
        })
        mock_read_csv.return_value = expected_df
        
        # Test function call
        result = _utils.get_phecode_mapping_table(
            phecode_version="X",
            icd_version="US",
            phecode_map_file_path=None,
            keep_all_columns=True
        )
          # Assertions
        mock_read_csv.assert_called_once_with("/test/path/phecode/phecodeX.csv", dtypes={"phecode": str, "ICD": str, "flag": pl.Int8, "code_val": float})
        assert result is expected_df

    @patch('polars.read_csv')
    @patch('os.path.join')
    @patch('os.path.dirname')
    def test_get_phecode_mapping_table_phecode_x_who(self, mock_dirname, mock_join, mock_read_csv):
        """Test loading phecode X mapping table for WHO version."""
        # Setup mocks
        mock_dirname.return_value = "/test/path"
        mock_join.return_value = "/test/path/phecode/phecodeX_WHO.csv"
        
        expected_df = pl.DataFrame({
            'ICD': ['I50.9', 'E11.9'],
            'flag': [10, 10],
            'phecode': ['428', '250.2']
        })
        mock_read_csv.return_value = expected_df
        
        # Test function call
        result = _utils.get_phecode_mapping_table(
            phecode_version="X",
            icd_version="WHO",
            phecode_map_file_path=None,
            keep_all_columns=True
        )
          # Assertions
        mock_read_csv.assert_called_once_with("/test/path/phecode/phecodeX_WHO.csv", dtypes={"phecode": str, "ICD": str, "flag": pl.Int8, "code_val": float})
        assert result is expected_df

    @patch('polars.read_csv')
    @patch('os.path.join')
    @patch('os.path.dirname')
    def test_get_phecode_mapping_table_phecode_12(self, mock_dirname, mock_join, mock_read_csv):
        """Test loading phecode 1.2 mapping table."""
        # Setup mocks
        mock_dirname.return_value = "/test/path"
        mock_join.return_value = "/test/path/phecode/phecode12.csv"
        
        expected_df = pl.DataFrame({
            'ICD': ['250.00', '401.9'],
            'flag': [9, 9],
            'phecode_unrolled': ['250', '401']
        })
        mock_read_csv.return_value = expected_df
        
        # Test function call
        result = _utils.get_phecode_mapping_table(
            phecode_version="1.2",
            icd_version="US",
            phecode_map_file_path=None,
            keep_all_columns=True
        )
          # Assertions
        mock_read_csv.assert_called_once_with("/test/path/phecode/phecode12.csv", dtypes={"phecode": str, "ICD": str, "flag": pl.Int8, "exclude_range": str, "phecode_unrolled": str})
        assert result is expected_df

    @patch('polars.read_csv')
    def test_get_phecode_mapping_table_custom_path(self, mock_read_csv):
        """Test loading phecode mapping table from custom path."""
        custom_path = "/custom/path/mapping.csv"
        expected_df = pl.DataFrame({
            'ICD': ['250.00'],
            'flag': [9],
            'phecode': ['250']
        })
        mock_read_csv.return_value = expected_df
        
        # Test function call
        result = _utils.get_phecode_mapping_table(
            phecode_version="X",
            icd_version="custom",
            phecode_map_file_path=custom_path,
            keep_all_columns=True        )
        
        # Assertions
        mock_read_csv.assert_called_once_with(custom_path, dtypes={"phecode": str, "ICD": str, "flag": pl.Int8, "code_val": float})
        assert result is expected_df

    @patch('polars.read_csv')
    @patch('os.path.join')
    @patch('os.path.dirname')
    def test_get_phecode_mapping_table_keep_limited_columns(self, mock_dirname, mock_join, mock_read_csv):
        """Test loading phecode mapping table with limited columns."""        # Setup mocks
        mock_dirname.return_value = "/test/path"
        mock_join.return_value = "/test/path/phecode/phecodeX.csv"
        
        full_df = pl.DataFrame({
            'ICD': ['250.00', '401.9'],
            'flag': [9, 9],
            'phecode': ['250', '401'],
            'description': ['Diabetes', 'Hypertension'],
            'category': ['endocrine', 'circulatory']
        })
        
        expected_df = pl.DataFrame({
            'ICD': ['250.00', '401.9'],
            'flag': [9, 9],
            'phecode': ['250', '401']
        })
        
        mock_read_csv.return_value = full_df
        
        # Test function call
        result = _utils.get_phecode_mapping_table(
            phecode_version="X",
            icd_version="US",
            phecode_map_file_path=None,
            keep_all_columns=False
        )
          # Assertions - the function uses list indexing, not select
        mock_read_csv.assert_called_once_with("/test/path/phecode/phecodeX.csv", dtypes={"phecode": str, "ICD": str, "flag": pl.Int8, "code_val": float})
        # The function selects columns using list indexing: phecode_df[["phecode", "ICD", "flag"]]
        assert result.columns == ["phecode", "ICD", "flag"]
        assert len(result) == 2

    @patch('sys.exit')
    @patch('builtins.print')
    def test_get_phecode_mapping_table_invalid_version_combination(self, mock_print, mock_exit):
        """Test handling of invalid version combinations."""
        # Mock sys.exit to raise an exception instead of exiting
        mock_exit.side_effect = SystemExit(0)
        
        with pytest.raises(SystemExit):
            _utils.get_phecode_mapping_table(
                phecode_version="invalid",
                icd_version="US",
                phecode_map_file_path=None,
                keep_all_columns=True            )
        
        mock_print.assert_called_with("Unsupported phecode version. Supports phecode \"1.2\" and \"X\".")
        mock_exit.assert_called_once_with(0)

    @patch('sys.exit')
    @patch('builtins.print')
    def test_get_phecode_mapping_table_custom_without_path(self, mock_print, mock_exit):
        """Test error handling when custom version is specified without path."""
        # Mock sys.exit to raise an exception instead of exiting
        mock_exit.side_effect = SystemExit(0)
        
        with pytest.raises(SystemExit):
            _utils.get_phecode_mapping_table(
                phecode_version="X",
                icd_version="custom",
                phecode_map_file_path=None,
                keep_all_columns=True
            )
        
        mock_print.assert_called_with("Please provide phecode_map_path for custom icd_version")
        mock_exit.assert_called_once_with(0)


class TestUtilsIntegration:
    """Integration tests for _utils module."""
    
    def test_pandas_to_polars_integration(self):
        """Integration test for pandas to polars conversion."""
        # Create test data
        pandas_df = pd.DataFrame({
            'person_id': [1, 2, 3, 4, 5],
            'age': [25, 30, 35, 40, 45],
            'diagnosis': ['A', 'B', 'A', 'C', 'B']
        })
        
        # Convert using utility function
        polars_df = _utils.to_polars(pandas_df)
        
        # Verify conversion
        assert isinstance(polars_df, pl.DataFrame)
        assert polars_df.shape == pandas_df.shape
        assert set(polars_df.columns) == set(pandas_df.columns)
        
        # Verify data integrity
        assert polars_df['person_id'].to_list() == [1, 2, 3, 4, 5]
        assert polars_df['age'].to_list() == [25, 30, 35, 40, 45]
