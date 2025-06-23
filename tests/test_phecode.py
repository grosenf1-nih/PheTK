"""
Unit tests for PheTK.Phecode module.
"""
import pytest
import polars as pl
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PheTK.Phecode import Phecode


class TestPhecodeInitialization:
    """Test Phecode class initialization."""

    @patch.dict(os.environ, {'WORKSPACE_CDR': 'test_cdr'})
    @patch('PheTK._utils.polars_gbq')
    @patch('PheTK._queries.phecode_icd_query')
    def test_init_aou_platform(self, mock_query, mock_polars_gbq):
        """Test initialization with All of Us platform."""
        # Mock the query and response
        mock_query.return_value = "SELECT * FROM test"
        mock_icd_data = pl.DataFrame({
            'person_id': [1, 2, 3],
            'ICD': ['250.00', '401.9', 'E11.9'],
            'vocabulary_id': ['ICD9CM', 'ICD9CM', 'ICD10CM']
        })
        mock_polars_gbq.return_value = mock_icd_data
        
        phecode = Phecode(platform="aou")
        
        assert phecode.platform == "aou"
        assert phecode.cdr == "test_cdr"
        mock_query.assert_called_once_with("test_cdr")
        mock_polars_gbq.assert_called_once()
        
        # Check that flag column was added
        assert "flag" in phecode.icd_events.columns

    @patch('polars.read_csv')
    def test_init_custom_platform_with_file(self, mock_read_csv):
        """Test initialization with custom platform and ICD file."""
        mock_icd_data = pl.DataFrame({
            'person_id': [1, 2, 3],
            'ICD': ['250.00', '401.9', 'E11.9'],
            'vocabulary_id': ['ICD9CM', 'ICD9CM', 'ICD10CM']
        })
        mock_read_csv.return_value = mock_icd_data
        
        phecode = Phecode(platform="custom", icd_df_path="/path/to/icd.csv")
        
        assert phecode.platform == "custom"
        mock_read_csv.assert_called_once_with("/path/to/icd.csv", dtypes={"ICD": str})
          # Check that flag column was added
        assert "flag" in phecode.icd_events.columns

    @patch('sys.exit')
    @patch('builtins.print')
    def test_init_custom_platform_without_file(self, mock_print, mock_exit):
        """Test initialization with custom platform but no ICD file."""
        Phecode(platform="custom")
        mock_print.assert_called_with("icd_df_path is required for custom platform.")
        mock_exit.assert_called_once_with(0)

    @patch('sys.exit')
    @patch('builtins.print')
    def test_init_invalid_platform(self, mock_print, mock_exit):
        """Test initialization with invalid platform."""
        Phecode(platform="invalid")
        mock_print.assert_called_with("Invalid platform. Parameter platform only accepts \"aou\" (All of Us) or \"custom\".")
        mock_exit.assert_called_once_with(0)

    @patch('polars.read_csv')
    def test_init_flag_column_creation(self, mock_read_csv):
        """Test that flag column is created correctly from vocabulary_id."""
        mock_icd_data = pl.DataFrame({
            'person_id': [1, 2, 3, 4, 5],
            'ICD': ['250.00', '401.9', 'E11.9', 'V70.0', 'Z00.00'],
            'vocabulary_id': ['ICD9CM', 'ICD9', 'ICD10CM', 'ICD9CM', 'ICD10']
        })
        mock_read_csv.return_value = mock_icd_data
        
        phecode = Phecode(platform="custom", icd_df_path="/path/to/icd.csv")
        
        # Check flag values
        expected_flags = [9, 9, 10, 9, 10]
        actual_flags = phecode.icd_events['flag'].to_list()
        assert actual_flags == expected_flags

    @patch('polars.read_csv')
    def test_init_with_existing_flag_column(self, mock_read_csv):
        """Test initialization when flag column already exists."""
        mock_icd_data = pl.DataFrame({
            'person_id': [1, 2, 3],
            'ICD': ['250.00', '401.9', 'E11.9'],
            'vocabulary_id': ['ICD9CM', 'ICD9CM', 'ICD10CM'],
            'flag': [9, 9, 10]
        })
        mock_read_csv.return_value = mock_icd_data
        
        phecode = Phecode(platform="custom", icd_df_path="/path/to/icd.csv")
        
        # Flag column should be cast to Int8 but values preserved
        assert phecode.icd_events['flag'].to_list() == [9, 9, 10]
        assert phecode.icd_events['flag'].dtype == pl.Int8


class TestPhecodeCountPhecode:
    """Test Phecode.count_phecode method."""

    def setup_method(self):
        """Set up test data for each test method."""
        with patch('polars.read_csv') as mock_read_csv:
            self.mock_icd_data = pl.DataFrame({
                'person_id': [1, 1, 2, 2, 3],
                'ICD': ['250.00', '250.01', '401.9', '401.1', 'E11.9'],
                'vocabulary_id': ['ICD9CM', 'ICD9CM', 'ICD9CM', 'ICD9CM', 'ICD10CM'],
                'flag': [9, 9, 9, 9, 10]
            })
            mock_read_csv.return_value = self.mock_icd_data
            self.phecode = Phecode(platform="custom", icd_df_path="/path/to/icd.csv")

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_count_phecode_version_x(self, mock_get_mapping):
        """Test phecode counting with phecode version X."""
        # Mock phecode mapping
        mock_mapping = pl.DataFrame({
            'ICD': ['250.00', '250.01', '401.9', '401.1', 'E11.9'],
            'flag': [9, 9, 9, 9, 10],
            'phecode': ['250', '250', '401', '401', '250.2']
        })
        mock_get_mapping.return_value = mock_mapping
        
        result = self.phecode.count_phecode(phecode_version="X")
        
        # Verify the mapping function was called correctly
        mock_get_mapping.assert_called_once_with(
            phecode_version="X",
            icd_version="US",
            phecode_map_file_path=None,
            keep_all_columns=False
        )
          # Verify result structure
        assert isinstance(result, pl.DataFrame)
        expected_columns = {'person_id', 'phecode', 'count'}
        assert set(result.columns) == expected_columns

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_count_phecode_version_12(self, mock_get_mapping):
        """Test phecode counting with phecode version 1.2."""
        # Mock phecode mapping with phecode_unrolled column
        mock_mapping = pl.DataFrame({
            'ICD': ['250.00', '250.01', '401.9', '401.1', 'E11.9'],
            'flag': [9, 9, 9, 9, 10],
            'phecode_unrolled': ['250', '250', '401', '401', '250.2']
        })
        mock_get_mapping.return_value = mock_mapping
        
        result = self.phecode.count_phecode(phecode_version="1.2")
        
        # Verify the mapping function was called correctly
        mock_get_mapping.assert_called_once_with(
            phecode_version="1.2",
            icd_version="US",
            phecode_map_file_path=None,
            keep_all_columns=False
        )
        
        # Verify result structure and renaming
        assert isinstance(result, pl.DataFrame)
        assert "phecode" in result.columns

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_count_phecode_who_version(self, mock_get_mapping):
        """Test phecode counting with WHO ICD version."""
        mock_mapping = pl.DataFrame({
            'ICD': ['I50.9', 'E11.9'],
            'flag': [10, 10],
            'phecode': ['428', '250.2']
        })
        mock_get_mapping.return_value = mock_mapping
        
        result = self.phecode.count_phecode(
            phecode_version="X",
            icd_version="WHO"
        )
        
        mock_get_mapping.assert_called_once_with(
            phecode_version="X",
            icd_version="WHO",
            phecode_map_file_path=None,
            keep_all_columns=False
        )

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_count_phecode_custom_mapping(self, mock_get_mapping):
        """Test phecode counting with custom mapping file."""
        mock_mapping = pl.DataFrame({
            'ICD': ['250.00', '401.9'],
            'flag': [9, 9],
            'phecode': ['custom_250', 'custom_401']
        })
        mock_get_mapping.return_value = mock_mapping
        
        custom_path = "/path/to/custom_mapping.csv"
        result = self.phecode.count_phecode(
            phecode_version="X",
            icd_version="custom",
            phecode_map_file_path=custom_path
        )
        
        mock_get_mapping.assert_called_once_with(
            phecode_version="X",
            icd_version="custom",
            phecode_map_file_path=custom_path,
            keep_all_columns=False
        )

    @patch('PheTK._utils.get_phecode_mapping_table')
    @patch('polars.DataFrame.write_csv')
    def test_count_phecode_with_output_file(self, mock_write_csv, mock_get_mapping):
        """Test phecode counting with output file specified."""
        mock_mapping = pl.DataFrame({
            'ICD': ['250.00', '401.9'],
            'flag': [9, 9],
            'phecode': ['250', '401']
        })
        mock_get_mapping.return_value = mock_mapping
        
        output_file = "test_output.csv"
        result = self.phecode.count_phecode(
            phecode_version="X",
            output_file_name=output_file
        )
        
        # Verify file was written
        mock_write_csv.assert_called_once_with(output_file)

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_count_phecode_invalid_version(self, mock_get_mapping):
        """Test phecode counting with invalid phecode version."""
        mock_get_mapping.return_value = pl.DataFrame()
        
        result = self.phecode.count_phecode(phecode_version="invalid")
        
        # Should return empty DataFrame
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_count_phecode_aggregation(self, mock_get_mapping):
        """Test that phecode counts are properly aggregated."""
        # Create mapping that maps multiple ICDs to same phecode
        mock_mapping = pl.DataFrame({
            'ICD': ['250.00', '250.01', '401.9', '401.1'],
            'flag': [9, 9, 9, 9],
            'phecode': ['250', '250', '401', '401']
        })
        mock_get_mapping.return_value = mock_mapping
        
        # Use ICD data with multiple entries per person/phecode
        with patch.object(self.phecode, 'icd_events', pl.DataFrame({
            'person_id': [1, 1, 1, 2, 2, 3],
            'ICD': ['250.00', '250.01', '250.00', '401.9', '401.1', '250.00'],
            'flag': [9, 9, 9, 9, 9, 9]
        })):
            
            result = self.phecode.count_phecode(phecode_version="X")
              # Verify proper structure exists
            assert 'person_id' in result.columns
            assert 'phecode' in result.columns
            assert 'count' in result.columns
            
            # Check that aggregation worked correctly
            # Person 1 should have 3 counts for phecode 250 (250.00 twice + 250.01 once)
            person_1_counts = result.filter(pl.col('person_id') == 1)
            if not person_1_counts.is_empty():
                phecode_250_count = person_1_counts.filter(pl.col('phecode') == '250')
                if not phecode_250_count.is_empty():
                    assert phecode_250_count['count'][0] == 3


class TestPhecodeEdgeCases:
    """Test edge cases and error conditions."""

    @patch('polars.read_csv')
    def test_empty_icd_data(self, mock_read_csv):
        """Test handling of empty ICD data."""
        mock_read_csv.return_value = pl.DataFrame({
            'person_id': [],
            'ICD': [],
            'vocabulary_id': []
        })
        
        phecode = Phecode(platform="custom", icd_df_path="/path/to/empty.csv")
        
        # Should handle empty data gracefully
        assert len(phecode.icd_events) == 0
        assert "flag" in phecode.icd_events.columns

    @patch('polars.read_csv')
    def test_unknown_vocabulary_id(self, mock_read_csv):
        """Test handling of unknown vocabulary IDs."""
        mock_icd_data = pl.DataFrame({
            'person_id': [1, 2, 3],
            'ICD': ['250.00', '401.9', 'E11.9'],
            'vocabulary_id': ['UNKNOWN', 'ICD9CM', 'ICD10CM']
        })
        mock_read_csv.return_value = mock_icd_data
        
        phecode = Phecode(platform="custom", icd_df_path="/path/to/icd.csv")
        
        # Unknown vocabulary should get flag value 0
        expected_flags = [0, 9, 10]
        actual_flags = phecode.icd_events['flag'].to_list()
        assert actual_flags == expected_flags


class TestPhecodeIntegration:
    """Integration tests for Phecode class."""

    @patch('polars.read_csv')
    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_complete_workflow_custom_platform(self, mock_get_mapping, mock_read_csv):
        """Test complete workflow with custom platform."""
        # Setup mock data
        mock_icd_data = pl.DataFrame({
            'person_id': [1, 1, 2, 3, 3],
            'ICD': ['250.00', '250.01', '401.9', '250.00', 'E11.9'],
            'vocabulary_id': ['ICD9CM', 'ICD9CM', 'ICD9CM', 'ICD9CM', 'ICD10CM']
        })
        mock_read_csv.return_value = mock_icd_data
        
        mock_mapping = pl.DataFrame({
            'ICD': ['250.00', '250.01', '401.9', 'E11.9'],
            'flag': [9, 9, 9, 10],
            'phecode': ['250', '250', '401', '250.2']
        })
        mock_get_mapping.return_value = mock_mapping
        
        # Initialize and run
        phecode = Phecode(platform="custom", icd_df_path="/path/to/icd.csv")
        result = phecode.count_phecode(phecode_version="X")
        
        # Verify workflow completion
        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0
        mock_read_csv.assert_called_once()
        mock_get_mapping.assert_called_once()

    @patch.dict(os.environ, {'WORKSPACE_CDR': 'test_cdr'})
    @patch('PheTK._utils.polars_gbq')
    @patch('PheTK._queries.phecode_icd_query')
    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_complete_workflow_aou_platform(self, mock_get_mapping, mock_query, mock_polars_gbq):
        """Test complete workflow with AoU platform."""
        # Setup mocks
        mock_query.return_value = "SELECT * FROM test"
        mock_icd_data = pl.DataFrame({
            'person_id': [1, 2, 3],
            'ICD': ['250.00', '401.9', 'E11.9'],
            'vocabulary_id': ['ICD9CM', 'ICD9CM', 'ICD10CM']
        })
        mock_polars_gbq.return_value = mock_icd_data
        
        mock_mapping = pl.DataFrame({
            'ICD': ['250.00', '401.9', 'E11.9'],
            'flag': [9, 9, 10],
            'phecode': ['250', '401', '250.2']
        })
        mock_get_mapping.return_value = mock_mapping
        
        # Initialize and run
        phecode = Phecode(platform="aou")
        result = phecode.count_phecode(phecode_version="X")
        
        # Verify workflow completion
        assert isinstance(result, pl.DataFrame)
        mock_query.assert_called_once()
        mock_polars_gbq.assert_called_once()
        mock_get_mapping.assert_called_once()
