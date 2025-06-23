"""
Unit tests for PheTK.Cohort module.
"""
import pytest
import polars as pl
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Check if Hail is available and properly configured
try:
    import hail as hl
    # Try a simple Hail operation to see if it's properly configured
    hl.init(quiet=True, log=None)
    HAIL_AVAILABLE = True
    hl.stop()
except Exception:
    HAIL_AVAILABLE = False

from PheTK.Cohort import Cohort


class TestCohortInitialization:
    """Test Cohort class initialization."""

    @patch.dict(os.environ, {'WORKSPACE_CDR': 'test_cdr', 'GOOGLE_PROJECT': 'test_project'})
    def test_init_aou_platform_default(self):
        """Test initialization with All of Us platform using defaults."""
        cohort = Cohort(platform="aou", aou_db_version=8)
        
        assert cohort.platform == "aou"
        assert cohort.db_version == 8
        assert cohort.cdr == "test_cdr"
        assert cohort.user_project == "test_project"

    @patch.dict(os.environ, {'WORKSPACE_CDR': 'test_cdr', 'GOOGLE_PROJECT': 'test_project'})
    def test_init_aou_platform_custom_cdr(self):
        """Test initialization with All of Us platform using custom CDR."""
        custom_cdr = "custom_test_cdr"
        cohort = Cohort(platform="aou", aou_db_version=7, aou_omop_cdr=custom_cdr)
        
        assert cohort.platform == "aou"
        assert cohort.db_version == 7
        assert cohort.cdr == custom_cdr
        assert cohort.user_project == "test_project"

    def test_init_custom_platform(self):
        """Test initialization with custom platform."""
        gbq_dataset = "custom_dataset"
        cohort = Cohort(platform="custom", gbq_dataset_id=gbq_dataset)
        
        assert cohort.platform == "custom"
        assert cohort.cdr == gbq_dataset

    def test_init_default_covariate_attributes(self):
        """Test that covariate attributes are initialized to False."""
        cohort = Cohort(platform="custom", gbq_dataset_id="test_dataset")
        
        assert cohort.natural_age is False
        assert cohort.age_at_last_event is False
        assert cohort.sex_at_birth is False
        assert cohort.ehr_length is False
        assert cohort.dx_code_occurrence_count is False
        assert cohort.dx_condition_count is False
        assert cohort.genetic_ancestry is False
        assert cohort.first_n_pcs == 0

    def test_init_invalid_platform(self):
        """Test initialization with invalid platform."""
        with patch('sys.exit') as mock_exit:
            Cohort(platform="invalid")
            mock_exit.assert_called_once_with(0)

    def test_init_aou_invalid_version(self):
        """Test initialization with invalid AoU database version."""
        with patch('sys.exit') as mock_exit:
            Cohort(platform="aou", aou_db_version=99)
            mock_exit.assert_called_once_with(0)

    def test_init_custom_without_dataset_id(self):
        """Test initialization with custom platform but no dataset ID."""
        with patch('sys.exit') as mock_exit:
            Cohort(platform="custom")
            mock_exit.assert_called_once_with(0)

    def test_init_case_insensitive_platform(self):
        """Test that platform parameter is case insensitive."""
        cohort = Cohort(platform="AOU", aou_db_version=8)
        assert cohort.platform == "aou"
        
        cohort = Cohort(platform="Custom", gbq_dataset_id="test")
        assert cohort.platform == "custom"


class TestCohortByGenotype:
    """Test Cohort.by_genotype method."""

    @pytest.mark.skipif(not HAIL_AVAILABLE, reason="Hail/Spark environment not properly configured")
    @patch.dict(os.environ, {'WORKSPACE_CDR': 'test_cdr', 'GOOGLE_PROJECT': 'test_project'})
    def test_by_genotype_parameter_validation(self):
        """Test parameter validation in by_genotype method."""
        cohort = Cohort(platform="aou", aou_db_version=8)
        
        # Test with basic parameters
        with patch('hail.init'), \
             patch('hail.read_matrix_table') as mock_read_mt, \
             patch('hail.Locus.parse') as mock_locus_parse, \
             patch('hail.parse_variant') as mock_parse_variant:
            
            mock_mt = Mock()
            mock_read_mt.return_value = mock_mt
            
            # Mock matrix table count_rows to return integer instead of Mock
            mock_mt.count_rows.return_value = 1
            
            # Mock Hail objects
            mock_locus_parse.return_value = "mock_locus"
            mock_variant = {"locus": "mock_locus", "alleles": ["A", "G"]}
            mock_parse_variant.return_value = mock_variant
            
            # Mock matrix table operations
            mock_filtered_mt = Mock()
            mock_filtered_mt.count_rows.return_value = 1
            mock_mt.filter_rows.return_value = mock_filtered_mt
            
            # Mock the second filter_rows call (for variant filtering)
            mock_filtered_mt2 = Mock()
            mock_filtered_mt2.count_rows.return_value = 1
            mock_filtered_mt.filter_rows.return_value = mock_filtered_mt2
            
            # Mock the entries and to_spark conversion
            mock_entries = Mock()
            mock_spark_df = Mock()
            mock_entries.to_spark.return_value = mock_spark_df
            mock_filtered_mt2.entries.return_value = mock_entries
            
            # Mock the row show method
            mock_mt.row.show = Mock()
            mock_filtered_mt.row.show = Mock()
            mock_filtered_mt2.row.show = Mock()
            
            # Mock _utils.spark_to_polars
            with patch('PheTK.Cohort._utils.spark_to_polars') as mock_spark_to_polars:
                # Mock the allele count check
                mock_spark_to_polars.side_effect = [
                    pl.DataFrame({"info.AF": [[0.5]]}),  # For allele count check
                    pl.DataFrame({  # For GT data
                        "s": [1, 2, 3],
                        "GT.alleles": [[0, 0], [0, 1], [1, 1]]
                    })
                ]
                
                result = cohort.by_genotype(
                    chromosome_number=1,
                    genomic_position=12345,
                    ref_allele="A",
                    alt_allele="G",
                    case_gt="AG",
                    control_gt="AA"
                )
                  # Verify that Hail operations were called
                assert mock_read_mt.called
                assert mock_mt.filter_rows.called

    @pytest.mark.skipif(not HAIL_AVAILABLE, reason="Hail/Spark environment not properly configured")
    def test_by_genotype_custom_platform_no_mt_path(self):
        """Test error handling when custom platform has no mt_path."""
        cohort = Cohort(platform="custom", gbq_dataset_id="test_dataset")
        
        with patch('sys.exit') as mock_exit, \
             patch('builtins.print') as mock_print:
            mock_exit.side_effect = SystemExit(0)
            
            with pytest.raises(SystemExit) as exc_info:
                cohort.by_genotype(
                    chromosome_number=1,
                    genomic_position=12345,
                    ref_allele="A",
                    alt_allele="G",
                    case_gt="AG",
                    control_gt="AA"
                )
            
            assert exc_info.value.code == 0
            mock_exit.assert_called_once_with(0)
            mock_print.assert_called_once_with("For custom platform, mt_path must not be None.")

    @pytest.mark.skipif(not HAIL_AVAILABLE, reason="Hail/Spark environment not properly configured")
    @patch('polars.DataFrame.write_csv')
    def test_by_genotype_output_file_csv_extension(self, mock_write_csv):
        """Test output file handling with .csv extension."""
        cohort = Cohort(platform="custom", gbq_dataset_id="test_dataset")
        
        with patch('hail.init'), \
             patch('hail.read_matrix_table') as mock_read_mt, \
             patch('hail.Locus.parse') as mock_locus_parse, \
             patch('hail.parse_variant') as mock_parse_variant:
            
            mock_mt = Mock()
            mock_read_mt.return_value = mock_mt
            
            # Mock matrix table count_rows to return integer instead of Mock
            mock_mt.count_rows.return_value = 1
            
            # Mock Hail objects
            mock_locus_parse.return_value = "mock_locus"
            mock_variant = {"locus": "mock_locus", "alleles": ["A", "G"]}
            mock_parse_variant.return_value = mock_variant
            
            # Mock matrix table operations
            mock_filtered_mt = Mock()
            mock_filtered_mt.count_rows.return_value = 1
            mock_mt.filter_rows.return_value = mock_filtered_mt
            
            # Mock the second filter_rows call (for variant filtering)
            mock_filtered_mt2 = Mock()
            mock_filtered_mt2.count_rows.return_value = 1
            mock_filtered_mt.filter_rows.return_value = mock_filtered_mt2
              # Mock the entries and to_spark conversion
            mock_entries = Mock()
            mock_spark_df = Mock()
            mock_entries.to_spark.return_value = mock_spark_df
            mock_filtered_mt2.entries.return_value = mock_entries
            
            # Mock the row show method
            mock_mt.row.show = Mock()
            mock_filtered_mt.row.show = Mock()
            mock_filtered_mt2.row.show = Mock()
            
            # Mock _utils.spark_to_polars
            with patch('PheTK.Cohort._utils.spark_to_polars') as mock_spark_to_polars:
                # Mock the allele count check and GT data
                mock_spark_to_polars.side_effect = [
                    pl.DataFrame({"info.AF": [[0.5]]}),  # For allele count check
                    pl.DataFrame({  # For GT data
                        "s": [1, 2, 3],
                        "GT.alleles": [[0, 0], [0, 1], [1, 1]]
                    })
                ]
                
                cohort.by_genotype(
                    chromosome_number=1,
                    genomic_position=12345,
                    ref_allele="A",
                    alt_allele="G",
                    case_gt="AG",
                    control_gt="AA",
                    mt_path="test_mt_path",
                    output_file_name="test_output.csv"                )
                
                mock_write_csv.assert_called_once_with("test_output.csv")

    @pytest.mark.skipif(not HAIL_AVAILABLE, reason="Hail/Spark environment not properly configured")
    def test_by_genotype_multiple_genotypes(self):
        """Test by_genotype with multiple genotypes for case and control."""
        cohort = Cohort(platform="custom", gbq_dataset_id="test_dataset")
        
        with patch('hail.init'), \
             patch('hail.read_matrix_table') as mock_read_mt, \
             patch('hail.Locus.parse') as mock_locus_parse, \
             patch('hail.parse_variant') as mock_parse_variant:
            
            mock_mt = Mock()
            mock_read_mt.return_value = mock_mt
            
            # Mock matrix table count_rows to return integer instead of Mock
            mock_mt.count_rows.return_value = 1
            
            # Mock Hail objects
            mock_locus_parse.return_value = "mock_locus"
            mock_variant = {"locus": "mock_locus", "alleles": ["A", "G"]}
            mock_parse_variant.return_value = mock_variant
              # Mock matrix table operations
            mock_filtered_mt = Mock()
            mock_filtered_mt.count_rows.return_value = 1
            mock_mt.filter_rows.return_value = mock_filtered_mt
            
            # Mock the second filter_rows call (for variant filtering)
            mock_filtered_mt2 = Mock()
            mock_filtered_mt2.count_rows.return_value = 1
            mock_filtered_mt.filter_rows.return_value = mock_filtered_mt2
            
            # Mock the entries and to_spark conversion
            mock_entries = Mock()
            mock_spark_df = Mock()
            mock_entries.to_spark.return_value = mock_spark_df
            mock_filtered_mt2.entries.return_value = mock_entries
            
            # Mock the row show method
            mock_mt.row.show = Mock()
            mock_filtered_mt.row.show = Mock()
            mock_filtered_mt2.row.show = Mock()
            
            # Mock _utils.spark_to_polars
            with patch('PheTK.Cohort._utils.spark_to_polars') as mock_spark_to_polars:
                # Mock the allele count check and GT data
                mock_spark_to_polars.side_effect = [
                    pl.DataFrame({"info.AF": [[0.5]]}),  # For allele count check
                    pl.DataFrame({  # For GT data
                        "s": [1, 2, 3],
                        "GT.alleles": [[0, 1], [1, 1], [0, 0]]
                    })
                ]
                
                # Test that multiple genotypes are handled correctly
                result = cohort.by_genotype(
                    chromosome_number=1,
                    genomic_position=12345,
                    ref_allele="A",
                    alt_allele="G",
                    case_gt=["AG", "GG"],  # Multiple case genotypes
                    control_gt=["AA"],     # Single control genotype
                    mt_path="test_mt_path"
                )
                
                # Verify that the method completes without error
                assert mock_read_mt.called


class TestCohortCovariates:
    """Test Cohort covariate-related methods."""

    def test_covariate_attributes_modification(self):
        """Test that covariate attributes can be modified."""
        cohort = Cohort(platform="custom", gbq_dataset_id="test_dataset")
        
        # Modify attributes
        cohort.natural_age = True
        cohort.sex_at_birth = True
        cohort.first_n_pcs = 10
        
        # Verify changes
        assert cohort.natural_age is True
        assert cohort.sex_at_birth is True
        assert cohort.first_n_pcs == 10
        assert cohort.age_at_last_event is False  # Should remain unchanged


class TestCohortEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.skipif(not HAIL_AVAILABLE, reason="Hail/Spark environment not properly configured")
    def test_case_sensitive_genotype_formats(self):
        """Test that genotype formats are handled correctly."""
        cohort = Cohort(platform="custom", gbq_dataset_id="test_dataset")
        
        with patch('hail.init'), \
             patch('hail.read_matrix_table') as mock_read_mt, \
             patch('hail.Locus.parse') as mock_locus_parse, \
             patch('hail.parse_variant') as mock_parse_variant:
            
            # Mock setup
            mock_mt = Mock()
            mock_read_mt.return_value = mock_mt
            mock_mt.count_rows.return_value = 1
            
            mock_locus_parse.return_value = "mock_locus"
            mock_variant = {"locus": "mock_locus", "alleles": ["A", "G"]}
            mock_parse_variant.return_value = mock_variant
            
            mock_filtered_mt = Mock()
            mock_filtered_mt.count_rows.return_value = 1
            mock_mt.filter_rows.return_value = mock_filtered_mt
            
            # Mock the second filter_rows call (for variant filtering)
            mock_filtered_mt2 = Mock()
            mock_filtered_mt2.count_rows.return_value = 1
            mock_filtered_mt.filter_rows.return_value = mock_filtered_mt2
            
            mock_entries = Mock()
            mock_spark_df = Mock()
            mock_entries.to_spark.return_value = mock_spark_df
            mock_filtered_mt2.entries.return_value = mock_entries
            
            mock_mt.row.show = Mock()
            mock_filtered_mt.row.show = Mock()
            mock_filtered_mt2.row.show = Mock()
            
            # Test various genotype formats
            test_cases = [
                ("AA", "GG"),
                ("aa", "gg"),
                (["AA", "AG"], ["GG"]),
                ("0/0", "1/1")
            ]
            
            for case_gt, control_gt in test_cases:
                with patch('PheTK.Cohort._utils.spark_to_polars') as mock_spark_to_polars:
                    mock_spark_to_polars.side_effect = [
                        pl.DataFrame({"info.AF": [[0.5]]}),  # For allele count check
                        pl.DataFrame({  # For GT data
                            "s": [1, 2, 3],
                            "GT.alleles": [[0, 0], [0, 1], [1, 1]]
                        })
                    ]
                    
                    try:
                        cohort.by_genotype(
                            chromosome_number=1,
                            genomic_position=12345,
                            ref_allele="A",
                            alt_allele="G",
                            case_gt=case_gt,
                            control_gt=control_gt,
                            mt_path="test_mt_path"
                        )
                    except Exception as e:
                        # Should not raise exceptions for valid formats
                        pytest.fail(f"Unexpected exception for genotypes {case_gt}, {control_gt}: {e}")

    @pytest.mark.skipif(not HAIL_AVAILABLE, reason="Hail/Spark environment not properly configured")
    def test_reference_genome_validation(self):
        """Test reference genome parameter validation."""
        cohort = Cohort(platform="custom", gbq_dataset_id="test_dataset")
        
        with patch('hail.init'), \
             patch('hail.read_matrix_table') as mock_read_mt, \
             patch('hail.Locus.parse') as mock_locus_parse, \
             patch('hail.parse_variant') as mock_parse_variant:
            
            # Mock setup
            mock_mt = Mock()
            mock_read_mt.return_value = mock_mt
            mock_mt.count_rows.return_value = 1
            
            mock_locus_parse.return_value = "mock_locus"
            mock_variant = {"locus": "mock_locus", "alleles": ["A", "G"]}
            mock_parse_variant.return_value = mock_variant
            mock_filtered_mt = Mock()
            mock_filtered_mt.count_rows.return_value = 1
            mock_mt.filter_rows.return_value = mock_filtered_mt
            
            # Mock the second filter_rows call (for variant filtering)
            mock_filtered_mt2 = Mock()
            mock_filtered_mt2.count_rows.return_value = 1
            mock_filtered_mt.filter_rows.return_value = mock_filtered_mt2
            
            mock_entries = Mock()
            mock_spark_df = Mock()
            mock_entries.to_spark.return_value = mock_spark_df
            mock_filtered_mt2.entries.return_value = mock_entries
            
            mock_mt.row.show = Mock()
            mock_filtered_mt.row.show = Mock()
            mock_filtered_mt2.row.show = Mock()
            
            # Test valid reference genomes
            for ref_genome in ["GRCh37", "GRCh38"]:
                with patch('PheTK.Cohort._utils.spark_to_polars') as mock_spark_to_polars:
                    mock_spark_to_polars.side_effect = [
                        pl.DataFrame({"info.AF": [[0.5]]}),  # For allele count check
                        pl.DataFrame({  # For GT data
                            "s": [1, 2, 3],
                            "GT.alleles": [[0, 0], [0, 1], [1, 1]]
                        })
                    ]
                    
                    try:
                        cohort.by_genotype(
                            chromosome_number=1,
                            genomic_position=12345,
                            ref_allele="A",
                            alt_allele="G",
                            case_gt="AG",
                            control_gt="AA",
                            reference_genome=ref_genome,
                            mt_path="test_mt_path"
                        )
                    except Exception as e:
                        pytest.fail(f"Unexpected exception for reference genome {ref_genome}: {e}")


class TestCohortIntegration:
    """Integration tests for Cohort class."""

    @patch.dict(os.environ, {'WORKSPACE_CDR': 'test_cdr', 'GOOGLE_PROJECT': 'test_project'})
    def test_complete_workflow_aou_platform(self):
        """Test complete workflow with AoU platform."""
        cohort = Cohort(platform="aou", aou_db_version=8)
        
        # Verify initialization
        assert cohort.platform == "aou"
        assert cohort.db_version == 8
        assert cohort.cdr == "test_cdr"
        
        # Test attribute modification
        cohort.natural_age = True
        cohort.first_n_pcs = 5
        
        assert cohort.natural_age is True
        assert cohort.first_n_pcs == 5

    def test_complete_workflow_custom_platform(self):
        """Test complete workflow with custom platform."""
        cohort = Cohort(platform="custom", gbq_dataset_id="my_dataset")
        
        # Verify initialization
        assert cohort.platform == "custom"
        assert cohort.cdr == "my_dataset"
        
        # Test that all covariate flags start as False
        covariate_flags = [
            cohort.natural_age,
            cohort.age_at_last_event,
            cohort.sex_at_birth,
            cohort.ehr_length,
            cohort.dx_code_occurrence_count,
            cohort.dx_condition_count,
            cohort.genetic_ancestry
        ]
        
        assert all(flag is False for flag in covariate_flags)
        assert cohort.first_n_pcs == 0
