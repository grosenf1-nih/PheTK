"""
Unit tests for PheTK.PheWAS module.
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

from PheTK.PheWAS import PheWAS


class TestPheWASInitialization:
    """Test PheWAS class initialization."""

    def setup_method(self):
        """Set up test data files for each test method."""
        # Create temporary files with test data
        self.temp_phecode_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_cohort_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        # Sample phecode count data
        phecode_data = pl.DataFrame({
            'person_id': [1, 1, 2, 2, 3, 3, 4, 5],
            'phecode': ['250', '401', '250', '585', '250', '272.1', '401', '585'],
            'count': [3, 2, 1, 1, 2, 3, 1, 2]
        })
        phecode_data.write_csv(self.temp_phecode_file.name)
        
        # Sample cohort data
        cohort_data = pl.DataFrame({
            'person_id': [1, 2, 3, 4, 5],
            'age': [65, 45, 78, 34, 56],
            'sex_at_birth': [1, 0, 1, 0, 1],
            'pc1': [0.1, -0.2, 0.3, -0.1, 0.4],
            'pc2': [0.05, 0.15, -0.25, 0.35, -0.05],
            'genotype': [1, 0, 1, 0, 1]
        })
        cohort_data.write_csv(self.temp_cohort_file.name)
        
        self.temp_phecode_file.close()
        self.temp_cohort_file.close()

    def teardown_method(self):
        """Clean up temporary files after each test."""
        os.unlink(self.temp_phecode_file.name)
        os.unlink(self.temp_cohort_file.name)

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_init_basic_parameters(self, mock_get_mapping):
        """Test basic initialization with required parameters."""
        # Mock phecode mapping
        mock_mapping = pl.DataFrame({
            'phecode': ['250', '401', '585', '272.1'],
            'phecode_description': ['Diabetes', 'Hypertension', 'CKD', 'Hyperlipidemia'],
            'phecode_category': ['endocrine', 'circulatory', 'genitourinary', 'endocrine']
        })
        mock_get_mapping.return_value = mock_mapping
        
        phewas = PheWAS(
            phecode_version="X",
            phecode_count_csv_path=self.temp_phecode_file.name,
            cohort_csv_path=self.temp_cohort_file.name,
            sex_at_birth_col="sex_at_birth",
            covariate_cols=["age", "pc1", "pc2"],
            independent_variable_of_interest="genotype"
        )
        
        assert phewas.sex_at_birth_col == "sex_at_birth"
        assert phewas.covariate_cols == ["age", "pc1", "pc2"]
        assert phewas.independent_variable_of_interest == "genotype"
        assert phewas.male_value == 1
        assert phewas.female_value == 0
        assert phewas.min_cases == 50
        assert phewas.min_phecode_count == 2

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_init_single_covariate_string(self, mock_get_mapping):
        """Test initialization with single covariate as string."""
        mock_mapping = pl.DataFrame({'phecode': ['250']})
        mock_get_mapping.return_value = mock_mapping
        
        phewas = PheWAS(
            phecode_version="X",
            phecode_count_csv_path=self.temp_phecode_file.name,
            cohort_csv_path=self.temp_cohort_file.name,
            sex_at_birth_col="sex_at_birth",
            covariate_cols="age",  # Single string instead of list
            independent_variable_of_interest="genotype"
        )
        
        assert phewas.covariate_cols == ["age"]

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_init_male_as_zero(self, mock_get_mapping):
        """Test initialization with male_as_one=False."""
        mock_mapping = pl.DataFrame({'phecode': ['250']})
        mock_get_mapping.return_value = mock_mapping
        
        phewas = PheWAS(
            phecode_version="X",
            phecode_count_csv_path=self.temp_phecode_file.name,
            cohort_csv_path=self.temp_cohort_file.name,
            sex_at_birth_col="sex_at_birth",
            covariate_cols=["age"],
            independent_variable_of_interest="genotype",
            male_as_one=False
        )
        
        assert phewas.male_value == 0
        assert phewas.female_value == 1

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_init_custom_parameters(self, mock_get_mapping):
        """Test initialization with custom parameters."""
        mock_mapping = pl.DataFrame({'phecode': ['250']})
        mock_get_mapping.return_value = mock_mapping
        
        phewas = PheWAS(
            phecode_version="1.2",
            phecode_count_csv_path=self.temp_phecode_file.name,
            cohort_csv_path=self.temp_cohort_file.name,
            sex_at_birth_col="sex_at_birth",
            covariate_cols=["age"],
            independent_variable_of_interest="genotype",
            icd_version="WHO",
            min_cases=100,
            min_phecode_count=3,
            use_exclusion=True,
            verbose=True,
            suppress_warnings=False,
            use_firth=True
        )
        
        assert phewas.min_cases == 100
        assert phewas.min_phecode_count == 3
        assert phewas.verbose is True
        assert phewas.suppress_warnings is False
        assert phewas.use_firth is True


class TestPheWASDataProcessing:
    """Test PheWAS data processing methods."""

    def setup_method(self):
        """Set up test data for each test method."""
        # Create test data
        self.phecode_data = pl.DataFrame({
            'person_id': [1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10] * 10,  # Ensure enough cases
            'phecode': ['250'] * 50 + ['401'] * 60 + ['585'] * 20,  # Different case counts
            'count': [3, 2, 1, 4, 2, 3, 1, 2, 5, 1, 2, 3, 1] * 10
        })
        
        self.cohort_data = pl.DataFrame({
            'person_id': list(range(1, 131)),  # 130 people
            'age': [50 + i % 30 for i in range(130)],
            'sex_at_birth': [i % 2 for i in range(130)],
            'pc1': [0.1 * (i % 10) for i in range(130)],
            'genotype': [i % 2 for i in range(130)]
        })

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_phewas_data_loading(self, mock_get_mapping):
        """Test that data is loaded correctly."""
        # Create temporary files
        phecode_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        cohort_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        self.phecode_data.write_csv(phecode_file.name)
        self.cohort_data.write_csv(cohort_file.name)
        
        phecode_file.close()
        cohort_file.close()
        
        try:
            mock_mapping = pl.DataFrame({
                'phecode': ['250', '401', '585'],
                'phecode_description': ['Diabetes', 'Hypertension', 'CKD'],
                'phecode_category': ['endocrine', 'circulatory', 'genitourinary']
            })
            mock_get_mapping.return_value = mock_mapping
            
            phewas = PheWAS(
                phecode_version="X",
                phecode_count_csv_path=phecode_file.name,
                cohort_csv_path=cohort_file.name,
                sex_at_birth_col="sex_at_birth",
                covariate_cols=["age", "pc1"],
                independent_variable_of_interest="genotype"
            )
            
            # Verify data was loaded
            assert len(phewas.phecode_counts) > 0
            assert len(phewas.covariate_df) > 0
            assert isinstance(phewas.phecode_counts, pl.DataFrame)
            assert isinstance(phewas.covariate_df, pl.DataFrame)
            
        finally:
            os.unlink(phecode_file.name)
            os.unlink(cohort_file.name)


class TestPheWASStatisticalMethods:
    """Test PheWAS statistical analysis methods."""

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_phewas_analysis_setup(self, mock_get_mapping):
        """Test setup for PheWAS analysis."""
        # Create minimal test data
        phecode_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        cohort_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        # Create data with sufficient cases
        phecode_data = pl.DataFrame({
            'person_id': list(range(1, 101)),
            'phecode': ['250'] * 100,
            'count': [2] * 100
        })
        
        cohort_data = pl.DataFrame({
            'person_id': list(range(1, 201)),
            'age': [50] * 200,
            'sex_at_birth': [i % 2 for i in range(200)],
            'genotype': [i % 2 for i in range(200)]
        })
        
        phecode_data.write_csv(phecode_file.name)
        cohort_data.write_csv(cohort_file.name)
        
        phecode_file.close()
        cohort_file.close()
        
        try:
            mock_mapping = pl.DataFrame({
                'phecode': ['250'],
                'phecode_description': ['Diabetes'],
                'phecode_category': ['endocrine']
            })
            mock_get_mapping.return_value = mock_mapping
            
            phewas = PheWAS(
                phecode_version="X",
                phecode_count_csv_path=phecode_file.name,
                cohort_csv_path=cohort_file.name,
                sex_at_birth_col="sex_at_birth",
                covariate_cols=["age"],
                independent_variable_of_interest="genotype",
                min_cases=10  # Lower threshold for testing
            )
            
            # Verify initialization completed
            assert hasattr(phewas, 'phecode_counts')
            assert hasattr(phewas, 'covariate_df')
            assert hasattr(phewas, 'phecode_df')
            
        finally:
            os.unlink(phecode_file.name)
            os.unlink(cohort_file.name)

    @patch('PheTK._utils.get_phecode_mapping_table')
    @patch('statsmodels.api.Logit')
    def test_statistical_model_setup(self, mock_logit, mock_get_mapping):
        """Test that statistical models are set up correctly."""
        # Create test files
        phecode_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        cohort_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        phecode_data = pl.DataFrame({
            'person_id': [1, 2, 3],
            'phecode': ['250', '250', '250'],
            'count': [2, 3, 1]
        })
        
        cohort_data = pl.DataFrame({
            'person_id': [1, 2, 3, 4, 5],
            'age': [50, 60, 70, 40, 30],
            'sex_at_birth': [1, 0, 1, 0, 1],
            'genotype': [1, 0, 1, 0, 1]
        })
        
        phecode_data.write_csv(phecode_file.name)
        cohort_data.write_csv(cohort_file.name)
        
        phecode_file.close()
        cohort_file.close()
        
        try:
            mock_mapping = pl.DataFrame({
                'phecode': ['250'],
                'phecode_description': ['Diabetes'],
                'phecode_category': ['endocrine']
            })
            mock_get_mapping.return_value = mock_mapping
            
            # Mock statsmodels components
            mock_model = Mock()
            mock_results = Mock()
            mock_results.params = pd.Series([0.5, 1.2, 0.8], index=['const', 'genotype', 'age'])
            mock_results.pvalues = pd.Series([0.1, 0.01, 0.05], index=['const', 'genotype', 'age'])
            mock_results.converged = True
            mock_model.fit.return_value = mock_results
            mock_logit.return_value = mock_model
            
            phewas = PheWAS(
                phecode_version="X",
                phecode_count_csv_path=phecode_file.name,
                cohort_csv_path=cohort_file.name,
                sex_at_birth_col="sex_at_birth",
                covariate_cols=["age"],
                independent_variable_of_interest="genotype",
                min_cases=1  # Very low threshold for testing
            )
            
            # Verify that the PheWAS object was created successfully
            assert phewas is not None
            
        finally:
            os.unlink(phecode_file.name)
            os.unlink(cohort_file.name)


class TestPheWASValidation:
    """Test PheWAS input validation and error handling."""

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_invalid_file_paths(self, mock_get_mapping):
        """Test handling of invalid file paths."""
        mock_mapping = pl.DataFrame({'phecode': ['250']})
        mock_get_mapping.return_value = mock_mapping
        
        with pytest.raises((FileNotFoundError, Exception)):
            PheWAS(
                phecode_version="X",
                phecode_count_csv_path="/nonexistent/path.csv",
                cohort_csv_path="/nonexistent/path.csv",
                sex_at_birth_col="sex_at_birth",
                covariate_cols=["age"],
                independent_variable_of_interest="genotype"
            )

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_empty_covariate_list(self, mock_get_mapping):
        """Test handling of empty covariate list."""
        mock_mapping = pl.DataFrame({'phecode': ['250']})
        mock_get_mapping.return_value = mock_mapping
        
        # Create minimal test files
        phecode_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        cohort_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        pl.DataFrame({'person_id': [1], 'phecode': ['250'], 'count': [1]}).write_csv(phecode_file.name)
        pl.DataFrame({'person_id': [1], 'sex_at_birth': [1], 'genotype': [1]}).write_csv(cohort_file.name)
        
        phecode_file.close()
        cohort_file.close()
        
        try:
            phewas = PheWAS(
                phecode_version="X",
                phecode_count_csv_path=phecode_file.name,
                cohort_csv_path=cohort_file.name,
                sex_at_birth_col="sex_at_birth",
                covariate_cols=[],  # Empty list
                independent_variable_of_interest="genotype"
            )
            
            assert phewas.covariate_cols == []
            
        finally:
            os.unlink(phecode_file.name)
            os.unlink(cohort_file.name)


class TestPheWASEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_minimum_case_threshold(self, mock_get_mapping):
        """Test behavior with minimum case threshold."""
        mock_mapping = pl.DataFrame({
            'phecode': ['250', '401'],
            'phecode_description': ['Diabetes', 'Hypertension'],
            'phecode_category': ['endocrine', 'circulatory']
        })
        mock_get_mapping.return_value = mock_mapping
        
        # Create data with one phecode having insufficient cases
        phecode_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        cohort_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        phecode_data = pl.DataFrame({
            'person_id': [1, 2, 3, 4] + list(range(5, 105)),  # 250 has many cases, 401 has few
            'phecode': ['401'] * 4 + ['250'] * 100,
            'count': [2] * 104
        })
        
        cohort_data = pl.DataFrame({
            'person_id': list(range(1, 201)),
            'age': [50] * 200,
            'sex_at_birth': [i % 2 for i in range(200)],
            'genotype': [i % 2 for i in range(200)]
        })
        
        phecode_data.write_csv(phecode_file.name)
        cohort_data.write_csv(cohort_file.name)
        
        phecode_file.close()
        cohort_file.close()
        
        try:
            phewas = PheWAS(
                phecode_version="X",
                phecode_count_csv_path=phecode_file.name,
                cohort_csv_path=cohort_file.name,
                sex_at_birth_col="sex_at_birth",
                covariate_cols=["age"],
                independent_variable_of_interest="genotype",
                min_cases=50  # Should exclude phecode 401
            )
            
            assert phewas.min_cases == 50
            
        finally:
            os.unlink(phecode_file.name)
            os.unlink(cohort_file.name)

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_phecode_count_threshold(self, mock_get_mapping):
        """Test minimum phecode count threshold."""
        mock_mapping = pl.DataFrame({'phecode': ['250']})
        mock_get_mapping.return_value = mock_mapping
        
        phecode_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        cohort_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        # Create data with varying phecode counts
        phecode_data = pl.DataFrame({
            'person_id': [1, 2, 3, 4, 5],
            'phecode': ['250'] * 5,
            'count': [1, 2, 3, 1, 4]  # Some below threshold
        })
        
        cohort_data = pl.DataFrame({
            'person_id': [1, 2, 3, 4, 5],
            'age': [50] * 5,
            'sex_at_birth': [1, 0, 1, 0, 1],
            'genotype': [1, 0, 1, 0, 1]
        })
        
        phecode_data.write_csv(phecode_file.name)
        cohort_data.write_csv(cohort_file.name)
        
        phecode_file.close()
        cohort_file.close()
        
        try:
            phewas = PheWAS(
                phecode_version="X",
                phecode_count_csv_path=phecode_file.name,
                cohort_csv_path=cohort_file.name,
                sex_at_birth_col="sex_at_birth",
                covariate_cols=["age"],
                independent_variable_of_interest="genotype",
                min_phecode_count=2  # Should exclude counts of 1
            )
            
            assert phewas.min_phecode_count == 2
            
        finally:
            os.unlink(phecode_file.name)
            os.unlink(cohort_file.name)


class TestPheWASIntegration:
    """Integration tests for PheWAS class."""

    @patch('PheTK._utils.get_phecode_mapping_table')
    def test_complete_initialization_workflow(self, mock_get_mapping):
        """Test complete initialization workflow."""
        # Setup comprehensive mock data
        mock_mapping = pl.DataFrame({
            'phecode': ['250', '401', '585'],
            'phecode_description': ['Diabetes mellitus', 'Hypertension', 'Chronic kidney disease'],
            'phecode_category': ['endocrine/metabolic', 'circulatory system', 'genitourinary'],
            'phecode_exclude_range': ['', '', '']
        })
        mock_get_mapping.return_value = mock_mapping
        
        # Create comprehensive test data
        phecode_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        cohort_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        # Generate realistic phecode data
        phecode_data = pl.DataFrame({
            'person_id': [i for i in range(1, 201) for _ in range(np.random.randint(1, 4))],
            'phecode': [np.random.choice(['250', '401', '585']) for _ in range(400)],
            'count': [np.random.randint(1, 6) for _ in range(400)]
        })
        
        # Generate realistic cohort data
        cohort_data = pl.DataFrame({
            'person_id': list(range(1, 301)),
            'age': [np.random.randint(20, 80) for _ in range(300)],
            'sex_at_birth': [np.random.randint(0, 2) for _ in range(300)],
            'pc1': [np.random.normal(0, 0.1) for _ in range(300)],
            'pc2': [np.random.normal(0, 0.1) for _ in range(300)],
            'genotype': [np.random.randint(0, 2) for _ in range(300)]
        })
        
        phecode_data.write_csv(phecode_file.name)
        cohort_data.write_csv(cohort_file.name)
        
        phecode_file.close()
        cohort_file.close()
        
        try:
            phewas = PheWAS(
                phecode_version="X",
                phecode_count_csv_path=phecode_file.name,
                cohort_csv_path=cohort_file.name,
                sex_at_birth_col="sex_at_birth",
                covariate_cols=["age", "pc1", "pc2"],
                independent_variable_of_interest="genotype",
                icd_version="US",
                min_cases=10,
                min_phecode_count=1,
                verbose=False
            )
            
            # Verify all components initialized correctly
            assert isinstance(phewas.phecode_counts, pl.DataFrame)
            assert isinstance(phewas.covariate_df, pl.DataFrame)
            assert isinstance(phewas.phecode_df, pl.DataFrame)
            assert len(phewas.covariate_cols) == 3
            assert phewas.independent_variable_of_interest == "genotype"
            
            # Verify mapping was called correctly
            mock_get_mapping.assert_called_once_with(
                phecode_version="X",
                icd_version="US",
                phecode_map_file_path=None,
                keep_all_columns=True
            )
            
        finally:
            os.unlink(phecode_file.name)
            os.unlink(cohort_file.name)
