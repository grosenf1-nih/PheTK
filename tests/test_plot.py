"""
Unit tests for PheTK.Plot module.
"""
import pytest
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PheTK.Plot import Plot


class TestPlotInitialization:
    """Test Plot class initialization."""

    def setup_method(self):
        """Set up test data for each test method."""
        # Create sample PheWAS results
        self.sample_phewas_data = pl.DataFrame({
            'phecode': ['250', '401', '585', '272.1', '428', '715.1'],
            'phecode_description': ['Diabetes mellitus', 'Hypertension', 'Chronic kidney disease', 
                                  'Hyperlipidemia', 'Heart failure', 'Osteoarthritis'],
            'phecode_category': ['endocrine/metabolic', 'circulatory system', 'genitourinary',
                               'endocrine/metabolic', 'circulatory system', 'musculoskeletal'],
            'case_count': [120, 200, 80, 150, 90, 110],
            'control_count': [1000, 900, 1100, 950, 1050, 980],
            'odds_ratio': [2.5, 1.8, 3.2, 1.6, 2.1, 1.4],
            'p_value': [0.001, 0.01, 0.0001, 0.05, 0.005, 0.1],
            'neg_log_p_value': [3.0, 2.0, 4.0, 1.3, 2.3, 1.0],
            'converged': [True, True, True, True, True, False],
            'beta': [0.916, 0.588, 1.163, 0.470, 0.742, 0.336]
        })
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_phewas_data.write_csv(self.temp_file.name)
        self.temp_file.close()

    def teardown_method(self):
        """Clean up temporary files after each test."""
        os.unlink(self.temp_file.name)

    def test_init_basic_parameters(self):
        """Test basic initialization with default parameters."""
        plot = Plot(phewas_result_csv_path=self.temp_file.name)
        
        assert isinstance(plot.phewas_result, pl.DataFrame)
        assert len(plot.phewas_result) == 6  # All results loaded
        assert plot.bonferroni > 0  # Bonferroni calculated
        assert plot.nominal_significance == -np.log10(0.05)
        assert plot.phecode_version == "X"  # Default    def test_init_converged_only_true(self):
        """Test initialization with converged_only=True."""
        plot = Plot(phewas_result_csv_path=self.temp_file.name, converged_only=True)
        
        # Should exclude non-converged results
        assert len(plot.phewas_result) == 5  # Excludes the non-converged one
        converged_values = plot.phewas_result['converged'].to_list()
        assert all(converged_values) if converged_values else True

    def test_init_converged_only_false(self):
        """Test initialization with converged_only=False."""
        plot = Plot(phewas_result_csv_path=self.temp_file.name, converged_only=False)
        
        # Should include all results
        assert len(plot.phewas_result) == 6  # Includes all results

    def test_init_custom_bonferroni(self):
        """Test initialization with custom Bonferroni threshold."""
        custom_bonferroni = 4.0
        plot = Plot(phewas_result_csv_path=self.temp_file.name, bonferroni=custom_bonferroni)
        
        assert plot.bonferroni == custom_bonferroni

    def test_init_phecode_version_12(self):
        """Test initialization with phecode version 1.2."""
        plot = Plot(phewas_result_csv_path=self.temp_file.name, phecode_version="1.2")
        
        assert plot.phecode_version == "1.2"

    def test_init_custom_color_palette(self):
        """Test initialization with custom color palette."""
        custom_colors = ("red", "blue", "green", "yellow")
        plot = Plot(phewas_result_csv_path=self.temp_file.name, color_palette=custom_colors)
        
        assert plot.color_palette == custom_colors

    def test_init_infinity_neg_log_p_handling(self):
        """Test handling of infinity values in neg_log_p_value."""
        # Create data with infinity values
        data_with_inf = self.sample_phewas_data.clone()
        data_with_inf = data_with_inf.with_columns(
            pl.when(pl.col('phecode') == '250')
            .then(float('inf'))
            .otherwise(pl.col('neg_log_p_value'))
            .alias('neg_log_p_value')
        )
        
        inf_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        data_with_inf.write_csv(inf_file.name)
        inf_file.close()
        
        try:
            plot = Plot(phewas_result_csv_path=inf_file.name)
            
            # Should have proxy value for infinity
            assert plot.inf_proxy is not None
            assert not np.isinf(plot.phewas_result['neg_log_p_value'].max())
            
        finally:
            os.unlink(inf_file.name)

    def test_init_color_mapping(self):
        """Test that color mapping is created correctly."""
        plot = Plot(phewas_result_csv_path=self.temp_file.name)
        
        # Should have color dictionary
        assert hasattr(plot, 'color_dict')
        assert isinstance(plot.color_dict, dict)
        
        # Should have label_color column added
        assert 'label_color' in plot.phewas_result.columns
        
        # Check categories are sorted
        assert plot.phecode_categories is not None
        assert len(plot.phecode_categories) == len(plot.phewas_result['phecode_category'].unique())

    def test_init_default_attributes(self):
        """Test that default attributes are set correctly."""
        plot = Plot(phewas_result_csv_path=self.temp_file.name)
        
        assert plot.positive_betas is None
        assert plot.negative_betas is None
        assert plot.data_to_label is None
        assert plot.ratio == 1
        assert plot.positive_alpha == 0.7
        assert plot.negative_alpha == 0.3
        assert plot.offset == 9
        assert plot.volcano_label_data is None


class TestPlotStaticMethods:
    """Test Plot static methods."""

    @patch('matplotlib.pyplot.savefig')
    @patch('datetime.datetime')
    def test_save_plot_default_name(self, mock_datetime, mock_savefig):
        """Test save_plot with default filename."""
        mock_datetime.now.return_value.strftime.return_value = "20250623_120000"
        
        Plot.save_plot()
        
        mock_savefig.assert_called_once_with("plot_20250623_120000.pdf", bbox_inches="tight")

    @patch('matplotlib.pyplot.savefig')
    def test_save_plot_custom_name_with_extension(self, mock_savefig):
        """Test save_plot with custom filename including extension."""
        Plot.save_plot(output_file_name="my_plot.png")
        
        mock_savefig.assert_called_once_with("my_plot.png", bbox_inches="tight")

    @patch('matplotlib.pyplot.savefig')
    def test_save_plot_custom_name_without_extension(self, mock_savefig):
        """Test save_plot with custom filename without extension."""
        Plot.save_plot(output_file_name="my_plot", output_file_type="svg")
        
        mock_savefig.assert_called_once_with("my_plot.svg", bbox_inches="tight")

    @patch('matplotlib.pyplot.savefig')
    @patch('datetime.datetime')
    def test_save_plot_custom_type_default_name(self, mock_datetime, mock_savefig):
        """Test save_plot with custom file type and default name."""
        mock_datetime.now.return_value.strftime.return_value = "20250623_120000"
        
        Plot.save_plot(plot_type="manhattan", output_file_type="png")
        
        mock_savefig.assert_called_once_with("manhattan_20250623_120000.png", bbox_inches="tight")


class TestPlotDataProcessing:
    """Test Plot data processing methods."""

    def setup_method(self):
        """Set up test data for each test method."""
        self.sample_data = pl.DataFrame({
            'phecode': ['250', '401', '585'],
            'phecode_category': ['endocrine/metabolic', 'circulatory system', 'genitourinary'],
            'neg_log_p_value': [3.0, 2.0, 4.0],
            'converged': [True, True, True],
            'beta': [0.5, -0.3, 0.8]
        })
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_data.write_csv(self.temp_file.name)
        self.temp_file.close()

    def teardown_method(self):
        """Clean up temporary files after each test."""
        os.unlink(self.temp_file.name)

    def test_bonferroni_calculation(self):
        """Test automatic Bonferroni correction calculation."""
        plot = Plot(phewas_result_csv_path=self.temp_file.name)
        
        expected_bonferroni = -np.log10(0.05 / len(self.sample_data))
        assert np.isclose(plot.bonferroni, expected_bonferroni)

    def test_phecode_categories_sorting(self):
        """Test that phecode categories are sorted correctly."""
        plot = Plot(phewas_result_csv_path=self.temp_file.name)
        
        categories = plot.phecode_categories
        assert categories == sorted(categories)

    def test_color_assignment_consistency(self):
        """Test that color assignment is consistent."""
        plot = Plot(phewas_result_csv_path=self.temp_file.name)
        
        # Colors should be assigned consistently to categories
        for category in plot.phecode_categories:
            assert category in plot.color_dict
            
        # Label colors should match category colors
        for i, row in enumerate(plot.phewas_result.iter_rows(named=True)):
            expected_color = plot.color_dict[row['phecode_category']]
            assert row['label_color'] == expected_color


class TestPlotValidation:
    """Test Plot input validation and error handling."""

    def test_invalid_file_path(self):
        """Test handling of invalid file paths."""
        with pytest.raises((FileNotFoundError, Exception)):
            Plot(phewas_result_csv_path="/nonexistent/path.csv")

    def test_empty_phewas_results(self):
        """Test handling of empty PheWAS results."""
        empty_data = pl.DataFrame({
            'phecode': pl.Series([], dtype=pl.Utf8),
            'phecode_category': pl.Series([], dtype=pl.Utf8),
            'neg_log_p_value': pl.Series([], dtype=pl.Float64),
            'converged': pl.Series([], dtype=pl.Boolean),
            'beta': pl.Series([], dtype=pl.Float64)
        })
        
        empty_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        empty_data.write_csv(empty_file.name)
        empty_file.close()
        
        try:
            plot = Plot(phewas_result_csv_path=empty_file.name)
            
            # Should handle empty data gracefully
            assert len(plot.phewas_result) == 0
            assert plot.bonferroni == float('inf')  # 0.05 / 0 division
            
        finally:
            os.unlink(empty_file.name)

    def test_missing_required_columns(self):
        """Test handling of missing required columns."""
        incomplete_data = pl.DataFrame({
            'phecode': ['250', '401'],
            'neg_log_p_value': [3.0, 2.0]
            # Missing required columns like 'converged'
        })
        
        incomplete_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        incomplete_data.write_csv(incomplete_file.name)
        incomplete_file.close()
        
        try:
            # Should raise an error or handle gracefully
            with pytest.raises((KeyError, Exception)):
                Plot(phewas_result_csv_path=incomplete_file.name)
                
        finally:
            os.unlink(incomplete_file.name)


class TestPlotEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_phecode_category(self):
        """Test with data containing only one phecode category."""
        single_category_data = pl.DataFrame({
            'phecode': ['250.1', '250.2', '250.3'],
            'phecode_category': ['endocrine/metabolic', 'endocrine/metabolic', 'endocrine/metabolic'],
            'neg_log_p_value': [3.0, 2.0, 4.0],
            'converged': [True, True, True],
            'beta': [0.5, -0.3, 0.8]
        })
        
        single_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        single_category_data.write_csv(single_file.name)
        single_file.close()
        
        try:
            plot = Plot(phewas_result_csv_path=single_file.name)
            
            assert len(plot.phecode_categories) == 1
            assert plot.phecode_categories[0] == 'endocrine/metabolic'
            assert len(plot.color_dict) == 1
            
        finally:
            os.unlink(single_file.name)

    def test_more_categories_than_colors(self):
        """Test with more phecode categories than available colors."""
        # Create data with many categories
        many_categories = [f'category_{i}' for i in range(20)]  # More than default color palette
        many_category_data = pl.DataFrame({
            'phecode': [f'{i}' for i in range(20)],
            'phecode_category': many_categories,
            'neg_log_p_value': [2.0] * 20,
            'converged': [True] * 20,
            'beta': [0.5] * 20
        })
        
        many_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        many_category_data.write_csv(many_file.name)
        many_file.close()
        
        try:
            plot = Plot(phewas_result_csv_path=many_file.name)
            
            # Should handle color cycling
            assert len(plot.phecode_categories) == 20
            assert len(plot.color_dict) == 20
            
            # Colors should be assigned (with cycling)
            for category in plot.phecode_categories:
                assert category in plot.color_dict
                
        finally:
            os.unlink(many_file.name)

    def test_all_non_converged_results(self):
        """Test with all non-converged results when converged_only=True."""
        non_converged_data = pl.DataFrame({
            'phecode': ['250', '401', '585'],
            'phecode_category': ['endocrine/metabolic', 'circulatory system', 'genitourinary'],
            'neg_log_p_value': [3.0, 2.0, 4.0],
            'converged': [False, False, False],
            'beta': [0.5, -0.3, 0.8]
        })
        
        non_converged_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        non_converged_data.write_csv(non_converged_file.name)
        non_converged_file.close()
        
        try:
            plot = Plot(phewas_result_csv_path=non_converged_file.name, converged_only=True)
            
            # Should result in empty DataFrame after filtering
            assert len(plot.phewas_result) == 0
            
        finally:
            os.unlink(non_converged_file.name)


class TestPlotIntegration:
    """Integration tests for Plot class."""

    def test_complete_initialization_workflow(self):
        """Test complete initialization workflow with realistic data."""
        # Create comprehensive PheWAS results
        realistic_data = pl.DataFrame({
            'phecode': [f'{i}' for i in range(100, 200)],
            'phecode_description': [f'Disease {i}' for i in range(100, 200)],
            'phecode_category': [
                'endocrine/metabolic' if i % 4 == 0 else
                'circulatory system' if i % 4 == 1 else
                'genitourinary' if i % 4 == 2 else
                'musculoskeletal' 
                for i in range(100)
            ],
            'case_count': [np.random.randint(50, 200) for _ in range(100)],
            'control_count': [np.random.randint(800, 1200) for _ in range(100)],
            'odds_ratio': [np.random.uniform(0.5, 3.0) for _ in range(100)],
            'p_value': [np.random.uniform(0.0001, 0.2) for _ in range(100)],
            'neg_log_p_value': [np.random.uniform(0.7, 4.0) for _ in range(100)],
            'converged': [np.random.choice([True, False], p=[0.9, 0.1]) for _ in range(100)],
            'beta': [np.random.normal(0, 0.5) for _ in range(100)]
        })
        
        realistic_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        realistic_data.write_csv(realistic_file.name)
        realistic_file.close()
        
        try:
            plot = Plot(
                phewas_result_csv_path=realistic_file.name,
                converged_only=True,
                phecode_version="X"
            )
            
            # Verify all components initialized correctly
            assert isinstance(plot.phewas_result, pl.DataFrame)
            assert len(plot.phewas_result) > 0
            assert plot.bonferroni > 0
            assert len(plot.phecode_categories) == 4  # Four categories in test data
            assert len(plot.color_dict) == 4
            assert 'label_color' in plot.phewas_result.columns
              # Verify data quality
            assert all(plot.phewas_result['converged'].to_list())  # Only converged results (all should be True)
            assert not any(np.isinf(plot.phewas_result['neg_log_p_value'].to_list()))  # No infinity values
            
        finally:
            os.unlink(realistic_file.name)

    def test_phecode_version_consistency(self):
        """Test that phecode version is handled consistently."""
        sample_data = pl.DataFrame({
            'phecode': ['250', '401'],
            'phecode_category': ['endocrine/metabolic', 'circulatory system'],
            'neg_log_p_value': [3.0, 2.0],
            'converged': [True, True],
            'beta': [0.5, -0.3]
        })
        
        test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        sample_data.write_csv(test_file.name)
        test_file.close()
        
        try:
            # Test version X
            plot_x = Plot(phewas_result_csv_path=test_file.name, phecode_version="x")
            assert plot_x.phecode_version == "X"
            
            # Test version 1.2
            plot_12 = Plot(phewas_result_csv_path=test_file.name, phecode_version="1.2")
            assert plot_12.phecode_version == "1.2"
            
            # Test None (default)
            plot_default = Plot(phewas_result_csv_path=test_file.name, phecode_version=None)
            assert plot_default.phecode_version == "X"
            
        finally:
            os.unlink(test_file.name)
