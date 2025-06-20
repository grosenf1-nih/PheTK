"""PheWAS (Phenome-Wide Association Study) analysis module.

This module provides functionality for conducting phenome-wide association studies
using phecode mappings and logistic regression analysis.
"""

import argparse
import copy
import logging
import os
import sys
import types
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import statsmodels.tools.sm_exceptions
from tqdm import tqdm

from PheTK import _utils

# Constants
SUPPORTED_PHECODE_VERSIONS = ["1.2", "X"]
SUPPORTED_ICD_VERSIONS = ["US", "WHO", "custom"]
DEFAULT_MIN_CASES = 50
DEFAULT_MIN_PHECODE_COUNT = 2
DEFAULT_BONFERRONI_ALPHA = 0.05
MALE_VALUE = 1
FEMALE_VALUE = 0

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Custom Exceptions
class PheWASError(Exception):
    """Base exception class for PheWAS-related errors."""
    pass


class InvalidPhecodeVersionError(PheWASError):
    """Raised when an invalid phecode version is provided."""
    pass


class InvalidDataError(PheWASError):
    """Raised when input data is invalid or malformed."""
    pass


class InsufficientDataError(PheWASError):
    """Raised when there is insufficient data for analysis."""
    pass


@dataclass
class RegressionResult:
    """Data class to hold regression results."""
    phecode: str
    cases: int
    controls: int
    p_value: float
    neg_log_p_value: float
    beta: float
    conf_int_1: float
    conf_int_2: float
    odds_ratio: float
    log10_odds_ratio: float
    converged: bool


@dataclass
class PheWASConfig:
    """Configuration class for PheWAS analysis."""
    phecode_version: str
    phecode_count_csv_path: str
    cohort_csv_path: str
    sex_at_birth_col: str
    covariate_cols: Union[str, List[str]]
    independent_variable_of_interest: str
    male_as_one: bool = True
    icd_version: str = "US"
    phecode_map_file_path: Optional[str] = None
    phecode_to_process: Union[str, List[str]] = "all"
    min_cases: int = DEFAULT_MIN_CASES
    min_phecode_count: int = DEFAULT_MIN_PHECODE_COUNT
    use_exclusion: bool = False
    output_file_name: Optional[str] = None
    verbose: bool = False
    suppress_warnings: bool = True
    use_firth: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.phecode_version not in SUPPORTED_PHECODE_VERSIONS:
            raise InvalidPhecodeVersionError(
                f"Phecode version must be one of {SUPPORTED_PHECODE_VERSIONS}, "
                f"got '{self.phecode_version}'"
            )
        
        if self.icd_version not in SUPPORTED_ICD_VERSIONS:
            raise InvalidDataError(
                f"ICD version must be one of {SUPPORTED_ICD_VERSIONS}, "
                f"got '{self.icd_version}'"
            )
        
        # Ensure covariate_cols is a list
        if isinstance(self.covariate_cols, str):
            self.covariate_cols = [self.covariate_cols]


class PheWAS:
    """Phenome-Wide Association Study (PheWAS) analysis class.
    
    This class performs phenome-wide association studies using phecode mappings
    and logistic regression analysis to identify associations between genetic
    variants or other variables of interest and phenotypes.
    """

    def __init__(self, config_or_phecode_version=None, **kwargs):
        """Initialize PheWAS analysis object.
        
        Args:
            config_or_phecode_version: Either a PheWASConfig object (new approach) 
                                     or phecode_version string (legacy approach).
            **kwargs: Legacy parameters for backward compatibility.
            
        Raises:
            InvalidDataError: If input data is invalid or malformed.
            InsufficientDataError: If there is insufficient data for analysis.
        """
        logger.info("Creating PheWAS Object")
        
        # Handle backward compatibility
        if isinstance(config_or_phecode_version, PheWASConfig):
            # New approach: config object provided
            self.config = config_or_phecode_version
        elif isinstance(config_or_phecode_version, str) or config_or_phecode_version is None:
            # Legacy approach: individual parameters provided
            if config_or_phecode_version is not None:
                kwargs['phecode_version'] = config_or_phecode_version
            
            # Create config from legacy parameters
            self.config = self._create_config_from_legacy_params(**kwargs)
        else:
            raise InvalidDataError(
                "First argument must be either a PheWASConfig object or phecode_version string"
            )
        
        self._initialize_data_attributes()
        self._load_data()
        self._setup_sex_encoding()
        self._setup_exclusion_rules()
        self._process_covariates()
        self._validate_data()
        self._setup_output_filename()
        
        logger.info("PheWAS object created successfully")

    def _create_config_from_legacy_params(self, **kwargs) -> PheWASConfig:
        """Create PheWASConfig from legacy parameters for backward compatibility."""
        # Map legacy parameter names to config parameter names
        legacy_mapping = {
            'phecode_count_csv_path': 'phecode_count_csv_path',
            'cohort_csv_path': 'cohort_csv_path', 
            'sex_at_birth_col': 'sex_at_birth_col',
            'covariate_cols': 'covariate_cols',
            'independent_variable_of_interest': 'independent_variable_of_interest',
            'male_as_one': 'male_as_one',
            'icd_version': 'icd_version',
            'phecode_map_file_path': 'phecode_map_file_path',
            'phecode_to_process': 'phecode_to_process',
            'min_cases': 'min_cases',
            'min_phecode_count': 'min_phecode_count',
            'use_exclusion': 'use_exclusion',
            'output_file_name': 'output_file_name',
            'verbose': 'verbose',
            'suppress_warnings': 'suppress_warnings',
            'use_firth': 'use_firth'
        }
        
        # Extract and validate required parameters
        required_params = ['phecode_version', 'phecode_count_csv_path', 'cohort_csv_path', 
                          'sex_at_birth_col', 'covariate_cols', 'independent_variable_of_interest']
        
        config_params = {}
        for param in required_params:
            if param not in kwargs:
                raise InvalidDataError(f"Required parameter '{param}' is missing")
            config_params[param] = kwargs[param]
        
        # Add optional parameters with defaults
        for legacy_param, config_param in legacy_mapping.items():
            if legacy_param in kwargs and legacy_param not in required_params:
                config_params[config_param] = kwargs[legacy_param]
        
        return PheWASConfig(**config_params)

    def _initialize_data_attributes(self) -> None:
        """Initialize all data attributes."""
        # Core data
        self.phecode_df: Optional[pl.DataFrame] = None
        self.phecode_counts: Optional[pl.DataFrame] = None
        self.covariate_df: Optional[pl.DataFrame] = None
        
        # Analysis configuration
        self.covariate_cols: List[str] = copy.deepcopy(self.config.covariate_cols)
        self.sex_at_birth_col: str = self.config.sex_at_birth_col
        self.independent_variable_of_interest: str = self.config.independent_variable_of_interest
        
        # Sex encoding
        self.male_value: int = MALE_VALUE if self.config.male_as_one else FEMALE_VALUE
        self.female_value: int = FEMALE_VALUE if self.config.male_as_one else MALE_VALUE
        
        # Cohort information
        self.cohort_size: int = 0
        self.cohort_ids: List[str] = []
        self.phecode_list: List[str] = []
        
        # Sex-related flags
        self.data_has_single_sex: bool = False
        self.sex_as_covariate: bool = False
        self.variable_of_interest_in_covariates: bool = False
        self.single_sex_value: Optional[int] = None
        self.sex_values: List[int] = []
        
        # Variable columns for analysis
        self.var_cols: List[str] = []
        self.gender_specific_var_cols: List[str] = []
        
        # Results
        self.results: Optional[pl.DataFrame] = None
        self.tested_count: int = 0
        self.not_tested_count: int = 0
        self.bonferroni: Optional[float] = None
        self.phecodes_above_bonferroni: Optional[pl.DataFrame] = None
        self.above_bonferroni_count: int = 0
        
        # Output
        self.output_file_name: str = ""

    def _load_data(self) -> None:
        """Load phecode mapping, phecode counts, and covariate data."""
        logger.info("Loading data files")

        try:
            # Load phecode mapping file
            self.phecode_df = _utils.get_phecode_mapping_table(
                phecode_version=self.config.phecode_version,
                icd_version=self.config.icd_version,
                phecode_map_file_path=self.config.phecode_map_file_path,
                keep_all_columns=True
            )
            logger.info(f"Loaded phecode mapping table with {len(self.phecode_df)} entries")
            
            # Load phecode counts data
            self.phecode_counts = pl.read_csv(
                self.config.phecode_count_csv_path,
                dtypes={"phecode": str}
            )
            logger.info(f"Loaded phecode counts for {self.phecode_counts['person_id'].n_unique()} participants")
            
            # Load covariate data
            self.covariate_df = pl.read_csv(self.config.cohort_csv_path)
            logger.info(f"Loaded covariate data for {len(self.covariate_df)} participants")
            
        except Exception as e:
            raise InvalidDataError(f"Failed to load data: {e}")

    def _setup_sex_encoding(self) -> None:
        """Set up sex encoding values based on configuration."""
        # Values are already set in _initialize_data_attributes
        pass

    def _setup_exclusion_rules(self) -> None:
        """Set up exclusion rules based on phecode version."""
        if self.config.phecode_version == "1.2":
            self.use_exclusion = self.config.use_exclusion
        elif self.config.phecode_version == "X":
            # Phecode X doesn't use exclusions
            self.use_exclusion = False
        else:
            self.use_exclusion = False

    def _process_covariates(self) -> None:
        """Process covariate columns and handle special cases."""
        # Check if independent variable is in covariates
        if self.independent_variable_of_interest in self.covariate_cols:
            self.variable_of_interest_in_covariates = True
            self.covariate_cols.remove(self.independent_variable_of_interest)
            logger.info(
                f"'{self.independent_variable_of_interest}' removed from covariates "
                "as it's the variable of interest"
            )

        # Check if sex is in covariates
        if self.sex_at_birth_col in self.covariate_cols:
            self.sex_as_covariate = True
            self.covariate_cols.remove(self.sex_at_birth_col)

        # Set up variable columns
        self.gender_specific_var_cols = [self.independent_variable_of_interest] + self.covariate_cols
        self._setup_variable_columns()

    def _setup_variable_columns(self) -> None:
        """Set up variable columns based on sex data characteristics."""
        self.sex_values = self.covariate_df[self.sex_at_birth_col].unique().to_list()
        
        # Check for single sex cohort
        if (len(self.sex_values) == 1 and 
            (0 in self.sex_values or 1 in self.sex_values)):
            self._handle_single_sex_cohort()
        elif (len(self.sex_values) == 2 and 
              0 in self.sex_values and 1 in self.sex_values):
            self._handle_mixed_sex_cohort()
        else:
            raise InvalidDataError(
                f"Invalid sex values in column '{self.sex_at_birth_col}'. "
                "Expected 0 and/or 1, got: {self.sex_values}"
            )

    def _handle_single_sex_cohort(self) -> None:
        """Handle cohort with single sex."""
        self.data_has_single_sex = True
        self.single_sex_value = self.sex_values[0]
        self.var_cols = [self.independent_variable_of_interest] + self.covariate_cols
        
        if self.sex_as_covariate:
            logger.info(
                f"'{self.sex_at_birth_col}' not used as covariate "
                "since there is only one sex in data"
            )
        
        if self.independent_variable_of_interest == self.sex_at_birth_col:
            raise InvalidDataError(
                f"Cannot use '{self.sex_at_birth_col}' as variable of interest "
                "in single sex cohorts"
            )

    def _handle_mixed_sex_cohort(self) -> None:
        """Handle cohort with mixed sexes."""
        if self.independent_variable_of_interest == self.sex_at_birth_col:
            self.var_cols = self.covariate_cols + [self.sex_at_birth_col]
        else:
            if not self.sex_as_covariate:
                logger.warning(
                    "Data has both sexes but sex was not specified as a covariate. "
                    "Running PheWAS without sex as a covariate."
                )
            self.var_cols = ([self.independent_variable_of_interest] + 
                           self.covariate_cols + [self.sex_at_birth_col])

    def _validate_data(self) -> None:
        """Validate input data for common issues."""
        # Check for string type variables among covariates
        if pl.Utf8 in self.covariate_df[self.var_cols].schema.values():
            str_cols = [k for k, v in self.covariate_df.schema.items() if v is pl.Utf8]
            raise InvalidDataError(
                f"Column(s) {str_cols} contain string type. Only numerical types are accepted."
            )

        # Keep only relevant columns and remove nulls
        cols_to_keep = list(set(["person_id"] + self.var_cols))
        self.covariate_df = self.covariate_df[cols_to_keep].drop_nulls()
        self.cohort_size = len(self.covariate_df)
        
        if self.cohort_size == 0:
            raise InsufficientDataError("No valid participants after data cleaning")

        # Update phecode counts and list
        self.cohort_ids = self.covariate_df["person_id"].unique().to_list()
        self.phecode_counts = self.phecode_counts.filter(
            pl.col("person_id").is_in(self.cohort_ids)
        )
        
        self._setup_phecode_list()
        
        logger.info(f"Final cohort size: {self.cohort_size} participants")
        logger.info(f"Number of phecodes to process: {len(self.phecode_list)}")

    def _setup_phecode_list(self) -> None:
        """Set up the list of phecodes to process."""
        if self.config.phecode_to_process == "all":
            self.phecode_list = self.phecode_counts["phecode"].unique().to_list()
        else:
            if isinstance(self.config.phecode_to_process, str):
                self.phecode_list = [self.config.phecode_to_process]
            else:
                self.phecode_list = list(self.config.phecode_to_process)

    def _setup_output_filename(self) -> None:
        """Set up output filename for results."""
        if self.config.output_file_name is not None:
            filename = self.config.output_file_name
            if filename.endswith(".csv"):
                filename = filename[:-4]
            self.output_file_name = f"{filename}.csv"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file_name = f"phewas_{timestamp}.csv"    @staticmethod
    def _to_polars(df: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
        """Convert pandas DataFrame to polars DataFrame if needed.
        
        Args:
            df: Input dataframe (pandas or polars).
            
        Returns:
            Polars DataFrame.
        """
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
        return df

    def _exclude_range(self, phecode: str, phecode_df: Optional[pl.DataFrame] = None) -> List[str]:
        """Process exclude_range column for phecode exclusions.
        
        Args:
            phecode: Phecode of interest.
            phecode_df: Optional phecode mapping dataframe.
            
        Returns:
            List of phecodes to exclude.
        """
        if phecode_df is None:
            phecode_df = self.phecode_df.clone()

        # Check if phecode has exclude_range
        phecodes_without_exclude_range = (
            phecode_df
            .filter(pl.col("exclude_range").is_null())["phecode"]
            .unique()
            .to_list()
        )
        
        if phecode in phecodes_without_exclude_range:
            return []

        # Get exclude_range value
        exclude_value = (
            phecode_df
            .filter(pl.col("phecode") == phecode)["exclude_range"]
            .unique()
            .to_list()[0]
        )

        return self._parse_exclude_range(exclude_value)

    def _parse_exclude_range(self, exclude_value: str) -> List[str]:
        """Parse exclude_range string into list of phecodes.
        
        Args:
            exclude_value: Comma-separated exclude range string.
            
        Returns:
            List of phecode strings to exclude.
        """
        exclude_range = []
        
        for item in exclude_value.split(","):
            item = item.strip()
            if "-" in item:
                # Handle range (e.g., "777-780")
                first_code, last_code = item.split("-", 1)
                range_codes = [str(i) for i in range(int(first_code), int(last_code))]
                range_codes.append(last_code)
                exclude_range.extend(range_codes)
            else:
                # Handle single code
                exclude_range.append(item)
        
        return exclude_range

    def _case_control_prep(
        self, 
        phecode: str,
        phecode_counts: Optional[pl.DataFrame] = None, 
        covariate_df: Optional[pl.DataFrame] = None, 
        phecode_df: Optional[pl.DataFrame] = None,
        var_cols: Optional[List[str]] = None, 
        gender_specific_var_cols: Optional[List[str]] = None, 
        keep_ids: bool = False
    ) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:
        """Prepare case and control groups for a specific phecode.
        
        Args:
            phecode: Phecode of interest.
            phecode_counts: Phecode counts dataframe.
            covariate_df: Covariate dataframe.
            phecode_df: Phecode mapping dataframe.
            var_cols: Variable columns for general case.
            gender_specific_var_cols: Variable columns for gender-specific case.
            keep_ids: Whether to keep person_id column in results.
            
        Returns:
            Tuple of (cases, controls, analysis_var_cols).
        """
        # Use defaults if not provided
        if phecode_counts is None:
            phecode_counts = self.phecode_counts.clone()
        if covariate_df is None:
            covariate_df = self.covariate_df.clone()
        if phecode_df is None:
            phecode_df = self.phecode_df.clone()
        if var_cols is None:
            var_cols = copy.deepcopy(self.var_cols)
        if gender_specific_var_cols is None:
            gender_specific_var_cols = copy.deepcopy(self.gender_specific_var_cols)

        # Get sex restriction for this phecode
        sex_restriction = self._get_sex_restriction(phecode, phecode_df)
        if sex_restriction is None:
            return pl.DataFrame(), pl.DataFrame(), []

        # Determine analysis variables
        analysis_var_cols = (var_cols if sex_restriction == "Both" and self.sex_as_covariate 
                           else gender_specific_var_cols)

        # Filter covariate data by sex restriction
        covariate_df = self._filter_by_sex_restriction(covariate_df, sex_restriction)
        if len(covariate_df) == 0:
            return pl.DataFrame(), pl.DataFrame(), []

        # Generate cases and controls
        cases, controls = self._generate_cases_controls(
            phecode, phecode_counts, covariate_df, phecode_df
        )        # Remove duplicates and select columns
        return self._finalize_case_control_data(
            cases, controls, analysis_var_cols, keep_ids
        )

    def _get_sex_restriction(self, phecode: str, phecode_df: pl.DataFrame) -> Optional[str]:
        """Get sex restriction for a phecode."""
        if phecode_df is None:
            raise ValueError("phecode_df cannot be None")
        
        if not isinstance(phecode_df, pl.DataFrame):
            raise TypeError(
                f"Expected pl.DataFrame for phecode_df, got {type(phecode_df)}. "
                f"Value: {phecode_df}"
            )
        
        try:
            filtered_df = phecode_df.filter(pl.col("phecode") == phecode)
            sex_values = filtered_df["sex"].unique().to_list()
            return sex_values[0] if sex_values else None
        except Exception as e:
            logger.error(f"Error filtering phecode_df for phecode {phecode}: {e}")
            logger.error(f"phecode_df type: {type(phecode_df)}")
            logger.error(f"phecode_df shape: {phecode_df.shape if hasattr(phecode_df, 'shape') else 'No shape attribute'}")
            raise

    def _filter_by_sex_restriction(
        self, 
        covariate_df: pl.DataFrame, 
        sex_restriction: str
    ) -> pl.DataFrame:
        """Filter covariate data by phecode sex restriction."""
        if self.data_has_single_sex:
            # Check if single sex cohort matches restriction
            if ((sex_restriction == "Male" and self.male_value not in self.sex_values) or
                (sex_restriction == "Female" and self.female_value not in self.sex_values)):
                return pl.DataFrame()
            return covariate_df
        
        # Filter by sex for mixed-sex cohorts
        if sex_restriction == "Male":
            return covariate_df.filter(pl.col(self.sex_at_birth_col) == self.male_value)
        elif sex_restriction == "Female":
            return covariate_df.filter(pl.col(self.sex_at_birth_col) == self.female_value)
        
        return covariate_df

    def _generate_cases_controls(
        self,
        phecode: str,
        phecode_counts: pl.DataFrame,
        covariate_df: pl.DataFrame,
        phecode_df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Generate case and control groups."""
        # Generate cases
        case_ids = (
            phecode_counts
            .filter(
                (pl.col("phecode") == phecode) & 
                (pl.col("count") >= self.config.min_phecode_count)
            )["person_id"]
            .unique()
            .to_list()
        )
        cases = covariate_df.filter(pl.col("person_id").is_in(case_ids))

        # Generate controls with exclusions
        exclude_range = ([phecode] + self._exclude_range(phecode, phecode_df) 
                        if self.use_exclusion else [phecode])
        
        exclude_ids = (
            phecode_counts
            .filter(pl.col("phecode").is_in(exclude_range))["person_id"]
            .unique()
            .to_list()
        )
        controls = covariate_df.filter(~pl.col("person_id").is_in(exclude_ids))

        return cases, controls

    def _finalize_case_control_data(
        self,
        cases: pl.DataFrame,
        controls: pl.DataFrame,
        analysis_var_cols: List[str],
        keep_ids: bool
    ) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:
        """Remove duplicates and select final columns."""
        duplicate_check_cols = ["person_id"] + analysis_var_cols
        
        cases = cases.unique(subset=duplicate_check_cols)
        controls = controls.unique(subset=duplicate_check_cols)

        if keep_ids:
            cases = cases[duplicate_check_cols]
            controls = controls[duplicate_check_cols]
        else:
            cases = cases[analysis_var_cols]
            controls = controls[analysis_var_cols]

        return cases, controls, analysis_var_cols

    def get_phecode_data(self, phecode: str) -> Optional[pl.DataFrame]:
        """Get combined case and control data for a specific phecode.
        
        Args:
            phecode: Phecode of interest.
            
        Returns:
            Combined dataframe with case/control indicator, or None if no data.
        """
        cases, controls, _ = self._case_control_prep(phecode=phecode, keep_ids=True)

        if len(cases) == 0 and len(controls) == 0:
            logger.warning(f"No phecode data for {phecode}")
            return None

        cases = cases.with_columns(pl.lit(True).alias("is_phecode_case"))
        controls = controls.with_columns(pl.lit(False).alias("is_phecode_case"))
        
        return cases.vstack(controls)    @staticmethod
    def _result_prep(result, var_of_interest_index: int) -> Dict[str, float]:
        """Process results from statsmodels logistic regression.
        
        Args:
            result: Statsmodels logistic regression result object.
            var_of_interest_index: Index of the variable of interest.
            
        Returns:
            Dictionary containing key statistics.
        """
        # Extract convergence information
        results_as_html = result.summary().tables[0].as_html()
        converged = pd.read_html(results_as_html)[0].iloc[5, 1]
        
        # Extract confidence intervals
        results_as_html = result.summary().tables[1].as_html()
        res = pd.read_html(results_as_html, header=0, index_col=0)[0]

        # Calculate key statistics
        p_value = result.pvalues[var_of_interest_index]
        neg_log_p_value = -np.log10(p_value)
        beta = result.params[var_of_interest_index]
        conf_int_1 = res.iloc[var_of_interest_index]['[0.025']
        conf_int_2 = res.iloc[var_of_interest_index]['0.975]']
        odds_ratio = np.exp(beta)
        log10_odds_ratio = np.log10(odds_ratio)

        return {
            "p_value": p_value,
            "neg_log_p_value": neg_log_p_value,
            "beta": beta,
            "conf_int_1": conf_int_1,
            "conf_int_2": conf_int_2,
            "odds_ratio": odds_ratio,            "log10_odds_ratio": log10_odds_ratio,
            "converged": converged
        }

    def _logistic_regression(
        self, 
        phecode: str,
        phecode_counts: Optional[pl.DataFrame] = None, 
        covariate_df: Optional[pl.DataFrame] = None,
        var_cols: Optional[List[str]] = None, 
        gender_specific_var_cols: Optional[List[str]] = None,
        phecode_df: Optional[pl.DataFrame] = None
    ) -> Optional[Dict[str, Union[str, int, float, bool]]]:
        """Perform logistic regression for a single phecode.
        
        Args:
            phecode: Phecode of interest.
            phecode_counts: Phecode counts dataframe.
            covariate_df: Covariate dataframe.
            var_cols: Variable columns for general case.
            gender_specific_var_cols: Variable columns for gender-specific case.
            phecode_df: Phecode mapping dataframe.
            
        Returns:
            Dictionary with regression results or None if regression failed.
        """
        # Use provided dataframes or fall back to instance variables
        if phecode_counts is None:
            phecode_counts = self.phecode_counts
        if covariate_df is None:
            covariate_df = self.covariate_df
        if phecode_df is None:
            phecode_df = self.phecode_df
        if var_cols is None:
            var_cols = self.var_cols
        if gender_specific_var_cols is None:
            gender_specific_var_cols = self.gender_specific_var_cols
            
        # Prepare case-control data
        cases, controls, analysis_var_cols = self._case_control_prep(
            phecode=phecode, 
            phecode_counts=phecode_counts, 
            covariate_df=covariate_df, 
            phecode_df=phecode_df,
            var_cols=var_cols, 
            gender_specific_var_cols=gender_specific_var_cols
        )

        # Check minimum case/control requirements
        if not self._check_minimum_requirements(cases, controls, phecode):
            return None

        # Prepare regression data
        regression_data = self._prepare_regression_data(cases, controls, analysis_var_cols)
        if regression_data is None:
            return None

        y_fit, X_fit, var_index = regression_data

        # Check variant distribution
        if not self._check_variant_distribution(y_fit, X_fit, var_index, phecode):
            return None

        # Perform regression
        result = self._perform_regression(X_fit, y_fit, phecode)
        if result is None:
            return None

        # Process and return results
        return self._process_regression_result(result, var_index, phecode, len(cases), len(controls))

    def _check_minimum_requirements(self, cases: pl.DataFrame, controls: pl.DataFrame, phecode: str) -> bool:
        """Check if minimum case/control requirements are met."""
        cases_count = len(cases)
        controls_count = len(controls)
        
        if cases_count < self.config.min_cases or controls_count < self.config.min_cases:
            if self.config.verbose:
                logger.info(
                    f"Phecode {phecode} ({cases_count} cases/{controls_count} controls): "
                    "Not enough cases or controls. Pass."
                )
            return False
        return True

    def _prepare_regression_data(
        self, 
        cases: pl.DataFrame, 
        controls: pl.DataFrame, 
        analysis_var_cols: List[str]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """Prepare data matrices for regression."""
        # Add case/control indicators
        cases = cases.with_columns(pl.Series([1] * len(cases)).alias("y"))
        controls = controls.with_columns(pl.Series([0] * len(controls)).alias("y"))
        
        # Merge data
        regressors = cases.vstack(controls)
        
        # Get variable index
        var_index = regressors[analysis_var_cols].columns.index(self.independent_variable_of_interest)
        
        # Suppress warnings if requested
        if self.config.suppress_warnings:
            warnings.simplefilter("ignore")
        
        # Prepare data matrices
        y = regressors["y"].to_numpy()
        data_mat = regressors[analysis_var_cols].to_numpy()
        data_mat = sm.tools.add_constant(data_mat, prepend=False)
        
        # Remove missing values
        mask = ~np.isnan(y) & (~np.isnan(data_mat).any(axis=1))
        y_fit = y[mask]
        X_fit = data_mat[mask]
        
        if len(y_fit) == 0:
            return None
            
        return y_fit, X_fit, var_index

    def _check_variant_distribution(
        self, 
        y_fit: np.ndarray, 
        X_fit: np.ndarray, 
        var_index: int, 
        phecode: str
    ) -> bool:
        """Check if variant has sufficient distribution in cases and controls."""
        var_col = X_fit[:, var_index]
        cases_mask = y_fit == 1
        controls_mask = y_fit == 0
        
        carriers_cases = int(var_col[cases_mask].sum())
        non_carriers_cases = int(cases_mask.sum() - carriers_cases)
        carriers_controls = int(var_col[controls_mask].sum())
        non_carriers_controls = int(controls_mask.sum() - carriers_controls)
        
        if (carriers_cases == 0 or non_carriers_cases == 0 or
            carriers_controls == 0 or non_carriers_controls == 0):
            if self.config.verbose:
                logger.info(
                    f"Phecode {phecode}: insufficient variant distribution "
                    f"(cases → carriers={carriers_cases}, non-carriers={non_carriers_cases}; "
                    f"controls → carriers={carriers_controls}, non-carriers={non_carriers_controls}). Skipping."
                )
            return False
        return True

    def _perform_regression(self, X_fit: np.ndarray, y_fit: np.ndarray, phecode: str):
        """Perform the actual regression (standard logistic or Firth)."""
        if self.config.use_firth:
            return self._perform_firth_regression(X_fit, y_fit, phecode)
        else:
            return self._perform_standard_regression(X_fit, y_fit, phecode)

    def _perform_firth_regression(self, X_fit: np.ndarray, y_fit: np.ndarray, phecode: str):
        """Perform Firth penalized logistic regression."""
        try:
            from firthlogist import FirthLogisticRegression  # type: ignore
            model = FirthLogisticRegression(fit_intercept=False)
            
            try:
                return model.fit(X_fit, y_fit)
            except AttributeError as attr_err:
                if '_validate_data' in str(attr_err):
                    # Monkey-patch missing _validate_data method
                    def _validate_data(self, X, y=None, reset=True, **kwargs):
                        return X, y
                    model._validate_data = types.MethodType(_validate_data, model)
                    return model.fit(X_fit, y_fit)
                else:
                    raise
        except ImportError:
            raise ImportError("To use Firth regression, please install the 'firthlogist' package.")
        except Exception as err:
            if self.config.verbose:
                logger.error(f"Phecode {phecode} Firth regression error: {err}")
            return None

    def _perform_standard_regression(self, X_fit: np.ndarray, y_fit: np.ndarray, phecode: str):
        """Perform standard logistic regression."""
        try:
            logit = sm.Logit(y_fit, X_fit)
            return logit.fit(disp=False)
        except (np.linalg.linalg.LinAlgError, statsmodels.tools.sm_exceptions.PerfectSeparationError) as err:
            if self.config.verbose:
                logger.warning(f"Phecode {phecode}: {str(err)}")
            return None

    def _process_regression_result(
        self, 
        result, 
        var_index: int, 
        phecode: str, 
        cases_count: int, 
        controls_count: int
    ) -> Dict[str, Union[str, int, float, bool]]:
        """Process regression results into standardized format."""
        base_dict = {
            "phecode": phecode,
            "cases": cases_count,
            "controls": controls_count
        }
        
        # Extract statistics based on regression type
        if self.config.use_firth:
            stats_dict = self._extract_firth_stats(result, var_index)
        else:
            stats_dict = self._result_prep(result, var_index)
        
        result_dict = {**base_dict, **stats_dict}
        
        # Log results if verbose
        if self.config.verbose:
            logger.info(f"Phecode {phecode} ({cases_count} cases/{controls_count} controls): {result_dict}")
        
        return result_dict

    def _extract_firth_stats(self, result, var_index: int) -> Dict[str, float]:
        """Extract statistics from Firth regression result."""
        coef = result.coef_
        pvals = result.pvals_
        cis = result.ci_
        
        beta = coef[var_index]
        p_value = pvals[var_index]
        conf_int_1, conf_int_2 = cis[var_index]
        odds_ratio = np.exp(beta)
        neg_log_p_value = -np.log10(p_value)
        log10_odds_ratio = np.log10(odds_ratio)
        
        return {
            "p_value": p_value,
            "neg_log_p_value": neg_log_p_value,
            "beta": beta,
            "conf_int_1": conf_int_1,
            "conf_int_2": conf_int_2,
            "odds_ratio": odds_ratio,
            "log10_odds_ratio": log10_odds_ratio,
            "converged": True  # Assume Firth always converges if no error
        }

    def run(
        self,
        parallelization: str = "multithreading",
        n_threads: Optional[int] = None
    ) -> Optional[pl.DataFrame]:
        """Run parallel logistic regressions for all phecodes.
        
        Args:
            parallelization: Parallelization method ("multithreading" only supported).
            n_threads: Number of threads to use. Defaults to 2/3 of CPU count.
            
        Returns:
            PheWAS results DataFrame or None if no results.
            
        Raises:
            ValueError: If invalid parallelization method is provided.
        """
        if n_threads is None:
            n_threads = max(1, round(os.cpu_count() * 2 / 3))
            
        logger.info("Running PheWAS")
        
        if parallelization != "multithreading":
            raise ValueError("Only 'multithreading' parallelization is currently supported")
        
        # Run parallel regression analysis
        result_dicts = self._run_parallel_regressions(n_threads)
        
        if not result_dicts:
            logger.warning("No analysis completed. Please check your inputs.")
            return None
        
        # Process and save results
        self._process_and_save_results(result_dicts)
        self._log_summary()
        
        return self.results

    def _run_parallel_regressions(self, n_threads: int) -> List[Dict]:
        """Run logistic regressions in parallel."""
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            jobs = [
                executor.submit(
                    self._logistic_regression,
                    phecode,
                    self.phecode_counts.clone(),
                    self.covariate_df.clone(),
                    copy.deepcopy(self.var_cols),
                    copy.deepcopy(self.gender_specific_var_cols),
                    self.phecode_df.clone()
                ) for phecode in self.phecode_list
            ]
            result_dicts = [
                job.result() for job in 
                tqdm(as_completed(jobs), total=len(self.phecode_list), desc="Running regressions")
            ]
        
        return [result for result in result_dicts if result is not None]

    def _process_and_save_results(self, result_dicts: List[Dict]) -> None:
        """Process results and save to file."""
        result_df = pl.from_dicts(result_dicts)
        
        # Join with phecode metadata
        self.results = result_df.join(
            self.phecode_df[["phecode", "sex", "phecode_string", "phecode_category"]].unique(),
            how="left",
            on="phecode"
        ).rename({"sex": "phecode_sex_restriction"})
        
        # Calculate summary statistics
        self.tested_count = len(self.results)
        self.not_tested_count = len(self.phecode_list) - self.tested_count
        self.bonferroni = -np.log10(DEFAULT_BONFERRONI_ALPHA / self.tested_count)
        self.phecodes_above_bonferroni = self.results.filter(
            pl.col("neg_log_p_value") > self.bonferroni
        )
        self.above_bonferroni_count = len(self.phecodes_above_bonferroni)
        
        # Save results
        self.results.write_csv(self.output_file_name)
        logger.info(f"PheWAS results saved to {self.output_file_name}")

    def _log_summary(self) -> None:
        """Log summary statistics."""
        logger.info("PheWAS Completed")
        logger.info(f"Number of participants in cohort: {self.cohort_size}")
        logger.info(f"Number of phecodes in cohort: {len(self.phecode_list)}")
        logger.info(f"Number of phecodes having less than {self.config.min_cases} cases or controls: {self.not_tested_count}")
        logger.info(f"Number of phecodes tested: {self.tested_count}")
        logger.info(f"Suggested Bonferroni correction (-log₁₀ scale): {self.bonferroni:.3f}")
        logger.info(f"Number of phecodes above Bonferroni correction: {self.above_bonferroni_count}")


def create_phewas_from_args(args) -> PheWAS:
    """Create PheWAS object from command line arguments."""
    # Use legacy parameter style for backward compatibility
    return PheWAS(
        phecode_version=args.phecode_version,
        phecode_count_csv_path=args.phecode_count_csv_path,
        cohort_csv_path=args.cohort_csv_path,
        sex_at_birth_col=args.sex_at_birth_col,
        covariate_cols=args.covariates,
        independent_variable_of_interest=args.independent_variable_of_interest,
        male_as_one=args.male_as_one,
        phecode_to_process=args.phecode_to_process,
        use_exclusion=args.use_exclusion,
        min_cases=args.min_case,
        min_phecode_count=args.min_phecode_count,
        output_file_name=args.output_file_name
    )


def main() -> None:
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description="PheWAS analysis tool.")
    
    # Required arguments
    parser.add_argument(
        "-p", "--phecode_count_csv_path",
        type=str, required=True,
        help="Path to the phecode count csv file."
    )
    parser.add_argument(
        "-c", "--cohort_csv_path",
        type=str, required=True,
        help="Path to the cohort csv file."
    )
    parser.add_argument(
        "-pv", "--phecode_version",
        type=str, required=True, choices=["1.2", "X"],
        help="Phecode version."
    )
    parser.add_argument(
        "-cv", "--covariates",
        nargs="+", type=str, required=True,
        help="List of covariates to use in PheWAS analysis."
    )
    parser.add_argument(
        "-i", "--independent_variable_of_interest",
        type=str, required=True,
        help="Independent variable of interest."
    )
    parser.add_argument(
        "-s", "--sex_at_birth_col",
        type=str, required=True,
        help="Sex at birth column."
    )
    
    # Optional arguments
    parser.add_argument(
        "-mso", "--male_as_one",
        type=bool, required=False, default=True,
        help="Whether male was assigned as 1 in data."
    )
    parser.add_argument(
        "-pl", "--phecode_to_process",
        nargs="+", type=str, required=False, default="all",
        help="List of specific phecodes to use in PheWAS analysis."
    )
    parser.add_argument(
        "-e", "--use_exclusion",
        type=bool, required=False, default=False,
        help="Whether to use phecode exclusions. Only applicable for phecode 1.2."
    )
    parser.add_argument(
        "-mc", "--min_case",
        type=int, required=False, default=DEFAULT_MIN_CASES,
        help="Minimum number of cases required to be tested."
    )
    parser.add_argument(
        "-mpc", "--min_phecode_count",
        type=int, required=False, default=DEFAULT_MIN_PHECODE_COUNT,
        help="Minimum number of phecode counts required to be considered as case."
    )
    parser.add_argument(
        "-t", "--threads",
        type=int, required=False, default=None,
        help="Number of threads to use for parallel processing."
    )
    parser.add_argument(
        "-o", "--output_file_name",
        type=str, required=False, default=None,
        help="Output file name for results."
    )
    
    args = parser.parse_args()
    
    try:
        # Create and run PheWAS
        phewas = create_phewas_from_args(args)
        phewas.run(n_threads=args.threads)
        
    except Exception as e:
        logger.error(f"PheWAS analysis failed: {e}")
        sys.exit(1)
if __name__ == "__main__":
    main()
