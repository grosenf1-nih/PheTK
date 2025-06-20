# PheTK - The Phenotype Toolkit
The official repository of PheTK, a fast python library for Phenome Wide Association Studies (PheWAS) utilizing both phecode 1.2 and phecodeX 1.0.

__Reference__: Tam C Tran, David J Schlueter, Chenjie Zeng, Huan Mo, Robert J Carroll, Joshua C Denny, PheWAS analysis on large-scale biobank data with PheTK, Bioinformatics, Volume 41, Issue 1, January 2025, btae719, https://doi.org/10.1093/bioinformatics/btae719

__Contact__: [PheTK@mail.nih.gov](mailto:PheTK@mail.nih.gov)

> **🆕 Latest Updates**: PheTK has been recently refactored with modern Python best practices, including improved type safety, better error handling, and a new configuration-based API for enhanced usability. The legacy API remains fully supported for backward compatibility. See the [PheWAS module section](#53-phewas-module) for details on the new configuration approach.

***

## QUICK LINKS
- [Installation](#1-installation)
- [System requirements](#2-system-requirements)
- [1-minute PheWAS demo](#3-1-minute-phewas-demo)
- [PheTK description](#4-descriptions)
- [Usage examples](#5-usage)
  - [Cohort module](#51-cohort-module)
  - [Phecode module](#52-phecode-module)
  - [PheWAS module](#53-phewas-module)
  - [Plot module](#54-plot-module)
- Platform specific tutorial(s):
  - ___All of Us___: [APOE PheWAS jupyter notebook](https://workbench.researchallofus.org/workspaces/aou-rw-42a1ea44/chargedemo/analysis/preview/2_PheWAS_with_PheTK_demo.ipynb).
This notebook demonstrates PheTK usage on the _All of Us_ Researcher Workbench with APOE PheWAS as a study example.
Please note that this link is only accessible to _All of Us_ registered users.
- [Changelogs and releases](https://github.com/nhgritctran/PheTK/releases): 
from v0.1.45, please use [GitHub Releases](https://github.com/nhgritctran/PheTK/releases) for latest versions and changelogs. 
Legacy changelogs were archived in [CHANGELOG.md](CHANGELOG.md).
- Resource to learn about PheWAS and phecode: [The PheWAS Catalog](https://phewascatalog.org/).

***

## 1. INSTALLATION
The latest version of PheTK can be installed using pip install command in a terminal (Python 3.7 or newer):

```
pip install PheTK --upgrade
```

or Jupyter Notebook (restart kernel after installation and prior to importing):
```
!pip install PheTK --upgrade
```

To check current installed version:
```
pip show PheTK | grep Version
```

## 2. SYSTEM REQUIREMENTS
PheTK was developed for efficient processing of large data while being resource friendly. 
It was tested on different platforms from laptops to different cloud environments.

Here are some minimum VM configuration suggestions for _All of Us_ users:
- Cohort module: default General Analysis (4 CPUs, 15GB RAM) or Hail Genomic Analysis 
(main VM and 2 workers of 4 CPUs, 15GB RAM each) VMs should work. 
Hail Genomic Analysis is only needed for .by_genotype() method.
- Phecode module: a minimum of 8 CPUs, 52GB RAM standard VM should work for current v7 data.
This might require more RAM depending on user custom workflow or data - 
usually Jupyter Python kernel would die at if there is not enough memory.
- PheWAS module: default General Analysis VM (4 CPUs, 15GB RAM) should work. 
However, more CPUs would speed up analysis and using low configurations do not necessarily save computing cost
since total runtime would be longer.
- Plot module: default General Analysis VM (4 CPUs, 15GB RAM) should work.

In practice, users could try different available machine configurations to achieve optimal performance and cost-effectiveness.

## 3. 1-MINUTE PHEWAS DEMO

User can run the quick 1-minute PheWAS demo with the following command in a terminal:

```
python3 -m PheTK.Demo
```

Or in Jupyter Notebook:

```
from PheTK import Demo

Demo.run()
```

The example files (`example_cohort.csv`, `example_phecode_counts.csv`, and `example_phewas_results.csv`) 
generated in this Demo should be in users' current working directory. 
New-to-PheWAS users could explore these files to get a sense of what data are used or generated in PheWAS with PheTK.

## 4. DESCRIPTIONS
PheTK is a fast python library for Phenome Wide Association Studies (PheWAS) utilizing both phecode 1.2 and phecodeX 1.0.

### 4.1. PheWAS workflow and PheTK modules
![PheWAS workflow and PheTK modules](img/readme/PheTK_flowchart.png)
Standard PheWAS workflow. Green texts are PheTK module names. 
Black components are supported while gray ones are not supported by PheTK currently.

### 4.2. PheTK module descriptions
This table will be updated as we update PheTK. 

All modules can be used together or independently, 
e.g., users who only need to run PheWAS analysis can provide their own cohort and phecode count data as input for PheWAS module. 

| Module  | Class   | Method(s)     | Platform    | Requirements/Notes                                                           |
|---------|---------|---------------|-------------|------------------------------------------------------------------------------|
| Cohort  | Cohort  | by_genotype   | _All of Us_ | None                                                                         |
|         |         |               | Other       | Variant data stored in Hail matrix table                                     |
|         |         | add_covariate | _All of Us_ | None                                                                         |
|         |         |               | Other       | Google BigQuery OMOP database                                                |
|         |         |               |             | Required tables: person, condition_occurrence, observation, death, & concept |
| Phecode | Phecode | count_phecode | _All of Us_ | None                                                                         | 
|         |         |               | Other       | User provided cohort ICD code data                                           |
|         |         |               |             | User can use custom ICD-to-phecode mapping table.                            |
| PheWAS  | PheWAS  | all methods   | Any         | None                                                                         |
| Plot    | Plot    | all methods   | Any         | None                                                                         |
| Demo    |         | all methods   | Any         | None                                                                         |

_All of Us_: the _All of Us_ Research Program (https://allofus.nih.gov/)

## 5. USAGE

### 5.1. Cohort module
Cohort module can be used for generating genetic cohort and add certain covariates to a cohort.

#### 5.1.1. by_genotype

This function takes genetic variant information as input, 
and generates cohort with matching genotypes as an output csv file.
As this function uses Hail to extract data from Hail matrix tables, it must be run in a compatible environment,
e.g., a dataproc cluster on All of Us researcher workbench or UK Biobank RAP.

For example, we generate cohort for _CFTR_ variant chr7-117559590-ATCT-A with 
heterozygous (0/1 genotype) participants as cases and homozygous reference (0/0 genotype) participants as controls.

#### Jupyter Notebook example for _All of Us_ Researcher Workbench:
For _All of Us_ data version 7, the default Hail matrix table is the ACAF (common variant) table.
User can use a different table by providing table location in the mt_path parameter.
```
from PheTK.Cohort import Cohort

# instantiate class Cohort object for _All of Us_ database version 7
cohort = Cohort(platform="aou", aou_db_version=7)

# generate cohort by genotype
cohort.by_genotype(
    chromosome_number=7,
    genomic_position=117559590,
    ref_allele="ATCT",
    alt_allele="A",
    case_gt="0/1",
    control_gt="0/0",
    reference_genome="GRCh38",
    mt_path=None,
    output_file_name="cftr_cohort.csv"
)
```

#### Jupyter Notebook example for other platforms:
For other platforms, user need to provide the location of Hail matrix table file for mt_path parameter.
```
from PheTK.Cohort import Cohort

# instantiate class Cohort object for _All of Us_ database version 7
cohort = Cohort(platform="custom")

# generate cohort by genotype
cohort.by_genotype(
    chromosome_number=7,
    genomic_position=117559590,
    ref_allele="ATCT",
    alt_allele="A",
    case_gt="0/1",
    control_gt="0/0",
    reference_genome="GRCh38",
    mt_path="/path/to/hail_matrix_table.mt",
    output_file_name="cftr_cohort.csv"
)
```

#### 5.1.2. add_covariates
This function is currently customized for the _All of Us_ Research Platform. 
It takes a cohort csv file and covariate selection as input, 
and generate a new cohort csv file with covariate data added as output. 
Input cohort data must have "person_id" column.

For non-_All of Us_ platforms, a Google BigQuery dataset ID must be provided.

In this example, we are adding age at last diagnosis event, sex at birth and 10 genetic PCs (provided by _All of Us_).
These options were set to True (or 10 in case of first_n_pcs).

The covariates shown in this example are currently supported by PheTK. Users should only change parameter value to True 
for covariates to be used in subsequent PheWAS. All parameters are set to False by default, i.e., user only need to 
specify parameters of interest as shown in the "short version". 

It is highly recommended that users should decide which covariates to use for the study based on their data, 
and it is perfectly fine to add or use their own covariate data if necessary. 

#### Jupyter Notebook example for _All of Us_ Researcher Workbench:
```
# user can skip the import and instantiation steps if running continuously 
# from previous by_genotype example, i.e., skip directly to add covariates step.
from PheTK.Cohort import Cohort

# instantiate class Cohort object for _All of Us_ database version 7
cohort = Cohort(platform="aou", aou_db_version=7)

# RUN EITHER LONG OR SHORT VERSION BELOW
# add covariates - long version, including all currently supported covariate options
cohort.add_covariates(
    cohort_csv_path="cftr_cohort.csv",
    natural_age=False,
    age_at_last_event=True,
    sex_at_birth=True,
    ehr_length=False,
    dx_code_occurrence_count=False,
    dx_condition_count=False,
    genetic_ancestry=False,
    first_n_pcs=10,
    drop_nulls=True,
    output_file_name="cohort_with_covariates.csv"
)

# add covariates - short version, i.e., users do not need to list unused covariates
cohort.add_covariates(
    cohort_csv_path="cftr_cohort.csv",
    age_at_last_event=True,
    sex_at_birth=True,
    first_n_pcs=10,
    drop_nulls=True,
    output_file_name="cohort_with_covariates.csv"
)
```

Covariate descriptions:
- _natural_age_: current age or age at death
- _age_at_last_event_: age at last diagnosis event (ICD or SNOMED) in EHR.
- _sex_at_birth_: sex at birth
- _ehr_length_: EHR duration, in year, from first to last diagnosis code
- _dx_code_occurrence_count_: counts the occurrences of diagnosis codes throughout EHR of each participant.
For example: person 1 having R50 (fever) code on 5 different dates, R05 (cough) code on 3 different dates, 
and R05.1 (acute cough) code on 2 different dates, will have a dx_code_occurrence_count = 10.
- _dx_condition_count_: counts the number of unique conditions occurred throughout EHR of each participant.
For example, for the same person 1 above, the dx_condition_count = 3 (R05 - cough, R05.1 - acute cough, R50 - fever).
- _genetic_ancestry_: returns string values of predicted ancestries, e.g., "eur", "afr", etc. 
These are only useful if user would like to filter data by genetic ancestries. 
- _first_n_pcs_: retrieves first n genetic PC components from genetic PCA data generated by All of Us.
- _drop_nulls_: remove rows containing null values in any column.

#### Jupyter Notebook example for other platforms with OMOP data stored in Google BigQuery:
The only difference in this case is that user need to provide dataset ID for gbq_dataset_id parameter.
The rest should be the same as above example.
Please make sure the custom database meet the requirements in section 4.2.
```
from PheTK.Cohort import Cohort

cohort = Cohort(platform="custom", gbq_dataset_id="Google_BigQuery_dataset_id")
```

### 5.2. Phecode module
Phecode module is used to retrieve ICD code data of participants, map ICD codes to phecode 1.2 or phecodeX 1.0, 
and aggregate the counts for each phecode of each participant.

The ICD code retrieval is done automatically for _All of Us_ platform when users instantiate class Phecode.
For other platforms, users must provide your own ICD code data.

Example of ICD code data: 
- Each row must be unique, i.e., there should not be 2 instances of 1 ICD code in the same day.
- Data must have these exact column names. 
- "vocabulary_id" column can be replaced with "flag" column with values of 9 and 10.

Example with "vocabulary_id" column:

| person_id | date      | vocabulary_id | ICD   |
|-----------|-----------|---------------|-------|
| 13579     | 1-11-2010 | ICD9CM        | 786.2 |
| 13579     | 1-31-2010 | ICD9CM        | 786.2 |
| 13579     | 12-4-2017 | ICD10CM       | R05.1 |
| 24680     | 3-12-2012 | ICD9CM        | 659.2 |
| 24680     | 4-18-2018 | ICD10CM       | R50   |

Example with "flag" column:

| person_id | date      | flag | ICD   |
|-----------|-----------|------|-------|
| 13579     | 1-11-2010 | 9    | 786.2 |
| 13579     | 1-31-2010 | 9    | 786.2 |
| 13579     | 12-4-2017 | 10   | R05.1 |
| 24680     | 3-12-2012 | 9    | 659.2 |
| 24680     | 4-18-2018 | 10   | R50   |

In these examples, we will map US ICD codes (ICD-9-CM & ICD-10-CM) to phecodeX for _All of Us_ and custom platforms.

#### Jupyter Notebook example for _All of Us_:
```
from PheTK.Phecode import Phecode

phecode = Phecode(platform="aou")
phecode.count_phecode(
    phecode_version="X", 
    icd_version="US",
    phecode_map_file_path=None, 
    output_file_name="my_phecode_counts.csv"
)
```

#### Jupyter Notebook example for other platforms
```
from PheTK.Phecode import Phecode

phecode = Phecode(platform="custom", icd_df_path="/path/to/my_icd_data.csv")
phecode.count_phecode(
    phecode_version="X", 
    icd_version="US", 
    phecode_map_file_path=None,
    output_file_name="my_phecode_counts.csv"
)
```

Users can provide their own phecode mapping file by adding a csv file path to `phecode_map_file_path`.
If user provides their own ICD data, platform should be set to "custom".

Custom phecode mapping file must have columns:
- `phecode`: contains code values of phecodes
- `ICD`: contains code values of ICD codes
- `flag`: specifies whether ICD codes is of version 9 or 10; takes numerical 9 or 10 as values.
- `sex`: sex of ICD/phecode; takes "Male", "Female", or "Both" as values.
- `phecode_string`: literal name of phecodes as strings
- `phecode_category`: specifies which category a phecode belongs to.
- `exclude_range`: only applicable for phecode 1.2.; specifies the range of exclusion if used in PheWAS;
for example, "008.5,008.7,008.51,008.52,008.6"

### 5.3. PheWAS module
It is recommended to run Demo example above and have a look at example cohort and phecode counts file to 
be familiar with input data format. The example files should be generated in user's current working directory.

PheWAS class can be instantiated in two ways:
1. **New configuration-based approach (recommended)**: Using a `PheWASConfig` object for better organization and validation
2. **Legacy approach**: Direct parameter passing (maintained for backward compatibility)

Both approaches can be used in Linux command line interface (CLI) and any Python environment, e.g., 
Jupyter Notebook/Lab.

In these examples, we would like to run PheWAS with phecodeX for example data generated by Demo module, 
with age, sex and 3 genetic PCs as covariates, and an independent variable of interest 
(for which PheWAS summary statistics will be generated).
For each phecode, a participant must have a minimum count of 2 phecode events to be considered a case.
There must be at least 50 cases and 50 controls for the phecode to be tested.

#### CLI example:
The command line interface remains unchanged:
```
python3 -m PheTK.PheWAS \
--phecode_version X \
--cohort_csv_path example_cohort.csv \
--phecode_count_csv_path example_phecode_counts.csv \
--sex_at_birth_col sex \
--male_as_one True \
--covariates age sex pc1 pc2 pc3 \
--independent_variable_of_interest independent_variable_of_interest \
--min_case 50 \
--min_phecode_count 2 \
--output_file_name example_phewas_results.csv
```

#### Jupyter Notebook example (new configuration-based approach - recommended):
```
from PheTK.PheWAS import PheWAS, PheWASConfig

# Create configuration object
config = PheWASConfig(
    phecode_version="X",
    phecode_count_csv_path="example_phecode_counts.csv",
    cohort_csv_path="example_cohort.csv",
    sex_at_birth_col="sex",
    male_as_one=True,
    covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
    independent_variable_of_interest="independent_variable_of_interest",
    min_cases=50,
    min_phecode_count=2,
    output_file_name="example_phewas_results.csv"
)

# Create PheWAS object with configuration
example_phewas = PheWAS(config)
example_phewas.run()
```

#### Jupyter Notebook example (legacy approach - still supported):
```
from PheTK.PheWAS import PheWAS

example_phewas = PheWAS(
    phecode_version="X",
    phecode_count_csv_path="example_phecode_counts.csv",
    cohort_csv_path="example_cohort.csv",
    sex_at_birth_col="sex",
    male_as_one=True,
    covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
    independent_variable_of_interest="independent_variable_of_interest",
    min_cases=50,
    min_phecode_count=2,
    output_file_name="example_phewas_results.csv"
)
example_phewas.run()
```

#### Benefits of the new configuration approach:
- **Better organization**: All parameters are clearly defined in one configuration object
- **Input validation**: Automatic validation of parameters at configuration creation
- **Type safety**: Enhanced error checking and better IDE support
- **Easier testing**: Configuration objects can be easily created and modified for testing
- **Documentation**: Built-in parameter documentation and help

Notes:
- Each entry in sex_at_birth column should be either 0 or 1 for female or male. The default is male = 1 and female = 0.
- User can use male_as_one to specify where male was coded as 1 (male_as_one=True) or 0 (male_as_one=False).
- In the above example, "sex" column was declared twice, once in sex_at_birth_col and once in covariate_cols.
sex_at_birth_col is always required as certain phecodes are sex restricted.
If user would like to use sex as a covariate, sex column must be included in covariate_cols. 
- PheWAS results often include both converged and non-converged phecodes. 
This was intentional, as there can be multiple factors affect regression model convergence. 
Therefore, non_converged phecodes are kept and flagged using `converged` column to allow users to do further investigation if needed. 

#### Configuration parameters
When using the `PheWASConfig` approach, all parameters are validated automatically. Here are the key configuration options:

**Required parameters:**
- `phecode_version`: "1.2" or "X"
- `phecode_count_csv_path`: Path to phecode counts CSV file
- `cohort_csv_path`: Path to cohort CSV file  
- `sex_at_birth_col`: Column name for sex at birth
- `covariate_cols`: List of covariate column names
- `independent_variable_of_interest`: Column name for the variable of interest

**Optional parameters (with defaults):**
- `male_as_one`: True (male=1, female=0)
- `icd_version`: "US" 
- `min_cases`: 50
- `min_phecode_count`: 2
- `use_exclusion`: False
- `verbose`: False
- `suppress_warnings`: True
- `use_firth`: False (requires firthlogist package if True)

For a complete list of parameters and their descriptions, see the `PheWASConfig` class documentation.

<a id="get_phecode_data"></a>
#### Get cohort data of a phecode
Method `.get_phecode_data()` can be used to get cohort data for a phecode of interest.
A common use case is to look at input data of certain phecodes after a PheWAS run.

In addition to the existing columns, e.g., person_id, covariates, etc., from input data, 
the generated table has `is_phecode_case` boolean column to specify whether an individual is a case or control for that phecode.

For example, after creating a PheWAS class instance using either approach above, 
user can call `.get_phecode_data()` method as below:
```
# get cohort data for phecode "R05.1"
phecode_data = example_phewas.get_phecode_data("R05.1")
print(phecode_data.head())
```


### 5.4. Plot module
Plot class is instantiated with path to PheWAS result csv file.
After that, a plot type method can be called to generate a plot, 
e.g., calling manhattan() method to make Manhattan plot.

In this example, we are generating a Manhattan plot for the PheWAS results created by module Demo.

#### Demo Jupyter Notebook example:
```
from PheTK.Plot import Plot

p = Plot("example_phewas_results.csv")
p.manhattan(label_values="p_value", label_count=1, save_plot=True)
```
The above code example generates this Manhattan plot figure:

![Example Manhattan plot](img/readme/example_manhattan.png)

#### Some main customization options:
The features below can be used individually or in combinations to customize Manhattan Plot.

##### Plot only converged phecodes:
As mentioned in PheWAS module above, PheWAS results contain both converged and non-converged phecodes.
By default, PheTK will exclude non-converged phecodes when plotting for better interpretation.

Users can include non-converged phecodes by setting boolean parameter `converged_only` to `False`:
```
p = Plot("example_phewas_results.csv", converged_only=False)
```

##### Set custom Bonferroni value/line:
```
p = Plot("example_phewas_results.csv", bonferroni=your_custom_value)
```

##### Set custom color palette:
Color palette should be provided in a tuple format. Each color will be used for a phecode category.
If the number of colors is less than number of phecode categories, the colors will be cycled back to the first color and so on.
PheTK uses matplotlib color names.
```
p = Plot("example_phewas_results.csv", color_palette=("blue", "indianred", "darkcyan"))
```

##### Use beta values (effect sizes) as marker size and change scale factor:
This feature can be turned on using boolean parameter `marker_size_by_beta` to `True`.
Users can adjust marker size using parameter `marker_scale_factor` which has a default value of 1.
(Note that this parameter only works when `marker_size_by_beta=True`).

```
p.manhattan(marker_size_by_beta=True, marker_scale_factor=1)
```

##### Select what to label
By default, PheTK will label by top p-values. This can be changed using parameter `label_values`.
This parameter accepts single string value or list of phecodes, or 3 preset values of "p_value", "positive_beta", or "negative_beta".
Note that, if provided text values do not match the preset values or phecodes in PheWAS results, they will not be labeled.

Label by top p-values:
```
p.manhattan(label_values="p_value")
```

Label by top positive beta values:
```
p.manhattan(label_values="positive_beta")
```

Label by top negative beta values:
```
p.manhattan(label_values="negative_beta")
```

Label by specific phecode(s) of interests:
```
p.manhattan(label_values=["GE_982", "NS_351"])
```

##### Number of data points to be labeled
By default, PheTK will label top 10 data points of label values. To change this, use parameter `label_count`
```
p.manhattan(label_count=10)
```

##### Select phecode category to label
Users can choose to label only in a specific phecode category or multiple categories using parameter `phecode_categories`
```
p.manhattan(phecode_categories=["Neoplasms", "Genetic"])
```

##### Save plot
```
p.manhattan(save_plot=True)
```

Further details on plot customization options will be provided in a separate document in the near future.
