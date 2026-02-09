# GENiAL
(Genetic and) EEG-based Neurodevelopmental Analysis in Autism/ADHD via Learning algorithms  

**[NEW/REVISED TITLE]** Behavioral Clustering and Resting-state EEG signatures of Neurodevelopmental Profiles Using Machine Learning

This repository includes the necessary code for the GENiAL project:  
* _Question._ In children with autism spectrum disorder, can machine learning applied to EEG data differentiate those with copy number variant (CNV) risks from those without?
* _Objectives._ The primary objective of the study is to determine if machine learning can 
* _Data._ Q1K + Brain Canada

## Dependencies
### Create a virtual environment
```python3 -m venv venv```

### Activate virtual environment
```source venv/bin/activate```  

### Install packages
```pip install matplotlib mne pandas```

_Select interpreter_  
```Cmd+Shift+P```  
```Python: Select Interpreter```  
```Python 3.x.x 64-bit ('venv': venv)```  

# Steps

## STEP 1 - Clean and prepare database
1. Import REDCap reports:
   - Q1K: `ECN-EEG-IQ-GEN-CHUSJ` and `ECNBEHAVIORALVERBALI` from REDCap
   - Brain Canada: `ECNBCSRSIQ` and `ECNDiagnostics` from REDCap
2. Save all REDCap exports in `/DATA/REDCAP_REPORTS/` (organized by Q1K and BrainCanada subfolders).
3. Prepare CNV files (if needed):
   - Extract CNV information from REDCap data (chromosome, boundaries, genome version)
   - Prepare files for the online CNV calculator at https://cnvprediction.urca.ca/index.html
   - Save CNV outputs to `/DATA/Genetic_cnv_scores/` folder
4. Run `SCRIPTS/1-initial-cleanup/preprocess_demog_beh_iq_gen.py`:
   - This script combines demographic, behavioral, IQ, and genetic data from Q1K and Brain Canada
   - Output: Preprocessed CSV file saved in `/DATA/OUTPUTS/Preprocessed/`

### EEG PREPROCESSING
1. Gather Raw EEGs and save on a HARD DRIVE.
2. Follow HAPPE preprocessing pipeline from [lab's repository](https://github.com/labo-NED/EEG_preprocessing_pipeline):
   - Run preprocessing with 2s parameters (for power feature extraction)
   - Run preprocessing with 5s parameters (for entropy/complexity feature extraction)  
   *NOTE* _This preprocessing step might take 4 days for each preprocessing (2s and 5s)._
3. Extract EEG features (choose one method):
   - **Python**: Run `SCRIPTS/3-EEG/compute_features.py` (extracts both 2s and 5s features)
   - **MATLAB**: Run `SCRIPTS/3-EEG/Matlab/compute_features.m` (extracts both 2s and 5s features)
   - Both scripts extract power features from 2s epochs and entropy/complexity features from 5s epochs
4. Aggregate features:
   - **By ROI**: Run `SCRIPTS/3-EEG/aggregate_features_by_roi.py` (aggregates features by brain regions)
   - **Global**: Run `SCRIPTS/3-EEG/aggregate_features_global.py` (averages features across all channels)
   - Output: Aggregated CSV files saved in `/DATA/OUTPUTS/eeg_features/`

## STEP 2 - Behavioral Clustering
1. Run behavioral clustering on preprocessed data:
   - **SOM clustering**: Run `SCRIPTS/2-clustering/SOM_behavioral_clustering.py`
   - **GFMM clustering**: Run `SCRIPTS/2-clustering/gfmm_behavioral_clustering.py`
   - Output: Clustered CSV files saved in `/DATA/OUTPUTS/Clustered/`

## STEP 3 - Merge EEG and Behavioral Data
1. Merge EEG features with clustered behavioral data:
   - Run `SCRIPTS/3-EEG/merge_eeg_features_to_db.py` (for global features)
   - Run `SCRIPTS/3-EEG/merge_5s_features_to_db.py` (for 5s-specific features, if needed)
   - Output: Merged CSV files saved in `/DATA/OUTPUTS/Final/`

## STEP 4 - Statistical Analysis
1. Run statistical analyses on merged behavioral and EEG data:
   - **Main analysis**: Run `SCRIPTS/4-statistical-analysis/genial_stats.r`
     - Performs ANCOVA for EEG features by cluster (controlling for age and sex)
     - Log-transforms power band features for normality
     - Generates boxplots with pairwise comparisons
     - Computes inter-feature correlations
     - Output: Statistical results and plots saved in `/DATA/OUTPUTS/Stats/`
   - **Diagnosis pie charts**: Run `SCRIPTS/4-statistical-analysis/diagnosis_pie_charts.r`
     - Creates diagnostic distribution pie charts by cluster
   - **Behavioral scores plots**: Run `SCRIPTS/4-statistical-analysis/generate_behavioral_scores_plot.r`
     - Generates visualizations of behavioral scores by cluster
   - **Additional analyses**: Run `SCRIPTS/4-statistical-analysis/stats.r` for comprehensive analyses including:
     - MANCOVA for behavioral measures
     - Demographic comparisons between clusters
     - EEG feature ANOVAs
     - Regression analyses (SRS/ADHD ~ EEG features)
     - Correlation analyses

# Database

## Preprocessing
| Initial Item              | Renamed/processed                                      | New  |
| ------------------------- | ------------------------------------------------------ | ---- |
| Participant_id             | Participant_id                                          | No   |
| EEG site                  | EEG site                                               | No   |
| Birthdate                 | Birthdate                                              | No   |
| EEG date                  | EEG date                                               | No   |
| Age at EEG (years)        | Age at EEG (years)                                     | No   |
| Sex at birth              | Sex at birth                                           | No   |
| Repeat Instrument         | Repeat Instrument                                      | No   |
| Genetic Status            | Genetic Status                                         | No   |
| Genetic Abnormality Type  | Genetic Abnormality Type                               | No   |
| Affected chromosome       | Affected chromosome                                    | No   |
| Full proximal boundary    | Full proximal boundary                                 | No   |
| Full distal boundary      | Full distal boundary                                   | No   |
| Human Genome Version      | Human Genome Version                                   | No   |
| family_member_type        | family_member_type                                     | Yes  |
| NVIQ_CIupr                | Estimated loss of Non-Verbal Intelligence Quotient     | Yes  |
| ORASD_upr                 | Estimated odds ratio for autism                        | Yes  |
| SRS_CIupr                 | Estimated gain of raw score of Social Responsiveness Scale | Yes  |
| PdN_CIupr                 | Estimated probability of being de novo                 | Yes  |
| sum_LOEUF_complete        | Sum LOEUF                                              | Yes  |

### Participant Codes
- M = Mother
- F = Father
- S = Sibling
- P = Proband
- O = Other (e.g., Grand-Mother)
- C = Child

## Renamed columns
**'NVIQ_CIupr': 'Estimated loss of Non-Verbal Intelligence Quotient'**  
- Verbal Intelligence Quotient - number of lost VIQ point

**'ORASD_upr': 'Estimated odds ratio for autism'**  
- Gives an estimation of ASD

**'SRS_CIupr': 'Estimated gain of raw score of Social Responsiveness Scale'**  
- Gives an estimation of SR

**'PdN_CIupr': 'Estimated probability of being de novo'**  
- Probability that the mutation is de novo (DNM)

**'sum_LOEUF_complete': 'Sum LOEUF'**
- The LOEUF score is the measure of loss-of-function observed/expected upper bound fraction
- Sum of LOEUF without correction for NON-DNM

## CNV calculation
- Using https://cnvprediction.urca.ca/index.html.
- Hg19 and Hg38
- One participant excluded for Hg18 genome version (cannot calculate CNV related risk).
- One participant excluded for lack of START/STOP boundary info (Turner only).

## Demographic Data Mapping (Q1K)

### Relation to Proband (`relation_to_proband`)
| Code | Description      |
| ---- | ---------------- |
| 1    | Yourself         |
| 2    | Parent/Caregiver |

### Household Income (`household_income`)
| Code | Description           |
| ---- | --------------------- |
| 1    | Less than $20,000     |
| 2    | $20,000 - $39,999     |
| 3    | $40,000 - $59,999     |
| 4    | $60,000 - $79,999     |
| 5    | $80,000 - $99,999     |
| 6    | $100,000 - $149,999   |
| 7    | $150,000 - $199,999   |
| 8    | $200,000 - $249,999   |
| 9    | $250,000 - $399,999   |
| 10   | >$400,000             |

### Highest Education Level (`highest_education_level`)
| Code | Description                                                   |
| ---- | ------------------------------------------------------------- |
| 1    | Elementary school or less                                     |
| 2    | Some high school                                              |
| 3    | High school diploma or certificate                            |
| 4    | Apprenticeship or other trades certificate or diploma         |
| 5    | College, CEGEP or other non-university certificate or diploma |
| 6    | Bachelor's degree                                             |
| 8    | Master's degree                                               |
| 9    | Doctorate                                                     |
| 10   | Other                                                         |

### Family Ethnicity (`family_ethnicity`)
This column is created by merging the following one-hot encoded ethnicity columns into a single comma-separated string:
`Indigenous`, `Arab`, `Black`, `Chinese`, `Filipino`, `Japanese`, `Korean`, `Latin_American`, `South_Asian`, `Southeast_Asian`, `West_Asian`, `White_Caucasian`, `Other_ethnicity`.

If a participant has no specified ethnicity, the value is set to `Unknown`.