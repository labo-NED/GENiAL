# GENiAL
Genetic and EEG-based Neurodevelopmental Analysis in Autism via Learning algorithms  
This repository includes the necessary code for the GENiAL project:  
* _Question._ In children with autism spectrum disorder, can machine learning applied to EEG data differentiate those with copy number variant (CNV) risks from those without?
* _Objectives._ The primary objective of the study is to determine if CNV variants play a mediating role in delineations of EEG-based clusters amongst individuals diagnosed ASD.
* _Data._ Q1K + Brain Canada

## Create a virtual environment
```python3 -m venv venv```

## Activate Virtual Environment
```source venv/bin/activate```  

## Install packages
```pip install matplotlib mne pandas```

_Select interpreter_  
```Cmd+Shift+P```  
```Python: Select Interpreter```  
```Python 3.x.x 64-bit ('venv': venv)```  

# Steps

## STEP 0 - Clean and prepare database
1. Import `ECN-EEG-IQ-GEN-CHUSJ` from REDCap.
2. Save it in your `/Data` folder.
3. Run all sections of `GENiAL_STEP0_prep_files.ipynb` until `Prep for CNV`.
4. Run the `Prep for CNV`. This section will prepare the files you will need to input to the online CNV calculator.
5. Save the CNV outputs to the `/Data/Genetics/CNV-Output` folder.
6. Run the last section of `GENiAL_STEP0_prep_files.ipynb`.

_EEG PREPROCESSING_
1. Gather Raw EEGs and save on a HARD DRIVE.
2. Follow HAPPE preprocessing pipeline from [lab's repository]{https://github.com/labo-NED/EEG_preprocessing_pipeline}.
2.1 Run preprocessing with 2s parameters (for all power feature extraction)
2.2 Run preprocessing with 5s parameters (needed for entropy feature extraction)
 *NOTE* _This preprocessing step might take 4 days for each preprocessing (2s and 5s)._
3. Run `MATLAB/feature_extraction.m` in MATLAB. This will extract power features.
4. Run `TODO` in MATLAB. This will extract entropy features.
5. RUN `MATLAB/roi_features_to_csv.m` in MATLAB. This concatenates features per subject and ROIs.
*NOTE* _The final CSV data is saved in "TO DO"._

## STEP 1 - Preprocessing
1. Combine EEG and demog/diag data.

# Database

## Preprocessing
| Initial Item              | Renamed/processed                                      | New  |
| ------------------------- | ------------------------------------------------------ | ---- |
| ParticipantID             | ParticipantID                                          | No   |
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
- O = Other (GM)
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