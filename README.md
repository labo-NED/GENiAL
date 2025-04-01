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