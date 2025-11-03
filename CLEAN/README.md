# Step 1
Download from REDcap:  
*Q1K*
- ECN-BEHAVIORAL-VERBAL_IQ
- ECN_DEM-EEG-DIA-BEH-IQ-GEN  
*Brain Canada*
- Diagnosis
- ECN_BC_SRS_IQ
- Genetic_data_BC

# Step 2
Save reports in `CLEAN/Redcap_reports` folder.

# Step 3
- Update file paths in main() function of `CLEAN/scripts/preprocess_demog_beh_iq_gen.py`
- run `CLEAN/scripts/preprocess_demog_beh_iq_gen.py`
- Clean output is saved in `CLEAN/Outputs`

# Step 4
- run `CLEAN/scripts/prep_genetics_for_CNV_scores.py`
- Outputs are saved here: `CLEAN/Genetic_files_for_tool`

# Step 5
- Run output of `step 4` in CNV online tool  

| Initial Item        | Meaning                                                                             |
|---------------------|-------------------------------------------------------------------------------------|
| pLI                 | Probability that a gene is intolerant to a loss of function mutation.               |
| NVIQ_CIupr          | Estimated loss of Non-Verbal Intelligence Quotient                                  |
| ORASD_upr           | Estimated odds ratio for autism                                                     |
| SRS_CIupr           | Estimated gain of raw score of Social Responsiveness Scale                          |
| sum_LOEUF_complete  | Sum LOEUF                                                                           |

# Step 6
- Run `CLEAN/scripts/merge_genetic_scores_demog_behav.py`
- Final DB is saved here: `CLEAN/Outputs`