import pandas as pd
import os

# Combine all rows per participant (record_id) into one row
# For each column, take the first non-null value across all rows for the same record_id
def combine_participant_data(group):
    """Combine multiple rows for the same participant into one row"""
    combined = {}
    for col in group.columns:
        if col == 'record_id':
            # For record_id, just take the first value (they should all be the same)
            combined[col] = group[col].iloc[0]
        else:
            # Get the first non-null value for other columns
            non_null_values = group[col].dropna()
            if len(non_null_values) > 0:
                combined[col] = non_null_values.iloc[0]
            else:
                combined[col] = None
    return pd.Series(combined)

def combine_diagnosis_columns(df):
    """
    Combine the diagnosis columns into a single column.
    """

    # Define diagnosis label groups where a label may have multiple columns
    diagnosis_label_map = {
        'autism': ['ghf_asd', 'diag_asd'],
        'autism_behavior': ['ghf_autistic_behav'],
        'adhd': ['ghf_adhd', 'diag_adhd'],
        'intellectual_disability': ['ghf_id', 'diag_intel'],
        'neurological_conditions': ['ghf_neuro'],
        'anxiety': ['ghf_anxiety', 'cfq_ment_ad_2'],
        'cognitive_impairment': ['ghf_cog_imp'],
        'language_impairment': ['ghf_li'],
        'learning_disability': ['ghf_ld', 'diag_learn'],
        'communication_disorder': ['diag_comm'],
        'delay_fine_motor_development': ['ghf_delay_fmd'],
        'motor_disorder': ['diag_motor'],
        'aggressive_behavior': ['ghf_agg_behav'],
        'fetal_alcohol_syndrome': ['diag_fas'],
        'hearing_disability': ['diag_hearing', 'cfq_ment_hearing_disability_2'],
        'physical_disability': ['diag_phys'],
        'genetic_disorder': ['diag_gene'],
        'depression_disorder': ['cfq_ment_dd_2'],
        'bipolar_disorder': ['cfq_ment_bd_2'],
        'ocd': ['cfq_ment_ocd_2'],
        'psychosis_episodes': ['cfq_ment_psyep_2'],
        'schizophrenia': ['cfq_ment_schizo_2'],
        'epilepsy': ['cfq_ment_epilepsy_2'],
        'visual_disability': ['cfq_ment_visual_disability_2']
    }

    def get_positive_diagnoses(row):
        positive_labels = []
        for label, cols in diagnosis_label_map.items():
            found_positive = False
            for col in cols:
                val = row.get(col, None)
                if pd.notnull(val):
                    if col.startswith('ghf_'):
                        # ghf_ fields: 1 = yes
                        if int(val) == 1:
                            found_positive = True
                            break
                    elif col.startswith('diag_'):
                        # diag_ fields: 2 = yes
                        if int(val) == 2:
                            found_positive = True
                            break
                    elif col.startswith('cfq_ment_'):
                        # cfq_ment_ fields: 1 = yes, 0 = no
                        if int(val) == 1:
                            found_positive = True
                            break
            if found_positive:
                positive_labels.append(label)
        return ', '.join(positive_labels) if positive_labels else 'None'

    df['diagnosis'] = df.apply(get_positive_diagnoses, axis=1)

    # Remove the initial diagnosis columns (the columns mapped in diagnosis_label_map)
    columns_to_remove = set()
    for cols in diagnosis_label_map.values():
        columns_to_remove.update(cols)
    # Only remove columns that actually exist in the DataFrame
    existing_to_remove = [col for col in columns_to_remove if col in df.columns]
    df = df.drop(columns=existing_to_remove)

    return df

def combine_ethnicity_columns(df):
    """
    Combine ethnicity checkbox columns into a single comma-separated label column
    and drop the original checkbox columns.
    Fix: Don't use row.get/col in row, but use explicit column access.
    """

    ethnicity_label_map = {
        'Indigenous': 'fbiq_q9___1',
        'Arab': 'fbiq_q9___2',
        'Black': 'fbiq_q9___3',
        'Chinese': 'fbiq_q9___4',
        'Filipino': 'fbiq_q9___5',
        'Japanese': 'fbiq_q9___6',
        'Korean': 'fbiq_q9___7',
        'Latin_American': 'fbiq_q9___8',
        'South_Asian': 'fbiq_q9___9',
        'Southeast_Asian': 'fbiq_q9___10',
        'West_Asian': 'fbiq_q9___11',
        'White_Caucasian': 'fbiq_q9___12',
        'Other_ethnicity': 'fbiq_q9___13',
    }

    # Make sure all relevant columns are present
    for col in ethnicity_label_map.values():
        if col not in df.columns:
            df[col] = pd.NA

    def get_ethnicities(row):
        labels = []
        for label, col in ethnicity_label_map.items():
            val = row[col]
            if pd.notnull(val):
                try:
                    is_checked = int(val) == 1
                except (ValueError, TypeError):
                    is_checked = str(val).strip() == '1'
                if is_checked:
                    labels.append(label)
        return ', '.join(labels) if labels else 'None'

    df['ethnicities'] = df.apply(get_ethnicities, axis=1)

    # Drop original checkbox columns if present
    checkbox_cols = [col for col in ethnicity_label_map.values() if col in df.columns]
    if checkbox_cols:
        df = df.drop(columns=checkbox_cols)

    return df

def clean_behavioral_scores(df_original):
    """
    Clean the behavioral scores columns.
    """
    df = df_original.copy()

    # 1. SRS Social Cognition T Score
    df['SRS_social_cognition_tscore'] = df[
        ['srs2sch_tscore_cog_v3', 'srs2sch_tscore_cog_v3', 'srs2adself_tscore_cog_v2', 'srs2sch_tscore_cog_v2', 'srsps2rs_tscore_cog_v2'] 
    ].bfill(axis=1).iloc[:, 0]

    # 2. SRS Social Communication T Score
    df['SRS_social_communication_tscore'] = df[
        ['srsps2rs_tscore_com_v2', 'srs2sch_tscore_com_v3', 'srs2adself_tscore_com_v2', 'srs2sch_tscore_com_v2']
    ].bfill(axis=1).iloc[:, 0]

    # 3. SRS Restrictive & Repetitive T Score
    df['SRS_restrictive_repetitive_tscore'] = df[
        ['srsps2rs_tscore_rrb_v2', 'srs2sch_tscore_rrb_v3', 'srs2adself_tscore_rrb_v2', 'srs2sch_tscore_rrb_v2']
    ].bfill(axis=1).iloc[:, 0]

    # Attention deficit / hyperactivity
    df['attention_deficit_hyperactivity_tscore'] = df[
        ['attention_deficit_hyperactd', 'attention_deficit_hyperactz', 
        'attention_deficit_hyperactfd_ts', 'attention_deficit_hyperactfd_ts',
        'attention_deficit_hyperactv', 'cbcl_6_18_attdef_hyp_tscore']
    ].bfill(axis=1).iloc[:, 0]

    cols_to_drop = [
        'srs2sch_tscore_cog_v3', 'srs2adself_tscore_cog_v2', 'srs2sch_tscore_cog_v2', 'srsps2rs_tscore_cog_v2',
        'srsps2rs_tscore_com_v2', 'srs2sch_tscore_com_v3', 'srs2adself_tscore_com_v2', 'srs2sch_tscore_com_v2',
        'srsps2rs_tscore_rrb_v2', 'srs2sch_tscore_rrb_v3', 'srs2adself_tscore_rrb_v2', 'srs2sch_tscore_rrb_v2',
        'attention_deficit_hyperactd', 'attention_deficit_hyperactz', 
        'attention_deficit_hyperactfd_ts', 'attention_deficit_hyperactv', 'cbcl_6_18_attdef_hyp_tscore'
    ]
    cols_present = [col for col in cols_to_drop if col in df.columns]
    if cols_present:
        df = df.drop(columns=cols_present)

    return df

def extract_specific_columns(input_file):
    """
    Extract the record_id column from the Q1K database CSV file.
    """
    
    # Read the CSV files
    df = pd.read_csv(input_file)
    
    # Extract columns from Q1K Database
    preprocessed_df = df[[
                            # demog
                            'record_id', 
                            'eeg_participant_code', # participant_id
                            'eeg_birthdate_v2_v2', # birthdate
                            'eeget_date_v2_v2', # test_date
                            'eeg_age_years_testdate', # age_at_test
                            'eeg_sex_birth', # sex
                            'fbiq_q9___1', #'Indigenous'
                            'fbiq_q9___2', #'Arab'
                            'fbiq_q9___3', #'Black'
                            'fbiq_q9___4', #'Chinese'
                            'fbiq_q9___5', #'Filipino'
                            'fbiq_q9___6', #'Japanese'
                            'fbiq_q9___7', #'Korean'
                            'fbiq_q9___8', #'Latin_American'
                            'fbiq_q9___9', #'South_Asian'
                            'fbiq_q9___10', #'Southeast_Asian'
                            'fbiq_q9___11', #'West_Asian'
                            'fbiq_q9___12', #'White_Caucasian'
                            'fbiq_q9___13', #'Other_ethnicity'
                            'fbiq_q5', #'highest_education_level'
                            'fbiq_q7', # family_income
                            'fbiq_q1', # relation_to_proband
                            
                            ## DIAGNOSIS
                            'ghf_asd',         # (1 = yes)
                            'diag_asd',        # (2 = yes)
                            'ghf_adhd',        # (1 = yes)
                            'diag_adhd',       # (2 = yes)
                            'ghf_id',          # (1 = yes)
                            'diag_intel',      # (2 = yes)
                            'ghf_autistic_behav',
                            'ghf_neuro',       # (neurological conditions e.g. epilepsy, seizures, movement disorder...)
                            'ghf_anxiety',
                            'ghf_cog_imp',
                            'ghf_li',          # (language impairment)
                            'diag_comm',       # (communication disorder, language disorder) (2 = yes)
                            'ghf_ld',          # (learning disability) (1 = yes)
                            'diag_learn',      # (2 = yes)
                            'ghf_delay_fmd',   # (fine motor development)
                            'ghf_agg_behav',
                            'diag_motor',      # (Tourette, etc) (2 = yes)
                            'cfq_ment_ts_2',   # Tourette's Syndrome (1 = Yes, 0 = No)
                            'diag_fas',        # (fetal alc., s) (2 = yes)
                            'diag_hearing',    # (2 = yes)
                            'cfq_ment_hearing_disability_2', # Hearing Disability (1 = Yes, 0 = No)
                            'diag_phys',       # (2 = yes)
                            'diag_gene',       # (genetic disorder) (2 = yes)
                            'cfq_ment_dd_2',       # Depression Disorder (1 = Yes, 0 = No)
                            'cfq_ment_ad_2',       # Anxiety Disorder (1 = Yes, 0 = No)
                            'cfq_ment_bd_2',       # Bipolar Disorder (1 = Yes, 0 = No)
                            'cfq_ment_ocd_2',      # Obsessive Compulsive Disorder (1 = Yes, 0 = No)
                            'cfq_ment_psyep_2',    # Psychosis Episodes (1 = Yes, 0 = No)
                            'cfq_ment_schizo_2',   # Schizophrenia (1 = Yes, 0 = No)
                            'cfq_ment_epilepsy_2', # Epilepsy (1 = Yes, 0 = No)
                            'cfq_ment_visual_disability_2',  # Visual Disability (1 = Yes, 0 = No)
                            
                            # TODO: Add sleep, impulsivity, and oppositional
                            # TODO: IQ
                            # EEG
                            'eeg_rsrio_done',
                            'eeg_rs_done',
                            'eeg_to_done',
                            'eeg_go_done',
                            'eeg_vep_done',
                            'eeg_aep_done',
                            'eeg_nsp_done',
                            'eeg_vs_done',
                            'eeg_as_done',
                            'eeg_fsp_done',
                            'eeg_mmn_done',
                            'eeg_participant_medic',
                            
                            # Genetics
                            'general_health_form_genetic_testing_cnv_complete', # CVN_done  (2 = yes)
                            'gt_cnv_chr', # CHR
                            'gt_cnv_prox_bound', # START
                            'gt_cnv_dist_bound', # STOP
                            'gt_cnv_genver', # (1 = Hg19, 2 = Hg38, 3 = Hg18)
                            'gt_cnv_status', # (1 = DEL, 2 = DUP, 3 = TRIP)
                            'gt_snv_single_gene_test', # SNV_done
                            'gt_snv_gene_name', # gene_name
                            ]].copy()

    # Map values in fbiq_q7 (family_income) and fbiq_q1 (relation_to_proband) to their descriptive labels
    income_mapping = {
        1: 'Less than $20,000',
        2: '$20,000 - $39,999',
        3: '$40,000 - $59,999',
        4: '$60,000 - $79,999',
        5: '$80,000 - $99,999',
        6: '$100,000 - $149,999',
        7: '$150,000 - $199,999',
        8: '$200,000 - $249,999',
        9: '$250,000 - $399,999',
        10: '>$400,000'
    }
    relation_mapping = {
        1: 'Yourself',
        2: 'Parent/Caregiver'
    }

    # Only map and replace values that are present in the mapping, leave other values (e.g., nan) untouched
    if 'fbiq_q7' in preprocessed_df.columns:
        preprocessed_df['fbiq_q7'] = preprocessed_df['fbiq_q7'].map(income_mapping)
    if 'fbiq_q1' in preprocessed_df.columns:
        preprocessed_df['fbiq_q1'] = preprocessed_df['fbiq_q1'].map(relation_mapping)

    # Map column names to their corresponding labels
    status_map = {
        'eeg_diagnosis___1': 'Control',
        'eeg_diagnosis___2': 'Neurodevelopmental disorder',
        'eeg_diagnosis___3': 'Genetic carrier',
        'eeg_diagnosis___4': 'Unknown (under investigation, suspected)',
        'eeg_diagnosis___5': 'Other (non-neurodevelopmental diagnosis)'
    }

    # Assign the status column, allowing for multiple statuses per row
    def get_statuses(row):
        statuses = []
        for col, label in status_map.items():
            if col in row and row[col] == 1:
                statuses.append(label)
        return ', '.join(statuses) if statuses else None

    preprocessed_df['status'] = df.apply(get_statuses, axis=1)
    
    # Update demographics columns to new names for clarity
    column_rename_map = {
        'eeg_participant_code': 'participant_id',
        'eeg_birthdate_v2_v2': 'birthdate',
        'eeget_date_v2_v2': 'test_date',
        'eeg_age_years_testdate': 'age_at_test',
        'eeg_sex_birth': 'sex',
        'fbiq_q5': 'highest_education_level',
        'fbiq_q7': 'family_income',
        'fbiq_q1': 'relation_to_proband',
        'highest_degree': 'family_highest_education_level',
    }
    preprocessed_df.rename(columns=column_rename_map, inplace=True)
    
    return preprocessed_df

def merge_beh_iq_data(preprocessed_df, beh_iq_file):
    """
    Merge the beh_iq_data into the preprocessed_df.
    """
    beh_iq_df = pd.read_csv(beh_iq_file)

    # Extract columns from Q1K Database
    merged_df = beh_iq_df[[
        'record_id',

        # Behavioral measures
        'srs2sch_tscore_cog_v3', 'srs2adself_tscore_cog_v2', 'srs2sch_tscore_cog_v2', 'srsps2rs_tscore_cog_v2', # 'SRS_social_cognition_tscore',
        'srsps2rs_tscore_com_v2', 'srs2sch_tscore_com_v3', 'srs2adself_tscore_com_v2', 'srs2sch_tscore_com_v2',# 'SRS_social_communication_tscore',
        'srsps2rs_tscore_rrb_v2', 'srs2sch_tscore_rrb_v3', 'srs2adself_tscore_rrb_v2', 'srs2sch_tscore_rrb_v2', # 'SRS_restrictive_repetitive_tscore',
        'attention_deficit_hyperactd', 'attention_deficit_hyperactz', 'attention_deficit_hyperactfd_ts', 'attention_deficit_hyperactv', 'cbcl_6_18_attdef_hyp_tscore', # 'attention_deficit_hyperactivity_tscore',
        
        # Verbal IQ
        'wais_verbcomp_comp','wisc_vci_cps','wppsi_47_verbal_v'
    ]].copy()

    return pd.merge(preprocessed_df, merged_df, on='record_id', how='left')

def merge_verbal_iq_columns(df):
    """
    Merge the verbal IQ columns into a single verbal_iq column.
    There should be no overlapping values: only one of these columns
    should be non-null per row, but we pick the first non-null in wais_verbcomp_comp,
    then wisc_vci_cps, then wppsi_47_verbal_v.
    Drops the original three columns after merging.
    """
    df = df.copy()
    verbal_iq_cols = ['wais_verbcomp_comp', 'wisc_vci_cps', 'wppsi_47_verbal_v']
    df['verbal_iq'] = df[verbal_iq_cols].bfill(axis=1).iloc[:, 0]
    cols_present = [col for col in verbal_iq_cols if col in df.columns]
    if cols_present:
        df = df.drop(columns=cols_present)
    return df

if __name__ == "__main__":
    # Input file path
    # input_file = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/Redcap_reports/Q1KDatabase-ECNDEMEEGDIABEHIQGEN_DATA_2025-10-21_1139.csv"
    input_file = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/Redcap_reports/Q1KDatabase-ECNDEMEEGDIABEHIQGEN_DATA_2025-10-21_1209.csv"
    beh_iq_file = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/Redcap_reports/Q1KDatabase-ECNBEHAVIORALVERBALI_DATA_2025-10-21_1431.csv"

    # Output file path
    output_file = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/Outputs/preprocessed_q1k_database_chusj.csv"
    
    # Extract specific columns & merge behavioral/iq scores
    preprocessed_df = extract_specific_columns(input_file)
    merged_df = merge_beh_iq_data(preprocessed_df, beh_iq_file)
   
    # Group by record_id and combine the data first
    grouped_preprocessed_df = merged_df.groupby('record_id').apply(combine_participant_data).reset_index(drop=True)

    # Cleanup diagnosis columns
    diagnosis_preprocessed_df = combine_diagnosis_columns(grouped_preprocessed_df)

    # Cleanup verbal IQ columns
    verbal_iq_preprocessed_df = merge_verbal_iq_columns(diagnosis_preprocessed_df)

    # TODO: cleanup ethnicity, family income, relation to proband
    # Cleanup ethnicity columns
    ethnicity_preprocessed_df = combine_ethnicity_columns(verbal_iq_preprocessed_df)

    # Cleanup behavioral scores columns
    behavioral_preprocessed_df = clean_behavioral_scores(ethnicity_preprocessed_df)
    
    # Cleanup genetics columns
    ## When CNV_done is 1, scores need to be 0

    final_df = behavioral_preprocessed_df

    # Save output to CSV
    final_df.to_csv(output_file, index=False)


## ARCHIVE - FOR LATER
# # Count rows where medication column is not empty, not NA, not None, as a single count
#     # Some possible medication "empty" values have "" around them in the csv, and some don't.
#     # Normalize by stripping leading/trailing spaces and any surrounding quotes, and .lower() for comparison.
#     def clean_cell(val):
#         if pd.isnull(val):
#             return None
#         val = str(val).strip().strip('"').strip("'")
#         return val.lower()

#     EMPTY_VALUES = {
#         '', 'na', 'n/a', 'none', 'no', 'aucun', '-', 'aucun', ' -'
#     }

#     eeg_medication_count = ethnicity_preprocessed_df[
#         ethnicity_preprocessed_df['eeg_participant_medic'].apply(
#             lambda x: (clean_cell(x) not in EMPTY_VALUES and clean_cell(x) not in {'aucun', ' -'})
#         )
#     ].shape[0]
#     print(f"EEG medication count: {eeg_medication_count}")

#     eeg_medication_voirdossier_count = ethnicity_preprocessed_df[
#         ethnicity_preprocessed_df['eeg_participant_medic'].apply(
#             lambda x: clean_cell(x) in {'voir dossier', 'yes see files'}
#         )
#     ].shape[0]
#     print(f"EEG -voir dossier- count: {eeg_medication_voirdossier_count}")

#     # Total participant count
#     participant_count = ethnicity_preprocessed_df['record_id'].nunique()
#     print(f"Total participant count: {participant_count}")