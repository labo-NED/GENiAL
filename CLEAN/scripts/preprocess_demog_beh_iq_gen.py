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
        'visual_disability': ['cfq_ment_visual_disability_2'],
        'Tourette_syndrome': ['cfq_ment_ts_2']
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
    Copy ethnicity values across family members (same family ID but different suffixes like _P, _F1, _M1, _S1, etc.)
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

    def extract_family_id(participant_id):
        """Extract family ID by removing family member suffixes (_P, _F1, _M1, _S1, etc.)"""
        if pd.isna(participant_id):
            return None
        
        # Remove common family member suffixes
        import re
        # Pattern matches _P, _F1, _F2, _M1, _M2, _S1, _S2, etc.
        pattern = r'_[FMS]\d*$|_P$'
        family_id = re.sub(pattern, '', str(participant_id))
        return family_id

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

    # First, get ethnicity values for each participant
    df['ethnicities'] = df.apply(get_ethnicities, axis=1)
    
    # Extract family IDs
    df['family_id'] = df['participant_id'].apply(extract_family_id)
    
    # Group by family_id and copy ethnicity values across family members
    def copy_ethnicities_across_family(group):
        # Get all non-None ethnicity values in this family
        family_ethnicities = group['ethnicities'].dropna()
        family_ethnicities = family_ethnicities[family_ethnicities != 'None']
        
        if len(family_ethnicities) > 0:
            # If there are multiple different ethnicity values, combine them
            unique_ethnicities = set()
            for eth in family_ethnicities:
                if eth and eth != 'None':
                    # Split by comma and add individual ethnicities
                    for individual_eth in eth.split(', '):
                        if individual_eth.strip():
                            unique_ethnicities.add(individual_eth.strip())
            
            # Assign the combined ethnicity to all family members
            combined_ethnicity = ', '.join(sorted(unique_ethnicities)) if unique_ethnicities else 'None'
            group['ethnicities'] = combined_ethnicity
        else:
            # If no ethnicity data, keep as 'None'
            group['ethnicities'] = 'None'
        
        return group
    
    # Apply the family ethnicity copying
    df = df.groupby('family_id', group_keys=False).apply(copy_ethnicities_across_family)
    
    # Drop the temporary family_id column
    df = df.drop(columns=['family_id'])

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

    df['oppositional_defiant_tscore'] = df[
        ['oppositional_defiant_6_18z','oppositional_defiant_probc','oppositional_defiant_6_18']
    ].bfill(axis=1).iloc[:, 0]

    cols_to_drop = [
        'srs2sch_tscore_cog_v3', 'srs2adself_tscore_cog_v2', 'srs2sch_tscore_cog_v2', 'srsps2rs_tscore_cog_v2',
        'srsps2rs_tscore_com_v2', 'srs2sch_tscore_com_v3', 'srs2adself_tscore_com_v2', 'srs2sch_tscore_com_v2',
        'srsps2rs_tscore_rrb_v2', 'srs2sch_tscore_rrb_v3', 'srs2adself_tscore_rrb_v2', 'srs2sch_tscore_rrb_v2',
        'attention_deficit_hyperactd', 'attention_deficit_hyperactz', 
        'attention_deficit_hyperactfd_ts', 'attention_deficit_hyperactv', 'cbcl_6_18_attdef_hyp_tscore',
        'oppositional_defiant_6_18z','oppositional_defiant_probc','oppositional_defiant_6_18'
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
                            
                            # ADHD measures + sleep
                            'oppositional_defiant_6_18z',
                            'oppositional_defiant_probc','oppositional_defiant_6_18',
                            'ghf_sleeping',

                            # NVIQ
                            'wais_percreas_comp',
                            'wppsi_47_fluidr_f',
                            'wisc_fri_cps',

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

                            # medication
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

def merge_nonverbal_iq_columns(df):
    """
    Merge the nonverbal IQ columns into a single nonverbal_iq column.
    There should be no overlapping values: only one of these columns
    should be non-null per row, but we pick the first non-null in wais_verbcomp_comp,
    then wisc_vci_cps, then wppsi_47_verbal_v.
    Drops the original three columns after merging.
    """
    df = df.copy()
    nonverbal_iq_cols = ['wais_percreas_comp', 'wppsi_47_fluidr_f', 'wisc_fri_cps']
    df['nonverbal_iq'] = df[nonverbal_iq_cols].bfill(axis=1).iloc[:, 0]
    cols_present = [col for col in nonverbal_iq_cols if col in df.columns]
    if cols_present:
        df = df.drop(columns=cols_present)
    return df

def merge_bc_data(df, bc_data_file, bc_diagnosis_file):
    """
    Merge the bc_data into the df.
    """
    bc_data_df = pd.read_csv(bc_data_file)
    bc_diagnosis_df = pd.read_csv(bc_diagnosis_file)
    # --- Clean and transform bc_data_df ---

    # 1. Prefix 'BC_' to record_id and participant_code
    if 'record_id' in bc_data_df.columns:
        bc_data_df['record_id'] = bc_data_df['record_id'].apply(lambda x: f"BC_{x}" if pd.notnull(x) else x)
    if 'participant_code' in bc_data_df.columns:
        bc_data_df['participant_id'] = bc_data_df['participant_code'].apply(lambda x: f"BC_{x}" if pd.notnull(x) else x)

    # 2. Drop unnecessary columns
    drop_cols = [
        'redcap_event_name',
        'eeg_resting_state_sequence'
    ]
    bc_data_df = bc_data_df.drop(columns=[col for col in drop_cols if col in bc_data_df.columns], errors='ignore')

    # 3. Combine scores for SRS social communication
    com_cols = [
        'fs_srs_communication_tscore', 'fs1_srs_communication_tscore_v3', 'fs2_srs_communication_tscore_v4',
        'fs2_srs_communication_tscore_v5', 'es_srs_communication_tscore', 'es1_srs_communication_tscore_v3',
        'es2_srs_communication_tscore_v4', 'es2_srs_communication_tscore_v5'
    ]
    present_com_cols = [col for col in com_cols if col in bc_data_df.columns]
    bc_data_df['SRS_social_communication_tscore'] = bc_data_df[present_com_cols].bfill(axis=1).iloc[:, 0] if present_com_cols else None

    # 4. Combine scores for SRS social cognition
    cog_cols = [
        'fs_srs_cognition_tscore', 'fs1_srs_cognition_tscore_v3', 'fs2_srs_cognition_tscore_v4',
        'fs2_srs_cognition_tscore_v5', 'es_srs_cognition_tscore', 'es1_srs_cognition_tscore_v3',
        'es2_srs_cognition_tscore_v4', 'es2_srs_cognition_tscore_v5'
    ]
    present_cog_cols = [col for col in cog_cols if col in bc_data_df.columns]
    bc_data_df['SRS_social_cognition_tscore'] = bc_data_df[present_cog_cols].bfill(axis=1).iloc[:, 0] if present_cog_cols else None

    # 5. Combine SRS restrictive repetitive scores
    rrb_cols = ['fs_srs_mannerisms_tscore','fs2_srs_mannerisms_tscore_v4','fs2_srs_mannerisms_tscore_v5',
                'es_srs_mannerisms_tscore','es2_srs_mannerisms_tscore_v4','es2_srs_mannerisms_tscore_v5']
    present_rrb_cols = [col for col in rrb_cols if col in bc_data_df.columns]
    bc_data_df['SRS_restrictive_repetitive_tscore'] = bc_data_df[present_rrb_cols].bfill(axis=1).iloc[:, 0] if present_rrb_cols else None

    # 6. Combine ADHD/Attention scores
    adhd_cols = ['fc2_cbcl_dsm_adhd_prob_t', 'ec2_cbcl_dsm_adhd_prob_t']
    present_adhd = [col for col in adhd_cols if col in bc_data_df.columns]
    bc_data_df['attention_deficit_hyperactivity_tscore'] = bc_data_df[present_adhd].bfill(axis=1).iloc[:, 0] if present_adhd else None

    # 7. Combine Oppositional Defiant scores
    oppo_cols = ['fc2_cbcl_dsm_oppo_prob_t', 'ec2_cbcl_dsm_oppo_prob_t']
    present_oppo = [col for col in oppo_cols if col in bc_data_df.columns]
    bc_data_df['oppositional_defiant_tscore'] = bc_data_df[present_oppo].bfill(axis=1).iloc[:, 0] if present_oppo else None

    # 8. Create 'ghf_sleeping' column from sleep_problem columns
    # sleep_problem___1 = 1 -> 0
    # sleep_problem___2 = 1 -> 1
    # sleep_problem___3 = 1 -> 2
    def consolidate_sleep(row):
        if 'sleep_problem___3' in bc_data_df.columns and pd.notnull(row.get('sleep_problem___3', None)) and row.get('sleep_problem___3', 0) == 1:
            return 2
        if 'sleep_problem___2' in bc_data_df.columns and pd.notnull(row.get('sleep_problem___2', None)) and row.get('sleep_problem___2', 0) == 1:
            return 1
        if 'sleep_problem___1' in bc_data_df.columns and pd.notnull(row.get('sleep_problem___1', None)) and row.get('sleep_problem___1', 0) == 1:
            return 0
        return None
    bc_data_df['ghf_sleeping'] = bc_data_df.apply(consolidate_sleep, axis=1)

    # 9. Keep fsiq as is
    if 'fsiq' in bc_data_df.columns:
        bc_data_df['fsiq'] = bc_data_df['fsiq']

    # 10. Combine PIQs to nonverbal_iq
    piq_cols = ['piq', 'piq_2', 'piq_3', 'piq_4']
    present_piq = [col for col in piq_cols if col in bc_data_df.columns]
    bc_data_df['nonverbal_iq'] = bc_data_df[present_piq].bfill(axis=1).iloc[:, 0] if present_piq else None

    # 11. Combine VIQs to verbal_iq
    viq_cols = ['viq', 'viq_2']
    present_viq = [col for col in viq_cols if col in bc_data_df.columns]
    bc_data_df['verbal_iq'] = bc_data_df[present_viq].bfill(axis=1).iloc[:, 0] if present_viq else None

    # 12. Combine faa_age and age_np as 'age_at_test'
    age_cols = ['faa_age', 'age_np']
    present_age = [col for col in age_cols if col in bc_data_df.columns]
    bc_data_df['age_at_test'] = bc_data_df[present_age].bfill(axis=1).iloc[:, 0] if present_age else None

    # 13. Rename gender->sex, mapping 1/2 to M/F
    if 'gender' in bc_data_df.columns:
        bc_data_df['sex'] = bc_data_df['gender'].replace({1: 'M', 2: 'F'}).fillna(bc_data_df['gender'])
        bc_data_df = bc_data_df.drop(columns=['gender'])

    # 14. Rename ethnicity->ethnicities
    if 'ethnicity' in bc_data_df.columns:
        bc_data_df.rename(columns={'ethnicity': 'ethnicities'}, inplace=True)

    # 15. schooling -> highest_education_level
    if 'schooling' in bc_data_df.columns:
        bc_data_df.rename(columns={'schooling': 'highest_education_level'}, inplace=True)

    # 16. Drop any columns not required
    keep_cols = [
        'record_id', 'participant_id',
        'age_at_test', 'sex', 'ethnicities', 'highest_education_level', 
        'nonverbal_iq', 'verbal_iq',
        'SRS_social_communication_tscore', 'SRS_social_cognition_tscore', 'SRS_restrictive_repetitive_tscore',
        'attention_deficit_hyperactivity_tscore', 'oppositional_defiant_tscore',
        'ghf_sleeping'
        
    ]
    # Only keep columns that exist in the df
    final_cols = [col for col in keep_cols if col in bc_data_df.columns]
    bc_data_df = bc_data_df[final_cols]

    # --- Clean and transform bc_diagnosis_df ---
    # Only keep record_id (prefix BC_), and drop redcap_event_name
    if 'record_id' in bc_diagnosis_df.columns:
        bc_diagnosis_df['record_id'] = bc_diagnosis_df['record_id'].apply(lambda x: f'BC_{x}' if pd.notnull(x) else x)
    if 'redcap_event_name' in bc_diagnosis_df.columns:
        bc_diagnosis_df = bc_diagnosis_df.drop(columns=['redcap_event_name'])

    # Build diagnosis column
    diagnosis_mapping = [
        ("development_delay___1", "Global developmental delay"),
        ("development_delay___2", "Motor developmental delay"),
        ("development_delay___3", "Language developmental delay"),
        ("development_delay___4", "Intellectual developmental delay"),
        ("psychiatric_troubles___1", "autism"),
        ("psychiatric_troubles___2", "adhd"),
        ("psychiatric_troubles___3", "specific learning disorder"),
        ("psychiatric_troubles___4", "schizophrenia spectrum"),
        ("psychiatric_troubles___5", "bipolar disorder"),
        ("psychiatric_troubles___6", "depressive disorder"),
        ("psychiatric_troubles___7", "anxiety disorder"),
        ("psychiatric_troubles___8", "ocd"),
        ("psychiatric_troubles___9", "sleep-wake disorder"),
        ("psychiatric_troubles___10", "sexual dysfunction"),
        ("psychiatric_troubles___11", "disruptive impulsive-control and conduct disorder"),
        ("psychiatric_troubles___12", "substance-related disorder"),
        ("psychiatric_troubles___13", "personality disorder"),
        ("psychiatric_troubles___14", "other psychiatric condition(s)"),
        ("psychiatric_troubles___15", "other psychiatric condition(s)"),
        ("epilepsy___2", "epilepsy"),
        ("epilepsy___3", "epilepsy in the past"),
        ("epilepsy___4", "epilepsy"),
    ]

    def gather_diagnoses(row):
        diagnoses = []
        for col, label in diagnosis_mapping:
            if col in row and pd.notnull(row[col]) and row[col] == 1:
                diagnoses.append(label)
        return "; ".join(diagnoses) if diagnoses else None

    bc_diagnosis_df['diagnosis'] = bc_diagnosis_df.apply(gather_diagnoses, axis=1)

    # Rename kinship to relation_to_proband
    if 'kinship' in bc_diagnosis_df.columns:
        bc_diagnosis_df.rename(columns={'kinship': 'relation_to_proband'}, inplace=True)

    # Retain only required columns: record_id, diagnosis, relation_to_proband
    needed_cols = ['record_id', 'diagnosis', 'relation_to_proband']
    bc_diagnosis_df = bc_diagnosis_df[[col for col in needed_cols if col in bc_diagnosis_df.columns]]

    # Merge bc_data_df and bc_diagnosis_df
    merged_df = pd.merge(bc_data_df, bc_diagnosis_df, on='record_id', how='left')

    # Compress merged_df so there is only 1 row per record_id by grouping and aggregating using 'first'
    merged_df = merged_df.groupby('record_id', as_index=False).first()

    # Append the new data from merged_df to df, matching column names and leaving blank where no column match
    # Get all columns from both DataFrames
    all_cols = sorted(set(df.columns) | set(merged_df.columns))
    
    # Reindex both dfs to have all columns, filling with NaN where missing
    df_aligned = df.reindex(columns=all_cols)
    merged_df_aligned = merged_df.reindex(columns=all_cols)
    
    # Concatenate both DataFrames
    return pd.concat([df_aligned, merged_df_aligned], ignore_index=True)

if __name__ == "__main__":
    # Input file path
    ## Q1K
    input_file = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/Redcap_reports/Q1K/Q1KDatabase-ECNDEMEEGDIABEHIQGEN_HSJ&MHC_2025-11-04.csv"
    beh_iq_file = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/Redcap_reports/Q1K/Q1KDatabase-ECNBEHAVIORALVERBALI_HSJ&MHC_2025-11-04.csv"
    
    ## Brain Canada
    bc_data_file = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/Redcap_reports/BrainCanada/NeurodevelopmentAsso-ECNBCSRSIQ_DATA_2025-11-03_1541.csv"
    bc_diagnosis_file = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/Redcap_reports/BrainCanada/NeurodevelopmentAsso-Diagnosis_DATA_2025-10-29_1333.csv"
    bc_genetic_file = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/Redcap_reports/BrainCanada/NeurodevelopmentAsso-GeneticdataBC_DATA_2025-10-29_1333.csv"

    # Output file path
    output_file = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/Outputs/preprocessed_Q1K_BC_HSJ&MHC_FULL_SRS_DATA.csv"
    
    # Extract specific columns & merge behavioral/iq scores
    preprocessed_df = extract_specific_columns(input_file)
    merged_df = merge_beh_iq_data(preprocessed_df, beh_iq_file)
   
    # Group by record_id and combine the data first
    grouped_preprocessed_df = merged_df.groupby('record_id').apply(combine_participant_data).reset_index(drop=True)

    # Cleanup diagnosis columns
    diagnosis_preprocessed_df = combine_diagnosis_columns(grouped_preprocessed_df)

    # Cleanup verbal IQ columns
    verbal_iq_preprocessed_df = merge_verbal_iq_columns(diagnosis_preprocessed_df)

    # Cleanup nonverbal IQ columns
    nonverbal_iq_preprocessed_df = merge_nonverbal_iq_columns(verbal_iq_preprocessed_df)

    # TODO: cleanup ethnicity, family income, relation to proband
    # Cleanup ethnicity columns
    ethnicity_preprocessed_df = combine_ethnicity_columns(nonverbal_iq_preprocessed_df)

    # Cleanup behavioral scores columns
    behavioral_preprocessed_df = clean_behavioral_scores(ethnicity_preprocessed_df)

    # Merge BC data
    bc_data_preprocessed_df = merge_bc_data(behavioral_preprocessed_df, bc_data_file, bc_diagnosis_file)

    # # Keep only participants with 3 SRS columns not empty
    # behavioral_cols = ['SRS_social_communication_tscore', 'SRS_social_cognition_tscore', 'SRS_restrictive_repetitive_tscore']
    # only_full_behavior_df = bc_data_preprocessed_df.dropna(subset=behavioral_cols, how='any')
    
    final_df = bc_data_preprocessed_df # only_full_behavior_df

    # count number of participants with age_at_test not empty
    age_at_test_mask = final_df['age_at_test'].notna()
    num_participants_with_age = age_at_test_mask.sum()
    print(f"Number of participants with age_at_test not empty: {num_participants_with_age}")

    # Reorder columns according to the specified output order
    final_column_order = [
        "participant_id",
        "record_id",
        "relation_to_proband",
        "sex",
        "test_date",
        "birthdate",
        "age_at_test",
        "status",
        "diagnosis",
        "ethnicities",
        "family_income",
        "highest_education_level",
        "SRS_restrictive_repetitive_tscore",
        "SRS_social_cognition_tscore",
        "SRS_social_communication_tscore",
        "oppositional_defiant_tscore",
        "attention_deficit_hyperactivity_tscore",
        "externalizing_behavior_tscore",
        "verbal_iq",
        "nonverbal_iq",
        "eeg_aep_done",
        "eeg_as_done",
        "eeg_fsp_done",
        "eeg_go_done",
        "eeg_mmn_done",
        "eeg_nsp_done",
        "eeg_participant_medic",
        "eeg_rs_done",
        "eeg_rsrio_done",
        "eeg_to_done",
        "eeg_vep_done",
        "eeg_vs_done",
        "general_health_form_genetic_testing_cnv_complete",
        "ghf_sleeping",
        "gt_cnv_chr",
        "gt_cnv_dist_bound",
        "gt_cnv_genver",
        "gt_cnv_prox_bound",
        "gt_cnv_status",
        "gt_snv_gene_name",
        "gt_snv_single_gene_test"
    ]

    # Keep only columns in final_column_order (if present), ignore missing ones
    final_df = final_df[[col for col in final_column_order if col in final_df.columns]]

    # count number of rows with participant_id ends with _P (handle NaNs safely)
    proband_mask = final_df['participant_id'].str.endswith('_P', na=False)
    num_probands = proband_mask.sum()
    print(f"Number of probands: {num_probands}")

    # Save output to CSV
    final_df.to_csv(output_file, index=False)