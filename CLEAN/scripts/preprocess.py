import pandas as pd
import os

def combine_diagnosis_columns(df):
    """
    Combine the diagnosis columns into a single column.
    """

    # Define diagnosis label groups where a label may have multiple columns
    diagnosis_label_map = {
        'autism': ['ghf_asd', 'diag_asd'],
        'autism_behavior': ['ghf_autistic_behav'],
        'adhd': ['ghf_adhd', 'diag_adhd'],
        'intellectual_disability': ['ghf_id', 'diag_id', 'ghf_intel'],
        'neurological_conditions': ['ghf_neuro'],
        'anxiety': ['ghf_anxiety'],
        'cognitive_impairment': ['ghf_cog_imp'],
        'language_impairment': ['ghf_li'],
        'learning_disability': ['ghf_ld', 'diag_learn'],
        'communication_disorder': ['diag_comm'],
        'delay_fine_motor_development': ['ghf_delay_fmd'],
        'motor_disorder': ['diag_motor'],
        'aggressive_behavior': ['ghf_aggr_behav'],
        'fetal_alcohol_syndrome': ['diag_fas'],
        'hearing_disability': ['diag_hearing2'],
        'physical_disability': ['diag_phys'],
        'genetic_disorder': ['diag_gene']
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

def extract_specific_columns(isSave=True):
    """
    Extract the record_id column from the Q1K database CSV file.
    """
    # Input file path
    input_file = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/Redcap_reports/Q1KDatabase-ECNEEGIQGENCHUSJ_DATA_2025-10-09_1158.csv"
    
    # Output file path
    output_file = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/Outputs/preprocessed_Q1KDatabase-ECNEEGIQGENCHUSJ_DATA_2025-10-09.csv"
    
    try:
        # Read the CSV file
        print(f"Reading CSV file: {input_file}")
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
                            'highest_degree', # family_highest_education_level
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
                            # behavioral
                            ## DIAGNOSIS
                            'ghf_asd',         # (1 = yes)
                            'diag_asd',        # (2 = yes)
                            'ghf_adhd',        # (1 = yes)
                            'diag_adhd',       # (2 = yes)
                            'ghf_id',          # (1 = yes)
                            'diag_id',         # (2 = yes)
                            'ghf_intel',       # (2 = yes)
                            'ghf_autistic_behav',
                            'ghf_neuro',       # (neurological conditions e.g. epilepsy, seizures, movement disorder...)
                            'ghf_anxiety',
                            'ghf_cog_imp',
                            'ghf_li',          # (language impairment)
                            'diag_comm',       # (communication disorder, language disorder) (2 = yes)
                            'ghf_ld',          # (learning disability) (1 = yes)
                            'diag_learn',      # (2 = yes)
                            'ghf_delay_fmd',   # (fine motor development)
                            'ghf_aggr_behav',
                            'diag_motor',      # (Tourette, etc) (2 = yes)
                            'diag_fas',        # (fetal alc., s) (2 = yes)
                            'diag_hearing2',   # (2 = yes)
                            'diag_phys',       # (2 = yes)
                            'diag_gene',       # (genetic disorder) (2 = yes)
                            ## Other behavioral measures
                            'SRS_social_cognition_tscore',
                            'SRS_social_communication_tscore',
                            'SRS_restrictive_repetitive_tscore',
                            'attention_deficit_hyperactivity_tscore',
                            # TODO: Add sleep, impulsivity, and oppositional
                            # TODO: IQ
                            # eeg
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
                            # genetics
                            'general_health_form_genetic_testing_cnv_complete', # CVN_done  (2 = yes)
                            'gt_cnv_chr', # CHR
                            'gt_cnv_prox_bound', # START
                            'gt_cnv_dist_bound', # STOP
                            'gt_cnv_genever', # (1 = Hg19, 2 = Hg38, 3 = Hg18)
                            'gt_cnv_stats', # (1 = DEL, 2 = DUP, 3 = TRIP)
                            'gt_snv_single_gene_test', # SNV_done
                            'gt_snv_gene_name', # gene_name
                            ]].copy()

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
        
        # Group by record_id and combine the data
        preprocessed_df = preprocessed_df.groupby('record_id').apply(combine_participant_data).reset_index(drop=True)
        
        if (isSave):
            # Save to new CSV file
            preprocessed_df.to_csv(output_file, index=False)
            
            print(f"Successfully extracted record_id column")
            print(f"Output saved to: {output_file}")
            print(f"Total records: {len(preprocessed_df)}")
            print(f"Unique record_ids: {preprocessed_df['record_id'].nunique()}")
            
            # Display first few records
            print("\nFirst 10 record_ids:")
            print(preprocessed_df.head(10))
        else:
            return preprocessed_df
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_file}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
   preprocessed_df = extract_specific_columns(isSave=False)
   
   # Cleanup diagnosis columns
   diagnosis_preprocessed_df = combine_diagnosis_columns(preprocessed_df)

   # Cleanup ethnicity columns

   # Cleanup behavioral scores columns

   # Cleanup IQ

   # Cleanup genetics columns
   ## When CNV_done is 1, scores need to be 0


