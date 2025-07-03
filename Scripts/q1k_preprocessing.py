import pandas as pd
import numpy as np
import os

def combine_rows(group):
    # For each column, combine non-empty, non-NaN values into a single string
    combined = {}
    for col in group.columns:
        # Get all non-empty, non-NaN, non-null values as strings
        vals = group[col]
        # print(f"Column: {col}, Type: {type(vals)}")  # Debug print
        if isinstance(vals, pd.DataFrame):
            # print(f"DataFrame shape for {col}: {vals.shape}")
            vals = vals.iloc[:, 0]
        vals = vals.dropna().astype(str)
        vals = vals[vals.str.strip() != '']
        
        # If there are any non-empty values, take the first one
        if len(vals) > 0:
            combined[col] = vals.iloc[0]
        else:
            # If all values are empty, keep it as empty
            combined[col] = ''
            
    return pd.Series(combined)

def flatten_csv(input_csv, output_csv):
    # Read the CSV
    df = pd.read_csv(input_csv, dtype=str, on_bad_lines='skip')

    # Drop the specified columns (strip spaces in column names)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[
        'redcap_event_name',
        'redcap_repeat_instrument',
        'eeg_attempted',
        'eeg_code_software',
        'eeg_code_dig',
        'eeg_sex_birth_specify',
        'redcap_repeat_instance'
    ], errors='ignore')
    
    # Rename eeg_diagnosis columns according to the image
    diagnosis_rename = {
        'eeg_diagnosis___1': 'Control', # Control
        'eeg_diagnosis___2': 'Neurodev', # Neurodev diagn
        'eeg_diagnosis___3': 'Genetic_carrier', # Genetic carrier
        'eeg_diagnosis___4': 'Unknown', # Unknown or suspected
        'eeg_diagnosis___5': 'Other_non-neurodev' # Other non-neurodevelopmental diagnosis
    }
    df = df.rename(columns=diagnosis_rename)
    
    # Group by record_id and combine rows
    result = df.groupby('record_id', as_index=False).apply(combine_rows)
    
    # Rename eeg_participant_code to participant_id and move to first column
    if 'eeg_participant_code' in result.columns:
        result = result.rename(columns={'eeg_participant_code': 'participant_id'})
        
        # Move participant_id to the first column
        cols = list(result.columns)
        cols.insert(0, cols.pop(cols.index('participant_id')))
        result = result[cols]
    
    # Save to new CSV
    result.to_csv(output_csv, index=False)
    print(f'Flattened CSV saved to: {output_csv}')

def clean_variables(input_csv, output_csv):
    # Read CSV that has been flattened already
    df = pd.read_csv(input_csv, dtype=str, on_bad_lines='skip')

    # --------------------
    # Human Genome Version
    df = df.rename(columns={'gt_cnv_genver': 'hg_version'})

    if 'hg_version' in df.columns:
        # Convert to numeric first
        df['hg_version'] = pd.to_numeric(df['hg_version'], errors='coerce')

        # Map and update values
        df['hg_version'] = df['hg_version'].map({
            1: 'Hg19',
            2: 'Hg38'
        }, na_action='ignore').fillna('').astype(str)

    # --------------------
    # Genetic Status
    df = df.rename(columns={'gt_cnv_status': 'TYPE'})

    if 'TYPE' in df.columns:
        df['TYPE'] = pd.to_numeric(df['TYPE'], errors='coerce')
        df['TYPE'] = df['TYPE'].map({
            1: 'DEL',
            2: 'DUP',
            3: 'TRI'
        }, na_action='ignore').fillna('').astype(str)
    
    # --------------------
    # Rename CHR/START/STOP and ensure no decimals in these columns
    df = df.rename(columns={
        'gt_cnv_chr': 'CHR',
        'gt_cnv_prox_bound': 'START',
        'gt_cnv_dist_bound': 'STOP'
    })
    
    # --------------------
    # Sex at birth
    df = df.rename(columns={'eeg_sex_birth': 'sex'})
    # Convert to numeric first
    df['sex'] = pd.to_numeric(df['sex'], errors='coerce')
    df['sex'] = df['sex'].map({
            1: 'F',
            2: 'M'
        }, na_action='ignore').fillna('').astype(str)
    
    # --------------------
    
    # --------------------
    # SNV Single Gene Test
    df = df.rename(columns={'gt_snv_single_gene_test': 'single_gene_test'})
    if 'single_gene_test' in df.columns:
        df['single_gene_test'] = pd.to_numeric(df['single_gene_test'], errors='coerce')
        df['single_gene_test'] = df['single_gene_test'].map({
            1: 'yes',
            0: 'no'
        }, na_action='ignore').fillna('').astype(str)
    
    # --------------------
    # Fragile X
    df = df.rename(columns={'gt_snv_fxs': 'fragile_x'})
    if 'fragile_x' in df.columns:
        df['fragile_x'] = pd.to_numeric(df['fragile_x'], errors='coerce')
        df['fragile_x'] = df['fragile_x'].map({
            0: 'normal'
        }, na_action='ignore').fillna('').astype(str)
    
    # --------------------
    # SMN1
    df = df.rename(columns={'gt_snv_smn1': 'smn1'})
    if 'smn1' in df.columns:
        df['smn1'] = pd.to_numeric(df['smn1'], errors='coerce')
        df['smn1'] = df['smn1'].map({
            0: 'normal'
        }, na_action='ignore').fillna('').astype(str)
    
    # --------------------
    # Panel Testing
    df = df.rename(columns={'gt_snv_panel': 'panel_testing'})
    if 'panel_testing' in df.columns:
        df['panel_testing'] = pd.to_numeric(df['panel_testing'], errors='coerce')
        df['panel_testing'] = df['panel_testing'].map({
            1: 'yes',
            0: 'no'
        }, na_action='ignore').fillna('').astype(str)
    
    # --------------------
    # Panel Name (text - no mapping needed)
    df = df.rename(columns={'gt_snv_panel_name': 'panel_name'})
    
    # --------------------
    # Gene Name (text - no mapping needed)
    df = df.rename(columns={'gt_snv_gene_name': 'gene_name'})
    
    # --------------------
    # OMIM Code (text - no mapping needed)
    df = df.rename(columns={'gt_snv_omim': 'omim_code'})
    
    # --------------------
    # Drop protein column
    df = df.drop(columns=['gt_snv_protein'], errors='ignore')
    
    # --------------------
    # Mutation Type
    df = df.rename(columns={'gt_snv_mut': 'mutation_type'})
    if 'mutation_type' in df.columns:
        df['mutation_type'] = pd.to_numeric(df['mutation_type'], errors='coerce')
        df['mutation_type'] = df['mutation_type'].map({
            4: 'splicing',
            3: 'frameshift',
            2: 'nonsense',
            1: 'misense'
        }, na_action='ignore').fillna('').astype(str)
    
    # --------------------
    # Zygosity
    df = df.rename(columns={'gt_snv_zygo': 'zygosity'})
    if 'zygosity' in df.columns:
        df['zygosity'] = pd.to_numeric(df['zygosity'], errors='coerce')
        df['zygosity'] = df['zygosity'].map({
            1: 'heterozygous',
            3: 'hemizygous',
            4: 'mosaic'
        }, na_action='ignore').fillna('').astype(str)
    
    # --------------------
    # Inheritance
    df = df.rename(columns={'gt_snv_inherit': 'inheritance'})
    if 'inheritance' in df.columns:
        df['inheritance'] = pd.to_numeric(df['inheritance'], errors='coerce')
        df['inheritance'] = df['inheritance'].map({
            1: 'de_novo',
            2: "mother's inheritance",
            3: "father's inheritance",
            4: 'unknown',
            5: 'mosaic'
        }, na_action='ignore').fillna('').astype(str)
    
    # --------------------
    # SNV Complete
    df = df.rename(columns={'general_health_form_genetic_testing_snv_complete': 'snv_complete'})
    if 'snv_complete' in df.columns:
        df['snv_complete'] = pd.to_numeric(df['snv_complete'], errors='coerce')
        df['snv_complete'] = df['snv_complete'].map({
            2: 'yes'
        }, na_action='ignore').fillna('').astype(str)

    # Save the updated DataFrame to CSV
    df.to_csv(output_csv, index=False)
    print(f'Cleaned variables saved to: {output_csv}')

def rename_columns(df):
    df = df.rename(columns={
    'eeg_birthdate_v2_v2': 'birthday',
    'eeg_age_years_testdate': 'eeg_test_age',
    })

    return df

def merge_cnv_data(df_path, hg19_path, hg38_path, output_path):
    df = pd.read_csv(df_path)  # cleaned CSV
    hg19 = pd.read_csv(hg19_path, sep='\\t', engine='python')  # output from CNV tool
    hg38 = pd.read_csv(hg38_path, sep='\\t', engine='python')  # output from CNV tool

    # Clean up column names on all dataframes
    df.columns = df.columns.str.strip()
    hg19.columns = hg19.columns.str.strip().str.replace('"', '')
    hg38.columns = hg38.columns.str.strip().str.replace('"', '')

    # Drop specified columns from hg19 and hg38 dataframes
    columns_to_drop = ["CHR", "START", "STOP", "TYPE", "warningSize", "warningSegDup", "warningDNM_NDD"]
    hg19 = hg19.drop(columns=columns_to_drop, errors='ignore')
    hg38 = hg38.drop(columns=columns_to_drop, errors='ignore')
    
    # Rename ID to record_id
    hg19 = hg19.rename(columns={'ID': 'record_id'})
    hg38 = hg38.rename(columns={'ID': 'record_id'})

    # Ensure record_id columns are string type and stripped of whitespace
    df['record_id'] = df['record_id'].astype(str).str.strip()
    hg19['record_id'] = hg19['record_id'].astype(str).str.strip()
    hg38['record_id'] = hg38['record_id'].astype(str).str.strip()

    # Merge hg19 and hg38 first to combine their columns
    merged_cnv = pd.merge(hg19, hg38, on='record_id', how='outer', suffixes=('', '_drop'))
    
    # Drop any duplicate columns (those ending with _drop)
    merged_cnv = merged_cnv.loc[:, ~merged_cnv.columns.str.endswith('_drop')]
    
    # Now merge with the main dataframe
    df = pd.merge(df, merged_cnv, on='record_id', how='left')

    df = rename_columns(df)

    # Save the merged dataframe
    df.to_csv(output_path, index=False)
    print(f'Merged CNV data saved to: {output_path}')
    
# Create new diagnosis columns based on the specified logic
def create_diagnosis_columns(df, output_path):
    # ASD - check ghf_asd first, then diag_asd (2 = confirmed)
    df['ASD'] = df['ghf_asd'].fillna(df['diag_asd'].map({2: 1}).fillna(0))
    
    # ASD behavior
    df['ASD_behavior'] = df['ghf_autistic_behav']
    
    # ADHD - check ghf_adhd first, then diag_adhd (2 = confirmed)
    df['ADHD'] = df['ghf_adhd'].fillna(df['diag_adhd'].map({2: 1}).fillna(0))
    
    # ID - check ghf_id first, then diag_intel (2 = confirmed)
    df['ID'] = df['ghf_id'].fillna(df['diag_intel'].map({2: 1}).fillna(0))
    
    # OCD
    df['OCD'] = df['cfq_ment_ocd_2']
    
    # Motor disorder - combine diag_motor (2 = confirmed) and cfq_ment_ts_2
    df['motor_disorder'] = df['diag_motor'].map({2: 1}).fillna(df['cfq_ment_ts_2']).fillna(0)
    
    # Anxiety - combine ghf_anxiety and cfq_ment_ad_2
    df['anxiety'] = df['ghf_anxiety'].fillna(df['cfq_ment_ad_2'])
    
    # Neurological conditions (convert to binary: 1 if 'Yes' (1), 0 otherwise)
    # Map: '1'=Yes, all others=0
    df['neurological_conditions'] = df['ghf_neuro'].str.strip().replace('', np.nan).map({'1': 1}).fillna(0).astype('Int64')
    
    # Genetic disorder (2 = confirmed)
    # Map: '2'=Yes, all others=0
    df['genetic_disorder'] = df['diag_gene'].str.strip().replace('', np.nan).map({'2': 1}).fillna(0).astype('Int64')
    
    # Other conditions - combine all remaining diagnosis columns
    other_columns = [
        # Columns for "other" diagnosis, based on the image and context
        'cfq_ment_dd_2',                  # Depression Disorder
        'cfq_ment_bd_2',                  # Bipolar Disorder
        'cfq_ment_psyep_2',               # Psychosis Episodes
        'cfq_ment_schizo_2',              # Schizophrenia
        'cfq_ment_epilepsy_2',            # Epilepsy
        'cfq_ment_hearing_disability_2',  # Hearing disability, such as deafness
        'cfq_ment_visual_disability_2',   # Visual disability, such as blindness
        'ghf_cog_imp',                     # Cognitive impairment
        'ghf_li',                          # Learning impairment
        'ghf_ld',                          # Language disorder
        'ghf_delay_fmd',                   # Delay in motor development
        'ghf_agg_behav',                   # Aggressive behavior
        'diag_comm',                       # Communication disorder (2 = confirmed)
        'diag_hearing',                    # Hearing disability (2 = confirmed)
        'diag_visual',                     # Visual disability (2 = confirmed)
        'diag_phys',                       # Physical disability (2 = confirmed)
        'diag_oth',                        # Other (2 = confirmed)
        'diag_susp_other'                  # Suspicious other (2 = confirmed)
    ]
    
    # Convert diag_ columns to binary (2 = 1, else 0) before checking for 'other'
    for col in other_columns:
        if col.startswith('diag_'):
            df[col] = df[col].map({2: 1}).fillna(0)
    
    # Create 'other' column by checking if any of the conditions are present (1)
    # Only count as 1 if at least one condition is present, otherwise 0
    # Convert all other_columns to numeric first, then check if any equals 1
    other_df = df[other_columns].apply(lambda col: pd.to_numeric(col, errors='coerce'))
    df['other'] = (other_df == 1).any(axis=1).astype(int)
    
    # Convert all diagnosis columns to integers to ensure they're numeric
    diagnosis_columns = [
        'ASD', 'ASD_behavior', 'ADHD', 'ID', 'OCD', 'motor_disorder',
        'anxiety', 'neurological_conditions', 'genetic_disorder', 'other'
    ]
    
    for col in diagnosis_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')  # Use nullable integer type
    
    # Save the new dataframe
    df.to_csv(output_path, index=False)
    print(f'Database with new columns saved to: {output_path}')

def create_IQ_column(df, output_path):
    # Create a new column 'IQ' that takes the value from the first non-null IQ score column
    iq_columns = [
        'wais_globalapt_comp',
        'wisc_gai_is',
        'wppsi_47_gaisco_',
        'leiter3_full_iq'
    ]
    df['IQ'] = df[iq_columns].bfill(axis=1).iloc[:, 0]
    # Save the new dataframe with the IQ column
    df.to_csv(output_path, index=False)
    print(f'Database with IQ column saved to: {output_path}')

def merge_selected_cnv_columns(main_csv, tsv_path, output_csv):
    # Read main file
    main_df = pd.read_csv(main_csv)
    main_df.columns = main_df.columns.str.strip()

    # Read CNV .tsv file and clean columns
    cnv = pd.read_csv(tsv_path, sep='\t', engine='python')
    cnv.columns = cnv.columns.str.strip().str.replace('"', '')

    # Select only the columns of interest
    cols_to_keep = [
        'ID', 'Genes', 'NVIQ_CIupr', 'ORASD_upr', 'SRS_CIupr', 'PdN_CIupr', 'sum_LOEUF_complete'
    ]
    cnv = cnv[cols_to_keep]

    # Merge on ID (tsv) and record_id (main)
    main_df['record_id'] = main_df['record_id'].astype(str).str.strip()
    cnv['ID'] = cnv['ID'].astype(str).str.strip()
    merged = pd.merge(main_df, cnv, left_on='record_id', right_on='ID', how='left')

    # Rename columns
    rename_dict = {
        'NVIQ_CIupr': 'Estimated loss of Non-Verbal Intelligence Quotient',
        'ORASD_upr': 'Estimated odds ratio for autism',
        'SRS_CIupr': 'Estimated gain of raw score of Social Responsiveness Scale',
        'PdN_CIupr': 'Estimated probability of being de novo',
        'sum_LOEUF_complete': 'Sum LOEUF'
    }
    merged = merged.rename(columns=rename_dict)

    # Save
    merged.to_csv(output_csv, index=False)
    print(f"Merged and saved to: {output_csv}")

def _clean_cnv_file(path, cols_to_keep):
    """Helper function to clean and process CNV files."""
    try:
        # Try reading with different options to handle malformed quotes
        cnv = pd.read_csv(path, sep='\\t', engine='python', quoting=3)  # QUOTE_NONE
    except Exception as e:
        try:
            cnv = pd.read_csv(path, sep='\\t', engine='c', on_bad_lines='skip')
        except Exception as e2:
            cnv = pd.read_csv(path, sep='\\t', engine='python', on_bad_lines='skip', quotechar=None)
    
    cnv.columns = cnv.columns.str.strip().str.replace('"', '')
    cnv = cnv[cols_to_keep]
    
    # Clean the ID column properly - remove quotes and strip whitespace
    cnv['ID'] = cnv['ID'].astype(str).str.replace('"', '').str.strip()
    
    return cnv

def merge_selected_cnv_columns_dual(main_csv, hg19_tsv, hg38_tsv, output_csv):
    # Read main file
    main_df = pd.read_csv(main_csv, on_bad_lines='skip')
    main_df.columns = main_df.columns.str.strip()

    # Define columns to keep
    cols_to_keep = [
        'ID', 'Genes', 'NVIQ_CIupr', 'ORASD_upr', 'SRS_CIupr', 'PdN_CIupr', 'sum_LOEUF_complete'
    ]

    # Clean both CNV files
    cnv19 = _clean_cnv_file(hg19_tsv, cols_to_keep)
    cnv38 = _clean_cnv_file(hg38_tsv, cols_to_keep)

    # Merge the two CNV tables, preferring non-null values from hg38, then hg19
    cnv_merged = pd.merge(cnv19, cnv38, on='ID', how='outer', suffixes=('_19', '_38'))

    # For each column, prefer hg38 if available, else hg19
    merged_cnv = pd.DataFrame()
    merged_cnv['ID'] = cnv_merged['ID']
    merged_cnv['Genes'] = cnv_merged['Genes_38'].combine_first(cnv_merged['Genes_19'])
    merged_cnv['NVIQ_CIupr'] = cnv_merged['NVIQ_CIupr_38'].combine_first(cnv_merged['NVIQ_CIupr_19'])
    merged_cnv['ORASD_upr'] = cnv_merged['ORASD_upr_38'].combine_first(cnv_merged['ORASD_upr_19'])
    merged_cnv['SRS_CIupr'] = cnv_merged['SRS_CIupr_38'].combine_first(cnv_merged['SRS_CIupr_19'])
    merged_cnv['PdN_CIupr'] = cnv_merged['PdN_CIupr_38'].combine_first(cnv_merged['PdN_CIupr_19'])
    merged_cnv['sum_LOEUF_complete'] = cnv_merged['sum_LOEUF_complete_38'].combine_first(cnv_merged['sum_LOEUF_complete_19'])

    # Merge with main file
    main_df['record_id'] = main_df['record_id'].astype(str).str.strip()
    merged_cnv['ID'] = merged_cnv['ID'].astype(str).str.strip()
    merged = pd.merge(main_df, merged_cnv, left_on='record_id', right_on='ID', how='left')

    # Rename columns
    rename_dict = {
        'NVIQ_CIupr': 'Estimated loss of Non-Verbal Intelligence Quotient',
        'ORASD_upr': 'Estimated odds ratio for autism',
        'SRS_CIupr': 'Estimated gain of raw score of Social Responsiveness Scale',
        'PdN_CIupr': 'Estimated probability of being de novo',
        'sum_LOEUF_complete': 'Sum LOEUF'
    }
    merged = merged.rename(columns=rename_dict)

    # Drop the duplicate ID column from the merge
    if 'ID' in merged.columns:
        merged = merged.drop(columns=['ID'])

    # Save
    merged.to_csv(output_csv, index=False)
    print(f"Merged and saved to: {output_csv}")