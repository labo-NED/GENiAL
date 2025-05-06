import pandas as pd
import numpy as np
import os

def combine_rows(group):
    # For each column, combine non-empty, non-NaN values into a single string
    combined = {}
    for col in group.columns:
        # Get all non-empty, non-NaN, non-null values as strings
        vals = group[col].dropna().astype(str)
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

    # Save the updated DataFrame to CSV
    df.to_csv(output_csv, index=False)
    print(f'Cleaned variables saved to: {output_csv}')


def merge_cnv_data(df_path, hg19_path, hg38_path, output_path):
    df = pd.read_csv(df_path)  # cleaned CSV
    hg19 = pd.read_csv(hg19_path, sep='\t')  # output from CNV tool
    hg38 = pd.read_csv(hg38_path, sep='\t')  # output from CNV tool

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

    # Save the merged dataframe
    df.to_csv(output_path, index=False)
    print(f'Merged CNV data saved to: {output_path}')


    
