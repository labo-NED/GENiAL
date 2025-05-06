import pandas as pd
import numpy as np
import os

# Input and output file paths
root_dir = '/Users/emmanuelle.coutu-nadeau/Library/Mobile Documents/com~apple~CloudDocs/UdeM/MSc Psycho/LABO NED - Personal Drive/Code/GENiAL'
input_csv = os.path.join(root_dir, 'Data/Q1KDatabase-ECNEEGIQGENCHUSJ_DATA_2025-05-06_1041.csv')
output_csv = os.path.join(root_dir, 'Data/Q1KDatabase-ECNEEGIQGENCHUSJ_DATA_2025-05-06_1041_flattened.csv')

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

def flatten_csv():
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
        'eeg_sex_birth_specify'
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

if __name__ == '__main__':
    # Flatten OG CSV from REDcap
    flatten_csv()

    
