#!/usr/bin/env python3
"""
This script merges all EEG features, diagnosis, and demographic information into 1 CSV file.
Note: Run prepare_cnv_files.py first to generate the CNV prediction input files.
"""

# Imports
import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory paths
root_dir = "/Users/emmanuelle.coutu-nadeau/Library/Mobile Documents/com~apple~CloudDocs/UdeM/MSc Psycho/LABO NED - Personal Drive/Code/GENiAL/"
data_dir = os.path.join(root_dir, 'Data/')

# Original Data CSVs from REDCAP
original_demog_genetics_data = os.path.join(root_dir,'Data/Genetics/Input/Q1K report EEG_NDD_génétique.csv')
original_dia_cogn_data = os.path.join(root_dir, 'Data/Diagnosis + Cogn Tests/Q1K-Dia_cogn.csv')
original_eeg_rs_data = os.path.join(root_dir, 'Data/EEG/Q1K_concatenated_features_RS.csv')  # Preprosessed EEG data with HAPPE

# CNV prediction outputs from tool
cnv_prediction_hg19_data = os.path.join(root_dir, 'Data/Genetics/Input/cnvprediction-hg19-output.csv')
cnv_prediction_hg38_data = os.path.join(root_dir, 'Data/Genetics/Input/cnvprediction-hg38-output.csv')

# Index to Participant Code Map
id_map = os.path.join(root_dir, 'Data/Genetics/Input/sample-id-map.csv')

# Output Files
preprocessed_data_path = os.path.join(root_dir, 'Data/Final/GENIAL-DB-preprocessed-V2.csv')

# Column mapping dictionary
column_mapping = {
    'Enter in the box participant\'s EEG code as written here :  [intake_arm1][q1k_relative_idgenerated_1] [intake_arm1][q1k_proband_id_1]': 'ParticipantID',
    'Was EEG attempted?': 'EEG_attempted',
    'EEG site:': 'EEG_site',
    'Birthdate': 'Birthdate',
    'EEG Date': 'EEG_date',
    'Age at EEG (years)': 'EEG_age',
    'Sex at birth:': 'Sex_at_birth',
    'Unknown - Specify:': 'diag_unknown_specify',
    'Other - Specify:': 'diag_other_specify',
    'Medication taken the morning of the EEG': 'medication_at_EEG',
    'Resting state with Rio done?': 'RS_Rio_done',
    'Participant\'s code for resting state with Rio :': 'RS_Rio_code',
    'Resting state done?': 'RS_done',
    'Participant\'s code for resting state :': 'RS_code',
    'Tone Oddball done?': 'TO_done',
    'Participant\'s code for TO': 'TO_code',
    'GO done?': 'GO_done',
    'Participant\'s code for GO:': 'GO_code',
    'VEP done?': 'VEP_done',
    'Participant\'s code for VEP:': 'VEP_code',
    'AEP done?': 'AEP_done',
    'Participant\'s code for AEP :   Choose version A or B': 'AEP_code',
    'Randomization file used (A or B)': 'AEP_randomization_file',
    'NSP done?': 'NSP_done',
    'Participant\'s code for NSP:': 'NSP_code',
    'VS done?': 'VS_done',
    'Participant\'s code for VS:': 'VS_code',
    'MMN Oddball done?': 'MMN_done',
    'Participant\'s code for MMN': 'MMN_code',
    'Result aCGH/ LP-WGS': 'Genetic_test_result',
    'Genetic status of the participant:': 'Genetic_status',
    'Affected chromosome:': 'Affected_chromosome',
    'Full proximal boundary (e.g., 2960000):': 'Proximal_boundary',
    'Full distal boundary (e.g., 3020000):': 'Distal_boundary',
    'Please indicate the Human Genome Version used': 'Genome_version',
    'Single gene testing:': 'Single_gene_testing',
    'Fragile X': 'Fragile_X',
    'Exome / Panel testing:': 'Exome_panel_testing',
    'Diagnosis (choice=Control (no genetic or neurodev disorder))': 'diag_control',
    'Diagnosis (choice=Neurodevelopmental disorder)': 'diag_neurodev',
    'Diagnosis (choice=Genetic carrier)': 'diag_genetic_carrier',
    'Diagnosis (choice=Unknown (under investigation, suspected))': 'diag_unknown',
    'Diagnosis (choice=Other (non neurodevelopmental diagnosis))': 'diag_other',
    'Inheritance (choice=De novo)': 'inheritance_denovo',
    'Inheritance (choice=Mothers inherited)': 'inheritance_mothers_inherited',
    'Inheritance (choice=Fathers inherited)': 'inheritance_fathers_inherited',
    'Inheritance (choice=Unknown)': 'inheritance_unknown',
    'Inheritance (choice=Mosaic)': 'inheritance_mosaic'
}

def categorize_family_member_type(id_value):
    """Determine the family member type based on the ID."""
    last_part = id_value.split('_')[-1]
    if last_part == 'P':
        return 'Proband'
    elif last_part.startswith('S') and last_part[1:].isdigit():
        return 'Sibling'
    elif last_part.startswith('F') and last_part[1:].isdigit():
        return 'Father'
    elif last_part.startswith('M') and last_part[1:].isdigit():
        return 'Mother'
    elif last_part.startswith('C') and last_part[1:].isdigit():
        return 'Child'
    elif last_part.startswith('O') and last_part[1:].isdigit():
        return 'Other'
    else:
        return pd.NA

def main():
    print("Starting database preparation...")
    
    # Import Data with more robust parameters
    print("Importing data files...")
    df = pd.read_csv(original_demog_genetics_data, 
                     on_bad_lines='warn',
                     low_memory=False,
                     quoting=1,
                     encoding='utf-8')
    
    print("\nInitial columns:")
    print(df.columns.tolist())
    
    # Strip spaces from column names
    df.columns = df.columns.str.strip()
    
    print("\nColumns after stripping spaces:")
    print(df.columns.tolist())
    
    # Create a new mapping with stripped keys
    stripped_mapping = {k.strip(): v for k, v in column_mapping.items()}
    
    # Verify all columns that were renamed exist in the dataframe
    print("\nVerifying columns...")
    missing_columns = []
    for orig_col in stripped_mapping.keys():
        if orig_col not in df.columns:
            missing_columns.append(orig_col)

    if missing_columns:
        print("\nWarning: The following original columns from column_mapping are missing in the dataframe:")
        for col in missing_columns:
            print(f"- {col}")
        print("\nAvailable columns in the dataframe:")
        for col in df.columns:
            print(f"- {col}")
    
    # Drop Event Name column
    if 'Event Name' in df.columns:
        df = df.drop(columns=['Event Name'])
    
    # Merge rows with same Record ID
    if 'Record ID' in df.columns:
        print("\nMerging rows with same Record ID...")
        # First, identify the column for ParticipantID before renaming
        participant_id_col = 'Enter in the box participant\'s EEG code as written here :  [intake_arm1][q1k_relative_idgenerated_1] [intake_arm1][q1k_proband_id_1]'
        
        def merge_rows(group):
            # For ParticipantID: take first non-null value
            participant_id = group[participant_id_col].dropna().iloc[0] if not group[participant_id_col].dropna().empty else group[participant_id_col].iloc[0]
            
            # For all other columns: take the first value
            result = group.iloc[0].copy()
            result[participant_id_col] = participant_id
            return result
        
        # Group by Record ID and apply the merge function
        df = df.groupby('Record ID', as_index=False).apply(merge_rows)
    else:
        print("Warning: 'Record ID' column not found")
        print("Available columns:", df.columns.tolist())

    # Keep only first age column and rename it
    age_cols = [col for col in df.columns if col == "Age in years"]
    if age_cols:
        df = df.rename(columns={age_cols[0]: "Age at EEG (years)"})

    # Rename columns using the stripped mapping
    print("\nRenaming columns...")
    print("Before renaming, columns are:", df.columns.tolist())
    df = df.rename(columns=stripped_mapping)
    print("After renaming, columns are:", df.columns.tolist())

    # Add family member type
    df['ParticipantID'] = df['ParticipantID'].astype('str')
    df['family_member_type'] = df['ParticipantID'].apply(categorize_family_member_type)

    # Merge CNV data
    print("Merging CNV data...")
    cnv_hg19_df = cnv_hg19_df.merge(id_map_df, on='ID', how='left')
    cnv_hg38_df = cnv_hg38_df.merge(id_map_df, on='ID', how='left')
    cnv_df = pd.concat([cnv_hg19_df, cnv_hg38_df], axis=0)

    # Force ParticipantID to be a string and strip spaces
    cnv_df['ParticipantID'] = cnv_df['ParticipantID'].astype(str).str.strip()
    df['ParticipantID'] = df['ParticipantID'].astype(str).str.strip()
    cnv_df.columns = cnv_df.columns.str.strip()
    df.columns = df.columns.str.strip()

    # Keep only the columns we want that actually exist
    print("\nFinal column selection:")
    columns_to_keep = list(stripped_mapping.values()) + ['family_member_type']
    print("Desired columns:", columns_to_keep)
    available_columns = [col for col in columns_to_keep if col in df.columns]
    print("Available columns:", available_columns)
    df = df[available_columns]

    # Save the final preprocessed data
    print(f"\nSaving preprocessed data to: {preprocessed_data_path}")
    df.to_csv(preprocessed_data_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main() 