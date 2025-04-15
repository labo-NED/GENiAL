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
    
    # Import Data
    print("Importing data files...")
    df = pd.read_csv(original_demog_genetics_data)
    dia_cogn_df = pd.read_csv(original_dia_cogn_data)
    eeg_rs_features_df = pd.read_csv(original_demog_genetics_data)
    cnv_hg19_df = pd.read_csv(cnv_prediction_hg19_data)
    cnv_hg38_df = pd.read_csv(cnv_prediction_hg38_data)
    id_map_df = pd.read_csv(id_map)

    print("Processing data...")
    # Keep only first age column and rename it
    age_cols = [col for col in df.columns if col == "Age in years"]
    df = df.rename(columns={age_cols[0]: "Age at EEG (years)"})

    # Rename columns using the mapping dictionary
    df = df.rename(columns=column_mapping)

    # Add family member type
    df['ParticipantID'] = df['ParticipantID'].astype('str')
    df['family_member_type'] = df['ParticipantID'].apply(categorize_family_member_type)

    # Merge CNV data
    print("Merging CNV data...")
    cnv_hg19_df = cnv_hg19_df.merge(id_map_df, on='ID', how='left')
    cnv_hg38_df = cnv_hg38_df.merge(id_map_df, on='ID', how='left')
    cnv_df = pd.concat([cnv_hg19_df, cnv_hg38_df], axis=0)

    # Force ParticipantID to be a string
    cnv_df['ParticipantID'] = cnv_df['ParticipantID'].astype(str).str.strip()
    df['ParticipantID'] = df['ParticipantID'].astype(str).str.strip()
    cnv_df.columns = cnv_df.columns.str.strip()
    df.columns = df.columns.str.strip()

    # Keep only the columns we want
    columns_to_keep = list(column_mapping.values()) + ['family_member_type'] + ['Record ID']
    df = df[columns_to_keep]

    # Merge by record ID
    df = df.groupby('Record ID', as_index=False).first()

    # Drop Record ID column
    df = df.drop(columns=['Record ID'])

    # Convert diagnosis and inheritance columns from Checked/Unchecked to binary 1/0
    diag_cols = [col for col in df.columns if col.startswith('diag_')]
    inheritance_cols = [col for col in df.columns if col.startswith('inheritance_')]
    for col in diag_cols + inheritance_cols:
        df[col] = df[col].map({'Checked': 1, 'Unchecked': 0})

    # Save the final preprocessed data
    print(f"Saving preprocessed data to: {preprocessed_data_path}")
    df.to_csv(preprocessed_data_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main() 