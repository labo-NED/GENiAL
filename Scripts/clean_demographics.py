import pandas as pd
import os

def _merge_ethnicities(row, ethnicity_columns):
    """Helper function to merge ethnicity columns for a single row."""
    ethnicities = [col for col in ethnicity_columns if row[col] == 1]
    return ', '.join(ethnicities) if ethnicities else 'Unknown'

def clean_demographic_data(input_path, output_path):
    """
    Cleans the demographic data by renaming columns for clarity and removing duplicates.

    Args:
        input_path (str): The path to the raw demographic data CSV file.
        output_path (str): The path to save the cleaned demographic data CSV file.
    """
    demog_df = pd.read_csv(input_path)

    # Keep only one row per participant and drop unnecessary columns
    demog_df = demog_df[demog_df['redcap_event_name'] == 'questionnaires_arm_1'].copy()
    columns_to_drop = ['redcap_event_name', 'redcap_repeat_instrument', 'redcap_repeat_instance', 'eeg_participant_code']
    demog_df.drop(columns=columns_to_drop, inplace=True)

    # Define the column mappings
    column_mapping = {
        'fbiq_q1': 'relation_to_proband',
        'fbiq_q7': 'household_income',
        'fbiq_q9___1': 'Indigenous',
        'fbiq_q9___2': 'Arab',
        'fbiq_q9___3': 'Black',
        'fbiq_q9___4': 'Chinese',
        'fbiq_q9___5': 'Filipino',
        'fbiq_q9___6': 'Japanese',
        'fbiq_q9___7': 'Korean',
        'fbiq_q9___8': 'Latin_American',
        'fbiq_q9___9': 'South_Asian',
        'fbiq_q9___10': 'Southeast_Asian',
        'fbiq_q9___11': 'West_Asian',
        'fbiq_q9___12': 'White_Caucasian',
        'fbiq_q9___13': 'Other_ethnicity',
        'fbiq_q5': 'highest_education_level'
    }

    # Rename the columns
    demog_df.rename(columns=column_mapping, inplace=True)

    # Define mappings for 'relation_to_proband' and 'household_income'
    relation_mapping = {
        1: 'Yourself',
        2: 'Parent/Caregiver'
    }
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

    # Apply the mappings
    demog_df['relation_to_proband'] = demog_df['relation_to_proband'].map(relation_mapping).fillna('NA')
    demog_df['household_income'] = demog_df['household_income'].map(income_mapping).fillna('NA')

    education_mapping = {
        1: 'Elementary school or less',
        2: 'Some high school',
        3: 'High school diploma or certificate',
        4: 'Apprenticeship or other trades certificate or diploma',
        5: 'College, CEGEP or other non-university certificate or diploma',
        6: "Bachelor's degree",
        8: "Master's degree",
        9: 'Doctorate',
        10: 'Other'
    }
    demog_df['highest_education_level'] = demog_df['highest_education_level'].map(education_mapping).fillna('NA')

    # Combine ethnicity columns into a single 'family_ethnicity' column
    ethnicity_columns = [
        'Indigenous', 'Arab', 'Black', 'Chinese', 'Filipino', 'Japanese',
        'Korean', 'Latin_American', 'South_Asian', 'Southeast_Asian',
        'West_Asian', 'White_Caucasian', 'Other_ethnicity'
    ]
    
    demog_df['family_ethnicity'] = demog_df.apply(_merge_ethnicities, axis=1, args=(ethnicity_columns,))

    # Drop the original ethnicity columns
    demog_df.drop(columns=ethnicity_columns, inplace=True)

    # Save the cleaned dataframe
    demog_df.to_csv(output_path, index=False)
    print(f"Cleaned demographic data saved to: {output_path}")

def merge_demographics_to_main(main_csv, demog_csv, output_csv):
    """
    Merges cleaned demographic columns into the main data file using record_id.
    Args:
        main_csv (str): Path to the main data file.
        demog_csv (str): Path to the cleaned demographic data file.
        output_csv (str): Path to save the merged output.
    """
    main_df = pd.read_csv(main_csv)
    demog_df = pd.read_csv(demog_csv)

    # Ensure record_id is string and stripped in both
    main_df['record_id'] = main_df['record_id'].astype(str).str.strip()
    demog_df['record_id'] = demog_df['record_id'].astype(str).str.strip()

    # Merge (left join to keep all main data)
    merged = pd.merge(main_df, demog_df, on='record_id', how='left')
    merged.to_csv(output_csv, index=False)
    print(f"Merged demographics to main data and saved to: {output_csv}")

if __name__ == '__main__':
    root_dir = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL'
    input_csv = os.path.join(root_dir, 'Data/Q1KDatabase-ECNDEMOG_DATA.csv')
    output_csv = os.path.join(root_dir, 'Data/Q1KDatabase-ECNDEMOG_DATA_cleaned.csv')
    clean_demographic_data(input_csv, output_csv) 