import pandas as pd
import os

def separate_genetic_files(input_csv, output_dir):
    """
    Separate genetic data by participant type (siblings, mothers/fathers, probands)
    and save START, STOP, CHR, genes columns to separate files.
    
    Args:
        input_csv (str): Path to the input CSV file
        output_dir (str): Directory to save the separated files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    print(f"Reading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Define the genetic columns to extract
    genetic_columns = ['hg_version', 'CHR', 'START', 'STOP', 'TYPE', 'Genes', 'Group']
    
    # Check if genetic columns exist in the dataframe
    missing_columns = [col for col in genetic_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing genetic columns: {missing_columns}")
        genetic_columns = [col for col in genetic_columns if col in df.columns]
    
    if not genetic_columns:
        print("Error: No genetic columns found in the data")
        return
    
    # Define participant type patterns
    participant_patterns = {
        'siblings': '_S',
        'parents': ['_M', '_F'],
        'probands': '_P'
    }
    
    # Track which participants have been processed
    processed_mask = pd.Series([False] * len(df), index=df.index)
    
    # Process each participant type
    for participant_type, pattern in participant_patterns.items():
        print(f"\nProcessing {participant_type}...")
        
        if isinstance(pattern, list):
            # For mothers/fathers, check for both _M and _F patterns
            mask = df['participant_id'].str.contains('|'.join(pattern), na=False)
        else:
            # For siblings and probands, check for single pattern
            mask = df['participant_id'].str.contains(pattern, na=False)
        
        # Update processed mask
        processed_mask = processed_mask | mask
        
        # Filter data for this participant type
        participant_data = df[mask].copy()
        
        if len(participant_data) == 0:
            print(f"No data found for {participant_type}")
            continue
        
        # Select only the genetic columns and record_id
        columns_to_save = ['participant_id', 'record_id'] + genetic_columns
        filtered_data = participant_data[columns_to_save].copy()
        
        # Remove rows where all genetic columns are NaN
        genetic_data_mask = filtered_data[genetic_columns].notna().any(axis=1)
        filtered_data = filtered_data[genetic_data_mask]
        
        if len(filtered_data) == 0:
            print(f"No genetic data found for {participant_type}")
            continue
        
        # Save to file
        output_filename = f"{participant_type}_genetic_data.csv"
        output_path = os.path.join(output_dir, output_filename)
        filtered_data.to_csv(output_path, index=False)
        
        print(f"Saved {len(filtered_data)} records to: {output_path}")
        print(f"Columns saved: {columns_to_save}")
    
    # Process "Other" category (participants not matching any pattern)
    print(f"\nProcessing other...")
    other_mask = ~processed_mask
    other_data = df[other_mask].copy()
    
    if len(other_data) > 0:
        # Select only the genetic columns and record_id
        columns_to_save = ['participant_id', 'record_id'] + genetic_columns
        filtered_data = other_data[columns_to_save].copy()
        
        # Remove rows where all genetic columns are NaN
        genetic_data_mask = filtered_data[genetic_columns].notna().any(axis=1)
        filtered_data = filtered_data[genetic_data_mask]
        
        if len(filtered_data) > 0:
            # Save to file
            output_filename = "other_genetic_data.csv"
            output_path = os.path.join(output_dir, output_filename)
            filtered_data.to_csv(output_path, index=False)
            
            print(f"Saved {len(filtered_data)} records to: {output_path}")
            print(f"Columns saved: {columns_to_save}")
        else:
            print(f"No genetic data found for other")
    else:
        print(f"No data found for other")
    
    print(f"\nSeparation complete! Files saved to: {output_dir}")

if __name__ == '__main__':
    # Define paths
    root_dir = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL'
    input_csv = os.path.join(root_dir, 'Data/Q1KDatabase-ECNEEGIQGENCHUSJ_DATA_flattened_cleaned_cnv_renamedcols_IQ_groups_demog.csv')
    output_dir = os.path.join(root_dir, 'Data/genetic_separated')
    
    # Run the separation
    separate_genetic_files(input_csv, output_dir)
