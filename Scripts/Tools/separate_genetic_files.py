import pandas as pd
import os

def separate_genetic_files_by_family_type(input_csv, output_dir):
    """
    Separate genetic data by participant type (siblings, mothers/fathers, probands)
    and save selected columns to separate files.
    
    Args:
        input_csv (str): Path to the input CSV file
        output_dir (str): Directory to save the separated files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    print(f"Reading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Define the columns to keep, based on the image and previous context
    columns_to_keep = [
        'participant_id', 'record_id',
        'single_gene_test', 'fragil_x', 'smn1', 'panel_testing', 'panel_name', 'gene_name',
        'omim_code', 'mutation_type', 'zygosity', 'inheritance', 'sm_complete',
        'hg_version', 'CHR', 'START', 'STOP', 'TYPE', 'Genes', 'Group'
    ]
    # Only keep columns that exist in the dataframe
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    if not columns_to_keep:
        print("Error: None of the required columns found in the data")
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
        
        # Select only the columns to keep
        filtered_data = participant_data[columns_to_keep].copy()
        
        # Remove rows where all genetic columns are NaN
        # Define the genetic columns for NaN filtering
        genetic_columns = [col for col in [
            'hg_version', 'CHR', 'START', 'STOP', 'TYPE', 'Genes', 'Group',
            'single_gene_test', 'fragil_x', 'smn1', 'panel_testing', 'panel_name', 'gene_name',
            'omim_code', 'mutation_type', 'zygosity', 'inheritance', 'sm_complete'
        ] if col in filtered_data.columns]
        if genetic_columns:
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
        print(f"Columns saved: {columns_to_keep}")
    
    # Process "Other" category (participants not matching any pattern)
    print(f"\nProcessing other...")
    other_mask = ~processed_mask
    other_data = df[other_mask].copy()
    
    if len(other_data) > 0:
        # Select only the columns to keep
        filtered_data = other_data[columns_to_keep].copy()
        
        # Remove rows where all genetic columns are NaN
        genetic_columns = [col for col in [
            'hg_version', 'CHR', 'START', 'STOP', 'TYPE', 'Genes', 'Group',
            'single_gene_test', 'fragil_x', 'smn1', 'panel_testing', 'panel_name', 'gene_name',
            'omim_code', 'mutation_type', 'zygosity', 'inheritance', 'sm_complete'
        ] if col in filtered_data.columns]
        if genetic_columns:
            genetic_data_mask = filtered_data[genetic_columns].notna().any(axis=1)
            filtered_data = filtered_data[genetic_data_mask]
        
        if len(filtered_data) > 0:
            # Save to file
            output_filename = "other_genetic_data.csv"
            output_path = os.path.join(output_dir, output_filename)
            filtered_data.to_csv(output_path, index=False)
            
            print(f"Saved {len(filtered_data)} records to: {output_path}")
            print(f"Columns saved: {columns_to_keep}")
        else:
            print(f"No genetic data found for other")
    else:
        print(f"No data found for other")
    
    print(f"\nSeparation complete! Files saved to: {output_dir}")

def separate_genetic_files_by_group(input_csv, output_dir):
    """
    Separate genetic data by Group column values
    and save selected columns to separate files.
    
    Args:
        input_csv (str): Path to the input CSV file
        output_dir (str): Directory to save the separated files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    print(f"Reading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Define the columns to keep, based on the image and previous context
    columns_to_keep = [
        'participant_id', 'record_id',
        'single_gene_test', 'fragil_x', 'smn1', 'panel_testing', 'panel_name', 'gene_name',
        'omim_code', 'mutation_type', 'zygosity', 'inheritance', 'sm_complete',
        'hg_version', 'CHR', 'START', 'STOP', 'TYPE', 'Genes', 'Group'
    ]
    # Only keep columns that exist in the dataframe
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    if not columns_to_keep:
        print("Error: None of the required columns found in the data")
        return

    # Check if Group column exists
    if 'Group' not in df.columns:
        print("Error: Group column not found in the data")
        return
    
    # Get unique group values
    unique_groups = df['Group'].dropna().unique()
    
    if len(unique_groups) == 0:
        print("No Group values found in the data")
        return
    
    print(f"Found {len(unique_groups)} unique groups: {unique_groups}")
    
    # Process each group
    for group in unique_groups:
        print(f"\nProcessing group: {group}")
        
        # Filter data for this group
        group_data = df[df['Group'] == group].copy()
        
        if len(group_data) == 0:
            print(f"No data found for group: {group}")
            continue
        
        # Select only the columns to keep
        filtered_data = group_data[columns_to_keep].copy()
        
        # Remove rows where all genetic columns are NaN
        # Define the genetic columns for NaN filtering
        genetic_columns = [col for col in [
            'hg_version', 'CHR', 'START', 'STOP', 'TYPE', 'Genes', 'Group',
            'single_gene_test', 'fragil_x', 'smn1', 'panel_testing', 'panel_name', 'gene_name',
            'omim_code', 'mutation_type', 'zygosity', 'inheritance', 'sm_complete'
        ] if col in filtered_data.columns]
        if genetic_columns:
            genetic_data_mask = filtered_data[genetic_columns].notna().any(axis=1)
            filtered_data = filtered_data[genetic_data_mask]
        
        if len(filtered_data) == 0:
            print(f"No genetic data found for group: {group}")
            continue
        
        # Save to file
        output_filename = f"group_{group}_genetic_data.csv"
        output_path = os.path.join(output_dir, output_filename)
        filtered_data.to_csv(output_path, index=False)
        
        print(f"Saved {len(filtered_data)} records to: {output_path}")
        print(f"Columns saved: {columns_to_keep}")
    
    # Process "Other" category (rows with NaN Group values)
    print(f"\nProcessing rows with no Group value...")
    no_group_data = df[df['Group'].isna()].copy()
    
        
if __name__ == '__main__':
    # Define paths
    root_dir = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL'
    input_csv = os.path.join(root_dir, 'Data/Q1KDatabase-ECNEEGIQGENCHUSJ_DATA_flattened_cleaned_cnv_renamedcols_IQ_groups_demog.csv')
    output_dir = os.path.join(root_dir, 'Data/genetic_separated')
    
    # Run the separation
    separate_genetic_files_by_group(input_csv, output_dir)
