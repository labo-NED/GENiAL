#!/usr/bin/env python3
"""
This script prepares input files for the CNV online prediction tool.
It separates the genetic data into different files based on Human Genome Version.
This script should be run before database_prep.py to create the CNV predictions that will be used as input.
"""

# Imports
import os
import pandas as pd

# Directory paths
root_dir = "/Users/emmanuelle.coutu-nadeau/Library/Mobile Documents/com~apple~CloudDocs/UdeM/MSc Psycho/LABO NED - Personal Drive/Code/GENiAL/"

# Input Files for CNV prediction
genetics_only_data = os.path.join(root_dir, 'Data/Genetics/Input/CNV-Analysis.csv')
hg38_input_data = os.path.join(root_dir,'Data/Genetics/Input/CNV-Analysis-Hg38.tsv')
hg19_input_data = os.path.join(root_dir,'Data/Genetics/Input/CNV-Analysis-Hg19.tsv')
hg18_input_data = os.path.join(root_dir,'Data/Genetics/Input/CNV-Analysis-Hg18.tsv')

def prepare_cnv_files():
    """Prepare input files for CNV online tool."""
    print("Reading genetic data...")
    # CNV data - separated into hg19, hg38, and hg18
    genetics_df = pd.read_csv(genetics_only_data)
    genetics_df['Human Genome Version'].astype(str)

    # Select and filter columns for each genome version
    selected_columns = ['Sample.ID','Sex','CHR','START','STOP','TYPE']
    df_38 = genetics_df[genetics_df['Human Genome Version'] == 'Hg38'][selected_columns]
    df_19 = genetics_df[genetics_df['Human Genome Version'] == 'Hg19'][selected_columns]
    df_18 = genetics_df[genetics_df['Human Genome Version'] == 'Hg18'][selected_columns]  # Hg18, ignore

    print("Saving TSV files...")
    # Save the DataFrames as TSV files without the index column
    df_38.to_csv(hg38_input_data, sep='\t', index=False)
    df_19.to_csv(hg19_input_data, sep='\t', index=False)
    df_18.to_csv(hg18_input_data, sep='\t', index=False)  # Hg18, ignore

    print(f"Files saved successfully:")
    print(f"- Hg38 data: {hg38_input_data}")
    print(f"- Hg19 data: {hg19_input_data}")
    print(f"- Hg18 data: {hg18_input_data}")

if __name__ == "__main__":
    prepare_cnv_files() 