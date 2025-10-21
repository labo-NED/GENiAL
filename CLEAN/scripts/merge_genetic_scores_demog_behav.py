#!/usr/bin/env python3
"""
Script to merge genetic CNV scores from two TSV files into the preprocessed database.
Merges specified columns based on ID = record_id matching.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def merge_genetic_scores():
    """
    Merge genetic CNV scores from two TSV files into the preprocessed database.
    """
    
    # Define file paths
    base_path = Path("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN")
    
    genetic_file1 = base_path / "Genetic_cnv_scores" / "68f7f1bd7730f.tsv"
    genetic_file2 = base_path / "Genetic_cnv_scores" / "68f7f136a297d.tsv"
    target_database = base_path / "Outputs" / "preprocessed_q1k_database_chusj.csv"
    
    # Columns to merge (as specified by user)
    columns_to_merge = [
        "CHR", "START", "STOP", "Genes", "pLI", 
        "PIQ", "VIQ", "NVIQ_CIupr", "ORASD_upr", 
        "SRS_CIupr", "sum_LOEUF_complete"
    ]
    
    print("Loading genetic CNV scores files...")
    
    # Load genetic files
    try:
        df_genetic1 = pd.read_csv(genetic_file1, sep='\t')
        print(f"Loaded {genetic_file1.name}: {df_genetic1.shape[0]} rows, {df_genetic1.shape[1]} columns")
    except Exception as e:
        print(f"Error loading {genetic_file1.name}: {e}")
        return
    
    try:
        df_genetic2 = pd.read_csv(genetic_file2, sep='\t')
        print(f"Loaded {genetic_file2.name}: {df_genetic2.shape[0]} rows, {df_genetic2.shape[1]} columns")
    except Exception as e:
        print(f"Error loading {genetic_file2.name}: {e}")
        return
    
    # Load target database
    try:
        df_target = pd.read_csv(target_database)
        print(f"Loaded target database: {df_target.shape[0]} rows, {df_target.shape[1]} columns")
    except Exception as e:
        print(f"Error loading target database: {e}")
        return
    
    # Check if required columns exist in genetic files
    missing_cols1 = [col for col in columns_to_merge if col not in df_genetic1.columns]
    missing_cols2 = [col for col in columns_to_merge if col not in df_genetic2.columns]
    
    if missing_cols1:
        print(f"Warning: Missing columns in {genetic_file1.name}: {missing_cols1}")
    if missing_cols2:
        print(f"Warning: Missing columns in {genetic_file2.name}: {missing_cols2}")
    
    # Combine genetic data from both files
    print("\nCombining genetic data from both files...")
    
    # Select only the required columns and ID from each genetic file
    available_cols1 = [col for col in columns_to_merge if col in df_genetic1.columns]
    available_cols2 = [col for col in columns_to_merge if col in df_genetic2.columns]
    
    df_genetic1_subset = df_genetic1[['ID'] + available_cols1].copy()
    df_genetic2_subset = df_genetic2[['ID'] + available_cols2].copy()
    
    # Combine the dataframes (concatenate rows, not columns)
    df_genetic_combined = pd.concat([df_genetic1_subset, df_genetic2_subset], ignore_index=True)
    
    # Remove duplicates based on ID, keeping the first occurrence
    df_genetic_combined = df_genetic_combined.drop_duplicates(subset=['ID'], keep='first')
    
    print(f"Combined genetic data: {df_genetic_combined.shape[0]} unique IDs")
    
    # Merge with target database
    print("\nMerging with target database...")
    
    # Convert ID to string for consistent matching
    df_genetic_combined['ID'] = df_genetic_combined['ID'].astype(str)
    df_target['record_id'] = df_target['record_id'].astype(str)
    
    # Perform the merge
    df_merged = pd.merge(df_target, df_genetic_combined, left_on='record_id', right_on='ID', how='left')
    
    # Drop the duplicate ID column
    df_merged = df_merged.drop('ID', axis=1)
    
    print(f"Merged database: {df_merged.shape[0]} rows, {df_merged.shape[1]} columns")
    
    # Check merge results
    if available_cols1:
        merged_count = df_merged[df_merged[available_cols1[0]].notna()].shape[0]
        print(f"Records with genetic data: {merged_count}")
    else:
        print("No genetic data columns found to check merge results")
    
    # For all new columns, if 'general_health_form_genetic_testing_cnv_complete' == 2.0 but genetic scores are missing, fill with 0 or NA as specified
    # Define relevant column lists (only include columns that actually exist)
    score_columns = [
        "pLI", "NVIQ_CIupr", "ORASD_upr",
        "SRS_CIupr", "sum_LOEUF_complete"
    ]
    genes_column = "Genes"

    # Only work on columns that were actually added and exist in the merged dataframe
    available_score_columns = [col for col in score_columns if col in df_merged.columns]
    available_genes_column = genes_column if genes_column in df_merged.columns else None

    if available_score_columns:
        # Boolean mask: (form complete) AND (ALL scores missing)
        mask_form_complete = df_merged['general_health_form_genetic_testing_cnv_complete'] == 2.0

        # Mask for missing scores (all score columns are NA or, for genes, also empty string)
        if available_genes_column:
            mask_scores_missing = df_merged[available_score_columns].isnull().all(axis=1) & (
                df_merged[available_genes_column].isnull() | (df_merged[available_genes_column] == "")
            )
        else:
            mask_scores_missing = df_merged[available_score_columns].isnull().all(axis=1)

        mask_update = mask_form_complete & mask_scores_missing

        # Fill 0 to scores and "NA" to 'genes'
        for col in available_score_columns:
            df_merged.loc[mask_update, col] = 0
        if available_genes_column:
            df_merged.loc[mask_update, available_genes_column] = "NA"
    
    # delete these columns:
    columns={'gt_cnv_chr', 'gt_cnv_prox_bound', 'gt_cnv_dist_bound', 'gt_cnv_genver', 'gt_cnv_status'}
    df_merged = df_merged.drop(columns=columns)

    # Save the merged database to a new file
    output_file = base_path / "Outputs" / "preprocessed_q1k_database_chusj_with_genetic_scores.csv"
    try:
        df_merged.to_csv(output_file, index=False)
        print(f"\nSuccessfully saved merged database to: {output_file}")
        
        # Print summary of added columns
        new_columns = [col for col in df_merged.columns if col not in df_target.columns]
        print(f"Added {len(new_columns)} new columns:")
        for col in sorted(new_columns):
            non_null_count = df_merged[col].notna().sum()
            print(f"  - {col}: {non_null_count} non-null values")
            
    except Exception as e:
        print(f"Error saving merged database: {e}")
        return
    
    print("\nMerge completed successfully!")

if __name__ == "__main__":
    merge_genetic_scores()
