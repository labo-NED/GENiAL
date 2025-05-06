import pandas as pd


def prepare_cnv_inputs(genetics_only_data, hg38_output, hg19_output, hg18_output):
    # CNV data - separated into hg19, hg38, and hg18
    # These will be used to input into the CNV prediction tool
    import pandas as pd
    
    genetics_df = pd.read_csv(genetics_only_data)
    genetics_df['Human Genome Version'].astype(str)
    
    selected_columns = ['Sample.ID','Sex','CHR','START','STOP','TYPE']
    df_38 = genetics_df[genetics_df['Human Genome Version'] == 'Hg38'][selected_columns]
    df_19 = genetics_df[genetics_df['Human Genome Version'] == 'Hg19'][selected_columns]
    df_18 = genetics_df[genetics_df['Human Genome Version'] == 'Hg18'][selected_columns] # Hg18, ignore
    
    # Save the DataFrame as a TSV file without the index column
    df_38.to_csv(hg38_output, sep='\t', index=False)
    df_19.to_csv(hg19_output, sep='\t', index=False)
    df_18.to_csv(hg18_output, sep='\t', index=False) # Hg18, ignore
    print('CNV input files saved.')