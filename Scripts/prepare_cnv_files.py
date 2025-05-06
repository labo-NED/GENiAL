import pandas as pd

def prepare_cnv_inputs(clean_db_csv, hg19_input, hg38_input):
    # CNV data - separated into hg19, AND hg38
    # These will be used to input into the CNV prediction tool
    df = pd.read_csv(clean_db_csv)
    df['hg_version'].astype(str)
    
    selected_columns = ['record_id','sex','CHR','START','STOP','TYPE']
    df_38 = df[df['hg_version'] == 'Hg38'][selected_columns]
    df_19 = df[df['hg_version'] == 'Hg19'][selected_columns]
    
    # Rename the columns to match the expected format
    df_38 = df_38.rename(columns={'record_id': 'Sample.ID'})
    df_19 = df_19.rename(columns={'record_id': 'Sample.ID'})

    # Remove decimals from CHR, START, STOP if present
    for col in ['CHR', 'START', 'STOP']:
        if col in df_38.columns:
            # Convert to numeric, then to int (to ensure no decimals)
            df_38[col] = pd.to_numeric(df_38[col], errors='coerce')
            df_38[col] = df_38[col].astype(int)
            
        if col in df_19.columns:
            # Convert to numeric, then to int (to ensure no decimals)
            df_19[col] = pd.to_numeric(df_19[col], errors='coerce')
            df_19[col] = df_19[col].astype(int)
    
    # Remove rows where TYPE is 'TRI'
    df_38 = df_38[df_38['TYPE'] != 'TRI']
    df_19 = df_19[df_19['TYPE'] != 'TRI']

    # Add 'chr' prefix to chromosome numbers
    df_38['CHR'] = 'chr' + df_38['CHR'].astype(str)
    df_19['CHR'] = 'chr' + df_19['CHR'].astype(str)

    # Save the DataFrame as a TSV file
    df_38.to_csv(hg38_input, sep='\t', index=False)
    df_19.to_csv(hg19_input, sep='\t', index=False)
    print('TSV input files saved and ready for CNV calculations.')