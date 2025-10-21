import pandas as pd

def prepare_cnv_inputs(clean_db_csv, hg19_input, hg38_input):
    # CNV data - separated into hg19, AND hg38
    # These will be used to input into the CNV prediction tool
    df = pd.read_csv(clean_db_csv)
    df = df.rename(columns={
        'gt_cnv_chr': 'CHR',
        'gt_cnv_prox_bound': 'START',
        'gt_cnv_dist_bound': 'STOP',
        'gt_cnv_genver': 'hg_version',
        'gt_cnv_status': 'TYPE'
    })
    
    selected_columns = ['record_id','sex','CHR','START','STOP','TYPE']

    # Map cnv type values: 1 = DEL, 2 = Normal, 3 = DUP
    cnv_type_map = {1: 'DEL', 2: 'Normal', 3: 'DUP'}
    df['TYPE'] = df['TYPE'].map(cnv_type_map).fillna(df['TYPE'])

    # Map hg version values: 1 = Hg19, 2 = Hg38
    def map_hg_version(val):
        if val == 1 or val == '1' or val == 1.0:
            return 'Hg19'
        elif val == 2 or val == '2' or val == 2.0:
            return 'Hg38'
        else:
            return ''  # Leave empty for now
            
    df['hg_version'] = df['hg_version'].apply(map_hg_version)
    
    # Handle empty values: if empty AND general_health_form_genetic_testing_cnv_complete == 2.0, default to Hg19
    empty_mask = (df['hg_version'] == '') | df['hg_version'].isna()
    cnv_complete_mask = df['general_health_form_genetic_testing_cnv_complete'] == 2.0
    df.loc[empty_mask & cnv_complete_mask, 'hg_version'] = 'Hg19'

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
            # Fill NaN values with 0 or drop rows with NaN values
            df_38[col] = df_38[col].fillna(0).astype(int)
            
        if col in df_19.columns:
            # Convert to numeric, then to int (to ensure no decimals)
            df_19[col] = pd.to_numeric(df_19[col], errors='coerce')
            # Fill NaN values with 0 or drop rows with NaN values
            df_19[col] = df_19[col].fillna(0).astype(int)
    
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

    # INSERT_YOUR_CODE
def main():
    # Define file paths for CNV input preparation
    clean_db_csv = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/Outputs/preprocessed_q1k_database_chusj.csv"
    hg19_input = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/Genetic_files_for_tool/Hg19.tsv"
    hg38_input = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/Genetic_files_for_tool/Hg38.tsv"

    # Run this only to prepare CNV input files for external tool
    prepare_cnv_inputs(clean_db_csv, hg19_input, hg38_input)

if __name__ == "__main__":
    main()
