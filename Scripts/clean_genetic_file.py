import pandas as pd

# Load the CSV file
genetic_file = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/Genetics/BC_genetic_info.csv"
df = pd.read_csv(genetic_file, dtype=str)  # Read all as string to handle empty cells
# Define required columns

# Add sex column by merging with Combined_Q1K_BC_DATA.csv on record_id
sex_file = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/Combined_Q1K_BC_DATA.csv"
sex_df = pd.read_csv(sex_file, dtype=str)[['record_id', 'sex']]

# Print how many record_ids are overlapping between the two dataframes
overlap = set(df['record_id']).intersection(set(sex_df['record_id']))
print(f"Number of overlapping record_ids between genetic file and sex file: {len(overlap)}")

df = df.merge(sex_df, on='record_id', how='left')

required_cols = [
    'record_id',
    'cnv1',
    'cnv1_chromosome',
    'cnv1_proximal',
    'cnv1_distal'
]

# Keep only rows where all required columns are non-empty and non-null
df_filtered = df.dropna(subset=required_cols)
df_filtered = df_filtered[(df_filtered[required_cols] != '').all(axis=1)]

# Map cnv1 values: 1 = DEL, 2 = Normal, 3 = DUP
cnv1_map = {'1': 'DEL', '2': 'Normal', '3': 'DUP'}
df_filtered['cnv1'] = df_filtered['cnv1'].map(cnv1_map).fillna(df_filtered['cnv1'])

# Remove rows where human_genome_version is '2' (Hg18)
df_filtered = df_filtered[(df_filtered['human_genome_version'].isna()) | (df_filtered['human_genome_version'] == '') | (df_filtered['human_genome_version'] != '2')]
# Map hg version values: Nothing = Hg19, 2 = Hg18 (but Hg18 rows are already removed)
def map_hg_version(val):
    if pd.isna(val) or val == '':
        return 'Hg19'
    else:
        return val

df_filtered['human_genome_version'] = df_filtered['human_genome_version'].apply(map_hg_version)

df_filtered = df_filtered.rename(columns={
    'cnv1': 'TYPE',
    'cnv1_chromosome': 'CHR',
    'cnv1_proximal': 'START',
    'cnv1_distal': 'STOP',
    'human_genome_version': 'hg_version'
})

# Save the filtered dataframe as CSV
df_filtered.to_csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/Genetics/BC_genetic_info_complete_cases.csv", index=False)

# Also save as TSV
df_filtered.to_csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/Genetics/CNV-Input/BC_genetic_info_complete_cases.tsv", sep='\t', index=False)
