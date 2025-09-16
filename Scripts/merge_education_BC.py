import pandas as pd

# Define the mapping from numeric codes to education labels (from 0 to 7)
education_labels = {
    0: "Special education",
    1: "Non-applicable (Child still in school)",
    2: "Completed mandatory school (15 years old)",
    3: "Apprenticeship, vocational education or training",
    4: "Completed high school or baccalaureate (18 years old)",
    5: "Bachelor",
    6: "Master",
    7: "Doctorate"
}

# Load the data
df = pd.read_csv('/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/Brain Canada Sources/NeurodevelopmentAsso-Highestdegree_DATA_2025-09-16_1112.csv')

# Ensure correct column names (strip whitespace, lower case, etc. if needed)
df.columns = [col.strip() for col in df.columns]

# Update record_id to add prefix 'BC_' in front of the existing record_id
if 'record_id' in df.columns:
    df['record_id'] = df['record_id'].apply(lambda x: f"BC_{x}" if pd.notnull(x) else x)

# Convert highest_degree to numeric (in case there are missing or non-numeric values)
df['highest_degree'] = pd.to_numeric(df['highest_degree'], errors='coerce')

# Compute the max education level per family and assign to all rows of that family
df['family_max_education_level'] = df.groupby('family')['highest_degree'].transform('max')

# Map the numeric codes to their corresponding education labels for family_max_education_level only
df['family_max_education_label'] = df['family_max_education_level'].map(education_labels)
# Keep only rows with family_max_education_label not empty
df = df[df['family_max_education_label'].notna()]

# Save the result with only family_max_education_label at the end
df.to_csv('/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/Brain Canada Sources/NeurodevelopmentAsso-Highestdegree_DATA_with_familymax.csv', index=False)

#############################################
## Replace highest edu from BC in final DF
# Load the proband clusters file
proband_path = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/proband_clusters_kmeans_complete_cases.csv'
proband_df = pd.read_csv(proband_path)

# Prepare a mapping from record_id to family_max_education_label
edu_map = df.set_index('record_id')['family_max_education_label'].to_dict()

# Only replace highest_education_level if a value exists in edu_map for that participant_id
def replace_if_available(row):
    new_val = edu_map.get(row['participant_id'], None)
    if pd.notnull(new_val):
        return new_val
    else:
        return row['highest_education_level']

if 'highest_education_level' in proband_df.columns:
    proband_df['highest_education_level'] = proband_df.apply(replace_if_available, axis=1)
else:
    # If the column does not exist, create it using the mapping (will be NaN if not found)
    proband_df['highest_education_level'] = proband_df['participant_id'].map(edu_map)

# Save the updated proband clusters file
final_path = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/final_cluster_db.csv'
proband_df.to_csv(final_path, index=False)

