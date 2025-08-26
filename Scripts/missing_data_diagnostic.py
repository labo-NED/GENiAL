# -------------------------
# AGE, SEX, DIAGNOSTIC ANALYSIS OF MISSINGNESS
# -------------------------
print("\n" + "="*60)
print("AGE, SEX, DIAGNOSTIC ANALYSIS OF MISSINGNESS")
print("="*60)

# Define columns for analysis
age_col = 'eeg_age_years_testdate'
sex_col = 'sex'  # or 'sex_at_birth' if that's the column name

# Add missing count column if not already present
if 'missing_count' not in df.columns:
    df['missing_count'] = df[behavioral_vars].isnull().sum(axis=1)

# 1. Age analysis
print("\nMissingness by Age (eeg_age_years_testdate):")
print("Participants with missing age:", df[age_col].isnull().sum())
print("Age summary for all participants:")
print(df[age_col].describe())
print("Age summary for participants with any missing behavioral data:")
print(df.loc[df['missing_count'] > 0, age_col].describe())
print("Age summary for participants with complete behavioral data:")
print(df.loc[df['missing_count'] == 0, age_col].describe())

# 2. Sex analysis
print("\nMissingness by Sex:")
print("Sex value counts (all):")
print(df[sex_col].value_counts(dropna=False))
print("Sex value counts (missing behavioral data):")
print(df.loc[df['missing_count'] > 0, sex_col].value_counts(dropna=False))
print("Sex value counts (complete behavioral data):")
print(df.loc[df['missing_count'] == 0, sex_col].value_counts(dropna=False))

# 3. Diagnostic analysis
print("\nMissingness by Diagnosis (proportion with missing behavioral data by diagnosis):")
for diag in diagnosis_cols:
    total = df[diag].sum()
    missing = df.loc[(df[diag] == 1) & (df['missing_count'] > 0)].shape[0]
    complete = df.loc[(df[diag] == 1) & (df['missing_count'] == 0)].shape[0]
    print(f"{diag}:")
    print(f"  Total with diagnosis: {int(total)}")
    print(f"  With missing behavioral data: {missing} ({missing/(total+1e-9):.1%})")
    print(f"  With complete behavioral data: {complete} ({complete/(total+1e-9):.1%})")

# Optionally, show a cross-tab of missingness by diagnosis
print("\nCross-tab: Diagnosis vs. Missingness (any missing behavioral data)")
for diag in diagnosis_cols:
    ctab = pd.crosstab(df[diag], df['missing_count'] > 0, rownames=[diag], colnames=['Any Missing'])
    print(f"\n{diag}:\n{ctab}")

# -------------------------
# MISSING DATA ANALYSIS
# -------------------------
print("\n" + "="*60)
print("MISSING DATA PATTERN ANALYSIS")
print("="*60)

# Check for missing values
print("Missing values per column:")
missing_counts = df[behavioral_vars].isnull().sum()
print(missing_counts)
print(f"\nTotal rows: {len(df)}")
print(f"Rows with complete data: {df[behavioral_vars].dropna().shape[0]}")

# Analyze missing data patterns
print(f"\nMissing data patterns:")
print(f"Total missing values: {missing_counts.sum()}")
print(f"Percentage of data missing: {missing_counts.sum() / (len(df) * len(behavioral_vars)) * 100:.1f}%")

# Count missing values per participant
df['missing_count'] = df[behavioral_vars].isnull().sum(axis=1)
print(f"\nMissing values per participant:")
print(f"Participants with 0 missing: {(df['missing_count'] == 0).sum()}")
print(f"Participants with 1-2 missing: {((df['missing_count'] >= 1) & (df['missing_count'] <= 2)).sum()}")
print(f"Participants with 3-4 missing: {((df['missing_count'] >= 3) & (df['missing_count'] <= 4)).sum()}")
print(f"Participants with 5+ missing: {(df['missing_count'] >= 5).sum()}")

# Show participants with most missing data
print(f"\nTop 10 participants with most missing data:")
most_missing = df.nlargest(10, 'missing_count')[['participant_id', 'missing_count'] + behavioral_vars]
print(most_missing[['participant_id', 'missing_count']])

# Check if missingness is correlated between variables
print(f"\nMissing data correlation matrix:")
missing_matrix = df[behavioral_vars].isnull()
missing_corr = missing_matrix.corr()
print(missing_corr.round(3))

# Visualize missing data patterns
plt.figure(figsize=(15, 10))

# Missing data heatmap
plt.subplot(2, 2, 1)
missing_heatmap = df[behavioral_vars].isnull()
sns.heatmap(missing_heatmap.T, cbar=True, cmap='viridis', 
            xticklabels=False, yticklabels=behavioral_vars)
plt.title('Missing Data Pattern (Yellow = Missing)')
plt.xlabel('Participants')

# Missing data correlation heatmap
plt.subplot(2, 2, 2)
sns.heatmap(missing_corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
plt.title('Missing Data Correlation Matrix')

# Distribution of missing values per participant
plt.subplot(2, 2, 3)
df['missing_count'].hist(bins=range(0, len(behavioral_vars)+2), alpha=0.7, edgecolor='black')
plt.xlabel('Number of Missing Variables')
plt.ylabel('Number of Participants')
plt.title('Distribution of Missing Values per Participant')

# Missing values per variable
plt.subplot(2, 2, 4)
missing_counts.plot(kind='bar')
plt.title('Missing Values per Variable')
plt.ylabel('Number of Missing Values')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Test for systematic missingness
print(f"\nSystematic missingness analysis:")
print(f"Are there participants with ALL variables missing? {(df['missing_count'] == len(behavioral_vars)).sum()}")
print(f"Are there participants with NO variables missing? {(df['missing_count'] == 0).sum()}")

# Check if certain variables tend to be missing together
print(f"\nVariables that tend to be missing together:")
for i, var1 in enumerate(behavioral_vars):
    for j, var2 in enumerate(behavioral_vars[i+1:], i+1):
        corr = missing_corr.iloc[i, j]
        if abs(corr) > 0.3:  # Moderate correlation threshold
            print(f"  {var1} & {var2}: r = {corr:.3f}")

# -------------------------
# DETAILED MISSING DATA BY PARTICIPANT
# -------------------------
print(f"\n" + "="*60)
print("DETAILED MISSING DATA BY PARTICIPANT")
print("="*60)

# Create a detailed missing data report
missing_report = df[['participant_id'] + behavioral_vars].copy()
missing_report['total_missing'] = missing_report[behavioral_vars].isnull().sum(axis=1)

# Show participants with missing data
participants_with_missing = missing_report[missing_report['total_missing'] > 0].copy()
participants_with_missing = participants_with_missing.sort_values('total_missing', ascending=False)

print(f"\nParticipants with missing data ({len(participants_with_missing)} out of {len(df)}):")
print(f"Total participants: {len(df)}")
print(f"Participants with missing data: {len(participants_with_missing)}")
print(f"Participants with complete data: {len(df) - len(participants_with_missing)}")

# Show detailed missing data for each participant
print(f"\nDetailed missing data by participant:")
print(f"{'Participant ID':<20} {'Missing Count':<15} {'Missing Variables':<50}")
print("-" * 85)

for idx, row in participants_with_missing.iterrows():
    participant_id = row['participant_id']
    missing_count = row['total_missing']
    
    # Find which variables are missing
    missing_vars = []
    for var in behavioral_vars:
        if pd.isna(row[var]):
            missing_vars.append(var)
    
    missing_vars_str = ", ".join(missing_vars)
    print(f"{participant_id:<20} {missing_count:<15} {missing_vars_str:<50}")

# Summary statistics
print(f"\n" + "="*60)
print("MISSING DATA SUMMARY")
print("="*60)

print(f"\nMissing data by count:")
for i in range(len(behavioral_vars) + 1):
    count = (participants_with_missing['total_missing'] == i).sum()
    if count > 0:
        print(f"  {i} missing variables: {count} participants")

print(f"\nParticipants with complete data:")
complete_participants = missing_report[missing_report['total_missing'] == 0]['participant_id'].tolist()
print(f"Count: {len(complete_participants)}")
print(f"IDs: {', '.join(complete_participants)}")

print("="*60)