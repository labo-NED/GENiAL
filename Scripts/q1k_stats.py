# Function to calculate counts and percentages for each group
def get_group_stats(df, group_col, diagnosis_col):
    total = df[group_col].sum()
    count = df[(df[group_col] == 1) & (df[diagnosis_col] == 1)].shape[0]
    percentage = (count / total * 100).round(1) if total > 0 else 0
    return count, percentage, total

