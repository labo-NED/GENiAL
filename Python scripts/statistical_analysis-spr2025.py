#!/usr/bin/env python3
"""
This script performs statistical analysis on the EEG features in relation to genetic diagnosis.
It includes:
1. Data preprocessing (normalization, handling missing values)
2. Multiple regression analysis with covariates (age, sex)
3. Statistical testing and visualization
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

# Directory paths
root_dir = "/Users/emmanuelle.coutu-nadeau/Library/Mobile Documents/com~apple~CloudDocs/UdeM/MSc Psycho/LABO NED - Personal Drive/Code/GENiAL/"
preprocessed_data_path = os.path.join(root_dir, 'Data/Final/GENIAL-DB-preprocessed-V2.csv')
output_dir = os.path.join(root_dir, 'Results/Statistical_Analysis/')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the data for analysis."""
    print("Loading and preparing data...")
    
    # Load the preprocessed data
    df = pd.read_csv(preprocessed_data_path)
    
    # Identify EEG feature columns (excluding non-numeric columns)
    non_numeric_cols = ['EEG_attempted', 'EEG_site', 'EEG_date', 'EEG_age', 'EEG_Age', 'EEG_Sex']
    eeg_cols = [col for col in df.columns if col.startswith('EEG_') and col not in non_numeric_cols]
    
    # Convert sex to binary (assuming 'Sex_at_birth' is the column name)
    df['Sex_binary'] = (df['Sex_at_birth'] == 'Male').astype(int)
    
    # Ensure diagnostic columns are integers
    df['diag_control'] = df['diag_control'].astype(int)
    df['diag_neurodev'] = df['diag_neurodev'].astype(int)
    df['diag_genetic_carrier'] = df['diag_genetic_carrier'].astype(int)
    
    # Create diagnostic groups
    # 0: Control (diag_control = 1)
    # 1: Neurodev only (diag_neurodev = 1 and diag_genetic_carrier = 0)
    # 2: Genetic carrier (diag_genetic_carrier = 1)
    
    df['diagnostic_group'] = 0  # Default to control
    df.loc[(df['diag_neurodev'] == 1) & (df['diag_genetic_carrier'] == 0), 'diagnostic_group'] = 1  # Neurodev only
    df.loc[df['diag_genetic_carrier'] == 1, 'diagnostic_group'] = 2  # Genetic carrier
    
    # Print group sizes for verification
    print("\nDiagnostic group sizes:")
    print("Controls:", sum(df['diagnostic_group'] == 0))
    print("Neurodev only:", sum(df['diagnostic_group'] == 1))
    print("Genetic carriers:", sum(df['diagnostic_group'] == 2))

    # Convert EEG columns to numeric, dropping any that can't be converted
    numeric_eeg_cols = []
    for col in eeg_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Only keep columns that have at least some numeric values
            if df[col].notna().any():
                numeric_eeg_cols.append(col)
        except:
            print(f"Skipping non-numeric column: {col}")
    
    print(f"\nNumber of numeric EEG features for analysis: {len(numeric_eeg_cols)}")
    
    return df, numeric_eeg_cols

def normalize_data(df, feature_cols):
    """Normalize EEG features using z-score standardization."""
    print("Normalizing data...")
    
    scaler = StandardScaler()
    df_normalized = df.copy()
    df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df_normalized

def check_assumptions(model, X, y, feature_name):
    """Check regression assumptions and create diagnostic plots."""
    # Residuals
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(fitted_values, residuals)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # 2. Q-Q plot
    sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True, ax=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    
    # 3. Scale-Location
    axes[1, 0].scatter(fitted_values, np.sqrt(np.abs(residuals)))
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel('√|Residuals|')
    axes[1, 0].set_title('Scale-Location')
    
    # 4. Leverage plot
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    axes[1, 1].scatter(leverage, residuals)
    axes[1, 1].set_xlabel('Leverage')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Leverage')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'diagnostic_plots_{feature_name}.png'))
    plt.close()

def run_multiple_regression(df, eeg_cols):
    """Run multiple regression for each EEG feature."""
    print("Running multiple regression analysis...")
    
    results = []
    
    for feature in eeg_cols:
        # Create dummy variables for diagnostic groups (control is reference)
        diagnostic_dummies = pd.get_dummies(df['diagnostic_group'], prefix='group', drop_first=True)
        
        # Prepare the data with dummy variables
        X = pd.concat([
            df[['EEG_age', 'Sex_binary']],
            diagnostic_dummies
        ], axis=1)
        
        y = df[feature]
        
        # Add constant for statsmodels
        X = sm.add_constant(X)
        
        # Fit the model
        model = sm.OLS(y, X).fit()
        
        # Check assumptions
        check_assumptions(model, X, y, feature)
        
        # Store results
        results.append({
            'Feature': feature,
            'R_squared': model.rsquared,
            'Adj_R_squared': model.rsquared_adj,
            'F_statistic': model.fvalue,
            'F_pvalue': model.f_pvalue,
            'Neurodev_coef': model.params['group_1'],
            'Neurodev_pvalue': model.pvalues['group_1'],
            'Genetic_coef': model.params['group_2'],
            'Genetic_pvalue': model.pvalues['group_2'],
            'Age_coef': model.params['EEG_age'],
            'Age_pvalue': model.pvalues['EEG_age'],
            'Sex_coef': model.params['Sex_binary'],
            'Sex_pvalue': model.pvalues['Sex_binary']
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction for multiple comparisons (separately for each group)
    _, results_df['Neurodev_pvalue_adj'] = multipletests(results_df['Neurodev_pvalue'], 
                                                        method='fdr_bh')
    _, results_df['Genetic_pvalue_adj'] = multipletests(results_df['Genetic_pvalue'], 
                                                       method='fdr_bh')
    
    return results_df

def create_summary_visualizations(results_df, df):
    """Create summary visualizations of the results."""
    print("Creating visualizations...")
    
    # 1. Significant features plot for both groups
    plt.figure(figsize=(15, 6))
    
    # Get significant features for either group
    sig_features_neurodev = results_df[results_df['Neurodev_pvalue_adj'] < 0.05]
    sig_features_genetic = results_df[results_df['Genetic_pvalue_adj'] < 0.05]
    sig_features = pd.concat([sig_features_neurodev, sig_features_genetic]).drop_duplicates()
    
    if not sig_features.empty:
        # Create a melted dataframe for plotting
        plot_data = pd.melt(sig_features,
                           id_vars=['Feature'],
                           value_vars=['Neurodev_coef', 'Genetic_coef'],
                           var_name='Group',
                           value_name='Coefficient')
        plot_data['Group'] = plot_data['Group'].map({
            'Neurodev_coef': 'Neurodev Only',
            'Genetic_coef': 'Genetic Carrier'
        })
        
        # Create grouped bar plot
        sns.barplot(data=plot_data,
                   x='Feature',
                   y='Coefficient',
                   hue='Group',
                   palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.title('Significant EEG Features by Group (FDR corrected p < 0.05)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'significant_features.png'))
        plt.close()
    
    # 2. Correlation matrix of significant features
    if not sig_features.empty:
        sig_feature_names = sig_features['Feature'].tolist()
        corr_matrix = df[sig_feature_names].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Significant Features')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()

def save_results(results_df):
    """Save the analysis results."""
    print("Saving results...")
    
    # Save detailed results
    results_df.to_csv(os.path.join(output_dir, 'regression_results.csv'), index=False)
    
    # Create summary of significant findings
    sig_neurodev = results_df[results_df['Neurodev_pvalue_adj'] < 0.05]
    sig_genetic = results_df[results_df['Genetic_pvalue_adj'] < 0.05]
    
    # Save summary report
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write("Statistical Analysis Summary\n")
        f.write("==========================\n\n")
        
        f.write(f"Total features analyzed: {len(results_df)}\n")
        f.write(f"Features significant for Neurodevelopmental Group: {len(sig_neurodev)}\n")
        f.write(f"Features significant for Genetic group: {len(sig_genetic)}\n\n")
        
        if not sig_neurodev.empty:
            f.write("\nSignificant Features for Neurodevelopmental Group:\n")
            f.write("--------------------------------------------\n")
            for _, row in sig_neurodev.iterrows():
                f.write(f"\nFeature: {row['Feature']}\n")
                f.write(f"Coefficient: {row['Neurodev_coef']:.4f}\n")
                f.write(f"P-value (FDR corrected): {row['Neurodev_pvalue_adj']:.4e}\n")
                f.write(f"R-squared: {row['R_squared']:.4f}\n")
                f.write("Covariates:\n")
                f.write(f"  Age: coef = {row['Age_coef']:.4f}, p = {row['Age_pvalue']:.4e}\n")
                f.write(f"  Sex: coef = {row['Sex_coef']:.4f}, p = {row['Sex_pvalue']:.4e}\n")
        
        if not sig_genetic.empty:
            f.write("\nSignificant Features for Genetic Carrier Group:\n")
            f.write("------------------------------------------\n")
            for _, row in sig_genetic.iterrows():
                f.write(f"\nFeature: {row['Feature']}\n")
                f.write(f"Coefficient: {row['Genetic_coef']:.4f}\n")
                f.write(f"P-value (FDR corrected): {row['Genetic_pvalue_adj']:.4e}\n")
                f.write(f"R-squared: {row['R_squared']:.4f}\n")
                f.write("Covariates:\n")
                f.write(f"  Age: coef = {row['Age_coef']:.4f}, p = {row['Age_pvalue']:.4e}\n")
                f.write(f"  Sex: coef = {row['Sex_coef']:.4f}, p = {row['Sex_pvalue']:.4e}\n")

def main():
    # Load and prepare data
    df, eeg_cols = load_and_prepare_data()
    
    # Normalize EEG features
    df_normalized = normalize_data(df, eeg_cols)
    
    # Run multiple regression analysis
    results_df = run_multiple_regression(df_normalized, eeg_cols)
    
    # Create visualizations
    create_summary_visualizations(results_df, df_normalized)
    
    # Save results
    save_results(results_df)
    
    print("Analysis complete! Results saved in:", output_dir)

if __name__ == "__main__":
    main() 