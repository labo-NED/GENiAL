#!/usr/bin/env python3
"""
Statistical Analysis Script for GENiAL Project
Performs comprehensive analyses including:
- Range checks and descriptive statistics
- Normality tests and distribution visualizations
- MANCOVA analysis
- Group means comparisons
- EEG features significance testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, f_oneway, kruskal
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# File path
DATA_PATH = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/merged_clustered_EEG_features_RSRio_NOV_24_2025.csv"
# DATA_PATH = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/merged_clustered_EEG_features_by_roi_RSRio_NOV_24_2025.csv"
OUTPUT_DIR = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/Stats/"

# Variables for analysis
RANGE_CHECK_VARS = [
    'age_at_test', 
    'SRS_restrictive_repetitive_tscore', 
    'SRS_social_communication_tscore', 
    'SRS_social_cognition_tscore', 
    'attention_deficit_hyperactivity_tscore', 
    'nonverbal_iq'
]

MANCOVA_DVS = [
    'SRS_restrictive_repetitive_tscore',
    'SRS_social_communication_tscore',
    'SRS_social_cognition_tscore',
    'attention_deficit_hyperactivity_tscore',
    'nonverbal_iq'
]

MANCOVA_IV = 'cluster'
MANCOVA_COVARIATES = ['age_at_test', 'sex']

# -- GLOBAL EEG FEATURES (matching R code) --
EEG_FEATURES = [
    'hurst_2s', 'pow_delta_2s', 'pow_theta_2s',
    'pow_alpha_2s', 'pow_beta_2s', 'pow_gamma_2s', 
    'pow_low_gamma_2s', 'pow_high_gamma_2s',
    'pow_per_delta_2s', 'pow_per_theta_2s', 'pow_per_alpha_2s', 
    'pow_per_beta_2s', 'pow_per_gamma_2s', 'pow_per_low_gamma_2s', 
    'pow_per_high_gamma_2s',
    'higuchi_fd_5s', 'katz_fd_5s', 'samp_entropy_5s',
    'CI_5s', 'CI_lowscale_5s', 'CI_highscale_5s'
]

# -- EEG FEATURES BY ROI (alternative) --
# EEG_FEATURES = [
#     'hurst_F_2s','pow_delta_F_2s','pow_theta_F_2s',
#     'pow_alpha_F_2s','pow_beta_F_2s','pow_gamma_F_2s','pow_low_gamma_F_2s','pow_high_gamma_F_2s',
#     'pow_per_delta_F_2s','pow_per_theta_F_2s','pow_per_alpha_F_2s', 'pow_per_beta_F_2s','pow_per_gamma_F_2s','pow_per_low_gamma_F_2s','pow_per_high_gamma_F_2s',
#     'higuchi_fd_F_5s','katz_fd_F_5s','samp_entropy_F_5s',
#     'CI_F_5s','CI_lowscale_F_5s','CI_highscale_F_5s'
# ]


def load_data(filepath):
    """Load and prepare data"""
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df


def check_ranges(df, variables):
    """Check range of values for specified variables"""
    print("="*80)
    print("(1) RANGE CHECK AND DESCRIPTIVE STATISTICS")
    print("="*80)
    
    results = []
    for var in variables:
        if var in df.columns:
            data = df[var].dropna()
            n_valid = len(data)
            n_missing = df[var].isna().sum()
            
            stats_dict = {
                'Variable': var,
                'N Valid': n_valid,
                'N Missing': n_missing,
                'Min': data.min(),
                'Max': data.max(),
                'Mean': data.mean(),
                'Median': data.median(),
                'SD': data.std(),
                'Q1': data.quantile(0.25),
                'Q3': data.quantile(0.75)
            }
            results.append(stats_dict)
            
            print(f"\n{var}:")
            print(f"  Valid N: {n_valid} | Missing: {n_missing}")
            print(f"  Range: [{data.min():.2f}, {data.max():.2f}]")
            print(f"  Mean ± SD: {data.mean():.2f} ± {data.std():.2f}")
            print(f"  Median (IQR): {data.median():.2f} ({data.quantile(0.25):.2f} - {data.quantile(0.75):.2f})")
        else:
            print(f"\n{var}: NOT FOUND IN DATASET")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}descriptive_statistics.csv", index=False)
    print(f"\n✓ Descriptive statistics saved to: {OUTPUT_DIR}descriptive_statistics.csv\n")
    
    return results_df


def check_normality_and_plot(df, variables):
    """Check normality and create distribution histograms"""
    print("="*80)
    print("(2) NORMALITY TESTS AND DISTRIBUTION PLOTS")
    print("="*80)
    
    # Create figure with subplots
    n_vars = len(variables)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_vars == 1 else axes
    
    normality_results = []
    
    for idx, var in enumerate(variables):
        if var in df.columns:
            data = df[var].dropna()
            
            # Normality tests
            shapiro_stat, shapiro_p = shapiro(data) if len(data) <= 5000 else (np.nan, np.nan)
            normaltest_stat, normaltest_p = normaltest(data)
            
            # Skewness and Kurtosis
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            
            normality_results.append({
                'Variable': var,
                'N': len(data),
                'Shapiro-Wilk W': shapiro_stat,
                'Shapiro-Wilk p': shapiro_p,
                'D\'Agostino K²': normaltest_stat,
                'D\'Agostino p': normaltest_p,
                'Skewness': skewness,
                'Kurtosis': kurtosis,
                'Normal?': 'Yes' if normaltest_p > 0.05 else 'No'
            })
            
            # Plot histogram with KDE
            ax = axes[idx]
            ax.hist(data, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
            
            # Overlay normal distribution
            mu, sigma = data.mean(), data.std()
            x = np.linspace(data.min(), data.max(), 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal fit')
            
            # Add KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            ax.plot(x, kde(x), 'g-', lw=2, label='KDE')
            
            ax.set_xlabel(var.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{var}\nN={len(data)}, Skew={skewness:.2f}, Kurt={kurtosis:.2f}\np={normaltest_p:.4f}', 
                        fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            
            print(f"\n{var}:")
            print(f"  Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.4f}" if not np.isnan(shapiro_stat) else "  Shapiro-Wilk: N/A (sample too large)")
            print(f"  D'Agostino: K²={normaltest_stat:.4f}, p={normaltest_p:.4f}")
            print(f"  Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}")
            print(f"  Distribution: {'NORMAL' if normaltest_p > 0.05 else 'NON-NORMAL'}")
    
    # Remove empty subplots
    for idx in range(n_vars, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}normality_histograms.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Histograms saved to: {OUTPUT_DIR}normality_histograms.png")
    plt.close()
    
    # Save normality results
    normality_df = pd.DataFrame(normality_results)
    normality_df.to_csv(f"{OUTPUT_DIR}normality_tests.csv", index=False)
    print(f"✓ Normality test results saved to: {OUTPUT_DIR}normality_tests.csv\n")
    
    return normality_df


def perform_mancova(df, dvs, iv, covariates):
    """Perform MANCOVA analysis"""
    print("="*80)
    print("(3) MANCOVA ANALYSIS")
    print("="*80)
    
    # Prepare data - remove missing values
    analysis_vars = dvs + [iv] + covariates
    df_clean = df[analysis_vars].dropna()
    
    print(f"\nSample size for MANCOVA: N = {len(df_clean)}")
    print(f"Original sample: N = {len(df)}")
    print(f"Excluded (missing data): N = {len(df) - len(df_clean)}\n")
    
    # Encode sex as numeric if it's not already
    if df_clean['sex'].dtype == 'object':
        df_clean['sex_numeric'] = df_clean['sex'].map({'M': 0, 'F': 1})
        sex_var = 'sex_numeric'
    else:
        sex_var = 'sex'
    
    # Print cluster distribution
    print("Cluster distribution:")
    print(df_clean[iv].value_counts().sort_index())
    print()
    
    # Build formula for MANOVA
    dv_formula = ' + '.join(dvs)
    formula = f"{dv_formula} ~ C({iv}) + age_at_test + {sex_var}"
    
    print(f"MANCOVA Formula: {formula}\n")
    
    try:
        # Perform MANOVA
        manova = MANOVA.from_formula(formula, data=df_clean)
        manova_results = manova.mv_test()
        
        print("MANCOVA Results:")
        print("="*80)
        print(manova_results)
        print("="*80)
        
        # Save results to file
        with open(f"{OUTPUT_DIR}mancova_results.txt", 'w') as f:
            f.write("MANCOVA ANALYSIS RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Formula: {formula}\n")
            f.write(f"Sample Size: N = {len(df_clean)}\n\n")
            f.write("Cluster Distribution:\n")
            f.write(str(df_clean[iv].value_counts().sort_index()) + "\n\n")
            f.write("Multivariate Tests:\n")
            f.write("="*80 + "\n")
            f.write(str(manova_results) + "\n\n")
        
        print(f"\n✓ MANCOVA results saved to: {OUTPUT_DIR}mancova_results.txt\n")
        
        # Perform univariate ANOVAs for each DV
        print("\nUnivariate ANOVAs (Follow-up tests):")
        print("="*80)
        
        univariate_results = []
        for dv in dvs:
            formula_uni = f"{dv} ~ C({iv}) + age_at_test + {sex_var}"
            model = ols(formula_uni, data=df_clean).fit()
            anova_table = anova_lm(model, typ=2)
            
            print(f"\n{dv}:")
            print(anova_table)
            
            # Extract cluster effect
            cluster_effect = anova_table.loc[f'C({iv})']
            univariate_results.append({
                'DV': dv,
                'Sum_Sq': cluster_effect['sum_sq'],
                'df': cluster_effect['df'],
                'F': cluster_effect['F'],
                'p-value': cluster_effect['PR(>F)'],
                'Significant': '***' if cluster_effect['PR(>F)'] < 0.001 else '**' if cluster_effect['PR(>F)'] < 0.01 else '*' if cluster_effect['PR(>F)'] < 0.05 else 'ns'
            })
        
        univariate_df = pd.DataFrame(univariate_results)
        univariate_df.to_csv(f"{OUTPUT_DIR}univariate_anovas.csv", index=False)
        print(f"\n✓ Univariate ANOVA results saved to: {OUTPUT_DIR}univariate_anovas.csv\n")
        
        return manova_results, df_clean, univariate_df
        
    except Exception as e:
        print(f"ERROR in MANCOVA: {e}")
        return None, df_clean, None


def calculate_group_means(df, dvs, iv):
    """Calculate and report means for every group"""
    print("="*80)
    print("(4) GROUP MEANS BY CLUSTER")
    print("="*80)
    
    # Remove missing values
    analysis_vars = dvs + [iv]
    df_clean = df[analysis_vars].dropna()
    
    means_results = []
    
    for dv in dvs:
        print(f"\n{dv}:")
        print("-"*60)
        
        # Overall mean
        overall_mean = df_clean[dv].mean()
        overall_sd = df_clean[dv].std()
        print(f"  Overall: M = {overall_mean:.2f}, SD = {overall_sd:.2f}")
        
        # Means by cluster
        cluster_means = df_clean.groupby(iv)[dv].agg(['count', 'mean', 'std', 'min', 'max'])
        print("\n  By Cluster:")
        for cluster in sorted(df_clean[iv].unique()):
            stats = cluster_means.loc[cluster]
            print(f"    Cluster {cluster}: N = {int(stats['count'])}, M = {stats['mean']:.2f}, SD = {stats['std']:.2f}")
            
            means_results.append({
                'Variable': dv,
                'Cluster': cluster,
                'N': int(stats['count']),
                'Mean': stats['mean'],
                'SD': stats['std'],
                'Min': stats['min'],
                'Max': stats['max']
            })
    
    means_df = pd.DataFrame(means_results)
    means_df.to_csv(f"{OUTPUT_DIR}group_means_by_cluster.csv", index=False)
    print(f"\n✓ Group means saved to: {OUTPUT_DIR}group_means_by_cluster.csv\n")
    
    # Create visualization - one subplot per cluster
    clusters = sorted(df_clean[iv].unique())
    n_clusters = len(clusters)
    
    # Arrange subplots
    if n_clusters == 4:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
    elif n_clusters == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axes = plt.subplots(1, n_clusters, figsize=(6*n_clusters, 6))
        if n_clusters == 1:
            axes = [axes]
    
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
    
    for idx, cluster in enumerate(clusters):
        ax = axes[idx]
        
        # Prepare data for each variable in this cluster
        var_data = [df_clean[df_clean[iv] == cluster][dv].values for dv in dvs]
        var_labels = [dv.replace('_', ' ').replace('tscore', '').replace('SRS ', '').replace('attention deficit hyperactivity', 'ADHD')[:20] for dv in dvs]
        
        bp = ax.boxplot(var_data, labels=range(1, len(dvs)+1), patch_artist=True)
        for patch_idx, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[patch_idx % len(colors)])
        
        ax.set_xlabel('Variable', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'Cluster {cluster}\n(N={len(df_clean[df_clean[iv] == cluster])})', 
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        ax.set_ylim(30, 140)
        
        # Add variable names as x-tick labels (rotated)
        ax.set_xticks(range(1, len(dvs)+1))
        ax.set_xticklabels(var_labels, rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}group_means_boxplots.png", dpi=300, bbox_inches='tight')
    print(f"✓ Group means boxplots saved to: {OUTPUT_DIR}group_means_boxplots.png\n")
    plt.close()
    
    return means_df


def linear_regression_eeg_features(df, eeg_features, iv, covariates=['age_at_test', 'sex'], reference_cluster=0):
    """
    Linear regression analysis for EEG features by cluster (Python implementation)
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    eeg_features : list
        List of EEG feature column names
    iv : str
        Independent variable (cluster column name)
    covariates : list
        List of covariates to control for
    reference_cluster : int
        Reference cluster for comparison
    
    Returns:
    --------
    results_df : DataFrame
        Regression results
    """
    print("="*80)
    print("(5) LINEAR REGRESSION: EEG FEATURES BY CLUSTER")
    print("="*80)
    
    # Prepare data
    df_eeg = df.copy()
    
    # Prepare analysis variables
    analysis_vars = eeg_features + [iv] + covariates
    df_clean = df_eeg[analysis_vars].dropna()
    
    # Encode sex as numeric if needed
    if 'sex' in covariates and df_clean['sex'].dtype == 'object':
        df_clean['sex'] = df_clean['sex'].map({'M': 0, 'F': 1})
    
    # Convert cluster to categorical with reference level
    df_clean[iv] = df_clean[iv].astype('category')
    df_clean[iv] = df_clean[iv].cat.reorder_categories(
        [reference_cluster] + [c for c in sorted(df_clean[iv].unique()) if c != reference_cluster]
    )
    
    print(f"Sample size for EEG analysis: N = {len(df_clean)}")
    print(f"Reference cluster: {reference_cluster}")
    print(f"Cluster distribution:")
    print(df_clean[iv].value_counts().sort_index())
    print(f"Controlling for: {', '.join(covariates)}")
    print(f"\nNOTE: Power features (pow_*) have been log10-transformed\n")
    
    # Store results
    all_results = []
    
    # Loop through each EEG feature
    for feature in eeg_features:
        if feature not in df_clean.columns:
            print(f"WARNING: {feature} not found in dataset")
            continue
        
        print(f"\n{'='*80}")
        print(f"Feature: {feature}")
        print(f"{'='*80}\n")
        
        # Build formula
        formula = f"{feature} ~ C({iv}, Treatment(reference={reference_cluster}))"
        for cov in covariates:
            formula += f" + {cov}"
        
        try:
            # Fit linear model
            model = ols(formula, data=df_clean).fit()
            
            # Extract results
            result_dict = {
                'Feature': feature,
                'R_squared': model.rsquared,
                'Adj_R_squared': model.rsquared_adj,
                'F_statistic': model.fvalue,
                'F_pvalue': model.f_pvalue,
                'Model_Significant': '***' if model.f_pvalue < 0.001 else '**' if model.f_pvalue < 0.01 else '*' if model.f_pvalue < 0.05 else 'ns'
            }
            
            # Get cluster means
            clusters = sorted(df_clean[iv].unique())
            for cluster in clusters:
                cluster_mean = df_clean[df_clean[iv] == cluster][feature].mean()
                result_dict[f'Cluster_{cluster}_Mean'] = cluster_mean
            
            # Extract coefficients for each cluster comparison
            print(f"  Model: {formula}")
            print(f"  R² = {model.rsquared:.4f}, Adj R² = {model.rsquared_adj:.4f}")
            print(f"  F({model.df_model:.0f}, {model.df_resid:.0f}) = {model.fvalue:.4f}, p = {model.f_pvalue:.4f}\n")
            
            ref_mean = df_clean[df_clean[iv] == reference_cluster][feature].mean()
            print(f"  Reference (Cluster {reference_cluster}): Mean = {ref_mean:.4e}")
            print(f"  Comparisons to Cluster {reference_cluster}:")
            
            for cluster in clusters:
                if cluster != reference_cluster:
                    param_name = f"C({iv}, Treatment(reference={reference_cluster}))[T.{cluster}]"
                    
                    if param_name in model.params.index:
                        coef = model.params[param_name]
                        pval = model.pvalues[param_name]
                        se = model.bse[param_name]
                        tval = model.tvalues[param_name]
                        ci_lower, ci_upper = model.conf_int().loc[param_name]
                        
                        cluster_mean = df_clean[df_clean[iv] == cluster][feature].mean()
                        
                        result_dict[f'Cluster_{cluster}_Coefficient'] = coef
                        result_dict[f'Cluster_{cluster}_SE'] = se
                        result_dict[f'Cluster_{cluster}_t_value'] = tval
                        result_dict[f'Cluster_{cluster}_p_value'] = pval
                        result_dict[f'Cluster_{cluster}_CI_lower'] = ci_lower
                        result_dict[f'Cluster_{cluster}_CI_upper'] = ci_upper
                        result_dict[f'Cluster_{cluster}_Significant'] = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                        
                        print(f"    Cluster {cluster}: β = {coef:.4e}, SE = {se:.4e}, t = {tval:.4f}, p = {pval:.4f} {result_dict[f'Cluster_{cluster}_Significant']}")
                        print(f"                  Mean = {cluster_mean:.4e}, 95% CI [{ci_lower:.4e}, {ci_upper:.4e}]")
            
            # Add covariate effects
            if covariates:
                print(f"\n  Covariates:")
                for cov in covariates:
                    if cov in model.params.index:
                        coef = model.params[cov]
                        pval = model.pvalues[cov]
                        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                        print(f"    {cov}: β = {coef:.4e}, p = {pval:.4f} {sig}")
                        
                        result_dict[f'{cov}_Coefficient'] = coef
                        result_dict[f'{cov}_p_value'] = pval
            
            all_results.append(result_dict)
            
        except Exception as e:
            print(f"\nERROR with {feature}: {e}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv(f"{OUTPUT_DIR}eeg_features_linear_regression_python.csv", index=False)
    print(f"\n✓ Linear regression results saved to: {OUTPUT_DIR}eeg_features_linear_regression_python.csv\n")
    
    # Create visualizations
    plot_linear_regression_results(results_df, df_clean, iv, eeg_features, reference_cluster)
    
    return results_df


def plot_linear_regression_results(results_df, df_clean, iv, eeg_features, reference_cluster=0):
    """
    Create visualizations of linear regression results with pairwise comparisons
    
    Parameters:
    -----------
    results_df : DataFrame
        Results from linear_regression_eeg_features
    df_clean : DataFrame
        Clean data used for regression
    iv : str
        Independent variable (cluster column)
    eeg_features : list
        List of EEG features
    reference_cluster : int
        Reference cluster
    """
    print("\nCreating plots for EEG features by cluster with pairwise comparisons...")
    
    from matplotlib.backends.backend_pdf import PdfPages
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    
    # Create PDF for all plots
    pdf_path = f"{OUTPUT_DIR}eeg_features_by_cluster_python.pdf"
    
    with PdfPages(pdf_path) as pdf:
        for feature in eeg_features:
            if feature not in results_df['Feature'].values:
                continue
            
            # Get results for this feature
            feat_results = results_df[results_df['Feature'] == feature].iloc[0]
            
            # Get cluster means and SDs
            clusters = sorted(df_clean[iv].unique())
            means = [feat_results[f'Cluster_{c}_Mean'] for c in clusters]
            sds = [df_clean[df_clean[iv] == c][feature].std() for c in clusters]
            
            # Run Tukey HSD for pairwise comparisons
            data_for_tukey = df_clean[[feature, iv]].dropna()
            try:
                tukey = pairwise_tukeyhsd(data_for_tukey[feature], data_for_tukey[iv], alpha=0.05)
                tukey_summary = tukey.summary()
                
                # Extract significant comparisons
                sig_comparisons = []
                for row_idx in range(1, len(tukey_summary.data)):
                    row = tukey_summary.data[row_idx]
                    group1 = int(row[0])
                    group2 = int(row[1])
                    pval = float(row[3])
                    
                    if pval < 0.05:
                        if pval < 0.001:
                            stars = '***'
                        elif pval < 0.01:
                            stars = '**'
                        else:
                            stars = '*'
                        sig_comparisons.append((group1, group2, stars, pval))
            except:
                sig_comparisons = []
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Determine if power feature (log-transformed)
            is_power = feature.startswith('pow_')
            
            # Calculate y-axis limits with space for comparison lines
            y_max = max([m + s for m, s in zip(means, sds)])
            y_min = min([m - s for m, s in zip(means, sds)])
            y_range = y_max - y_min
            
            # Extra space for pairwise comparison lines
            n_sig_comp = len(sig_comparisons)
            if n_sig_comp > 0:
                y_upper = y_max + (0.25 + 0.15 * n_sig_comp) * y_range
            else:
                y_upper = y_max + 0.15 * y_range
            
            # Set appropriate y-limits (don't force 0 for log data)
            if is_power or y_min < 0:
                y_lower = y_min - 0.05 * y_range
            else:
                y_lower = 0
            
            # Create bar plot
            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
            x_pos = np.arange(len(clusters))
            
            # For log-transformed data, bars should start from bottom of plot
            if is_power or y_min < 0:
                # Calculate bar heights from bottom of y-axis
                bar_heights = [m - y_lower for m in means]
                bars = ax.bar(x_pos, bar_heights, bottom=y_lower,
                             color=colors[:len(clusters)], 
                             edgecolor='black', linewidth=1.5)
            else:
                # Normal bars from 0
                bars = ax.bar(x_pos, means, color=colors[:len(clusters)], 
                             edgecolor='black', linewidth=1.5)
            
            # Add error bars
            ax.errorbar(x_pos, means, yerr=sds, fmt='none', ecolor='black', 
                       capsize=5, capthick=2, linewidth=2)
            
            # Add grid
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Draw pairwise comparison lines and stars
            if n_sig_comp > 0:
                line_height_start = y_max + 0.1 * y_range
                line_height_increment = 0.15 * y_range
                
                for comp_idx, (group1, group2, stars, pval) in enumerate(sig_comparisons):
                    # Find positions
                    idx1 = list(clusters).index(group1)
                    idx2 = list(clusters).index(group2)
                    
                    x1 = x_pos[idx1]
                    x2 = x_pos[idx2]
                    
                    # Calculate line height
                    line_height = line_height_start + comp_idx * line_height_increment
                    
                    # Draw horizontal line with vertical ticks
                    ax.plot([x1, x2], [line_height, line_height], 'k-', linewidth=2)
                    ax.plot([x1, x1], [line_height - 0.02 * y_range, line_height + 0.02 * y_range], 
                           'k-', linewidth=2)
                    ax.plot([x2, x2], [line_height - 0.02 * y_range, line_height + 0.02 * y_range], 
                           'k-', linewidth=2)
                    
                    # Add stars at midpoint
                    ax.text((x1 + x2) / 2, line_height + 0.03 * y_range, stars, 
                           ha='center', fontsize=14, color='red', fontweight='bold')
            
            # Set y-limits
            ax.set_ylim(y_lower, y_upper)
            
            # Formatting
            ax.set_xlabel('Cluster', fontsize=12)
            y_label = 'Mean Value (log10)' if is_power else 'Mean Value'
            ax.set_ylabel(y_label, fontsize=12)
            
            title = f'{feature}'
            if is_power:
                title += ' (log10-transformed)'
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(c) for c in clusters])
            
            # Add legend
            legend_text = [
                '*** p < 0.001',
                '** p < 0.01', 
                '* p < 0.05',
                'Lines show pairwise comparisons (Tukey HSD)'
            ]
            ax.text(0.98, 0.98, '\n'.join(legend_text), 
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
    
    print(f"✓ Plots saved to: {pdf_path}\n")
    
    # Create summary heatmap of significant features
    create_significance_heatmap(results_df, reference_cluster)


def create_significance_heatmap(results_df, reference_cluster=0):
    """Create heatmap showing ALL pairwise cluster differences"""
    from scipy.stats import f_oneway
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    
    print("Creating significance heatmap with ALL pairwise comparisons...")
    print("Computing Tukey HSD post-hoc tests...")
    
    # Re-load data for post-hoc tests
    df = pd.read_csv(DATA_PATH)
    
    # Get features
    features = results_df['Feature'].tolist()
    
    # Get clusters
    cluster_col = 'cluster'  # Assuming this is the cluster column name
    clusters = sorted(df[cluster_col].dropna().unique())
    
    # Generate all pairwise comparisons
    pairwise_comparisons = []
    for i, c1 in enumerate(clusters):
        for c2 in clusters[i+1:]:
            pairwise_comparisons.append(f'C{c1}-C{c2}')
    
    # Create matrix: rows = features, cols = all pairwise comparisons
    sig_matrix = np.zeros((len(features), len(pairwise_comparisons)))
    
    # For each feature, run Tukey HSD
    for feat_idx, feature in enumerate(features):
        if feature not in df.columns:
            continue
        
        # Prepare data
        data_clean = df[[feature, cluster_col]].dropna()
        
        try:
            # Run Tukey HSD
            tukey = pairwise_tukeyhsd(data_clean[feature], data_clean[cluster_col], alpha=0.05)
            
            # Extract pairwise results
            tukey_summary = tukey.summary()
            
            # Parse results
            for row_idx in range(len(tukey_summary.data) - 1):  # Skip header
                row = tukey_summary.data[row_idx + 1]
                group1 = int(row[0])
                group2 = int(row[1])
                pval = float(row[3])
                
                # Find comparison index
                comp_str = f'C{min(group1, group2)}-C{max(group1, group2)}'
                if comp_str in pairwise_comparisons:
                    comp_idx = pairwise_comparisons.index(comp_str)
                    
                    # Assign significance level
                    if pval < 0.001:
                        sig_matrix[feat_idx, comp_idx] = 3
                    elif pval < 0.01:
                        sig_matrix[feat_idx, comp_idx] = 2
                    elif pval < 0.05:
                        sig_matrix[feat_idx, comp_idx] = 1
                    else:
                        sig_matrix[feat_idx, comp_idx] = 0
        except Exception as e:
            print(f"  Warning: Could not compute Tukey HSD for {feature}: {e}")
            continue
    
    print(f"✓ Computed pairwise comparisons for {len(features)} features\n")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(pairwise_comparisons) * 0.8), 
                                    max(10, len(features) * 0.4)))
    
    im = ax.imshow(sig_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=3)
    
    # Set ticks
    ax.set_xticks(np.arange(len(pairwise_comparisons)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(pairwise_comparisons)
    ax.set_yticklabels(features, fontsize=8)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['ns', '*', '**', '***'])
    cbar.set_label('Significance', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(features)):
        for j in range(len(pairwise_comparisons)):
            val = sig_matrix[i, j]
            if val > 0:
                text_val = '*' * int(val)
                ax.text(j, i, text_val, ha='center', va='center', 
                       color='white' if val > 1 else 'black', fontweight='bold', fontsize=8)
    
    ax.set_title('ALL Pairwise Cluster Differences in EEG Features\n(Tukey HSD Post-hoc)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}eeg_significance_heatmap_python.png", dpi=300, bbox_inches='tight')
    print(f"✓ Significance heatmap saved to: {OUTPUT_DIR}eeg_significance_heatmap_python.png\n")
    plt.close()

def log_transform_power_features(df, features):
    """Log transform power features"""
    for feature in features:
        if feature in df.columns:
            df[feature] = np.log10(df[feature])
    return df

def main():
    """Main analysis pipeline"""
    print("\n" + "="*80)
    print(" "*20 + "GENiAL STATISTICAL ANALYSIS")
    print("="*80 + "\n")
    
    # Load data
    df = load_data(DATA_PATH)
    
    # (1) Check ranges
    descriptive_stats = check_ranges(df, RANGE_CHECK_VARS)
    
    # (2) Check normality and create histograms
    normality_results = check_normality_and_plot(df, RANGE_CHECK_VARS)

    # (3) Log transform power features
    df = log_transform_power_features(df, EEG_FEATURES)
    
    # (4) MANCOVA
    mancova_results, df_clean, univariate_results = perform_mancova(
        df, MANCOVA_DVS, MANCOVA_IV, MANCOVA_COVARIATES
    )
    
    # (5) Group means
    group_means = calculate_group_means(df, MANCOVA_DVS, MANCOVA_IV)
    
    # (6) EEG features significance using Linear Regression
    eeg_results = linear_regression_eeg_features(
        df, EEG_FEATURES, MANCOVA_IV,
        covariates=['age_at_test', 'sex'],
        reference_cluster=0
    )
    
    print("="*80)
    print(" "*25 + "ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nOutput files:")
    print("  - descriptive_statistics.csv")
    print("  - normality_tests.csv")
    print("  - normality_histograms.png")
    print("  - mancova_results.txt")
    print("  - univariate_anovas.csv")
    print("  - group_means_by_cluster.csv")
    print("  - group_means_boxplots.png")
    print("  - eeg_features_linear_regression_python.csv")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

