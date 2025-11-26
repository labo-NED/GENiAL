#!/usr/bin/env python3
"""
Diagnosis Distribution Analysis and Visualization
Python equivalent of diagnosis_pie_charts.r

Creates:
- Pie charts for diagnostic distribution by cluster
- Frequency analysis of individual diagnoses
- Bar plots of top diagnoses per cluster
- Heatmap of diagnosis frequencies across clusters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 10

########################## Configuration ##########################

# File paths
DATA_PATH = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/merged_clustered_EEG_features_by_roi_RSRio_NOV_24_2025.csv"
OUTPUT_DIR = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/Stats/by ROI/"

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Custom color palette for diagnosis groups (matching R script)
DIAGNOSIS_COLORS = {
    "ASD": "#1976D2",              # deep blue
    "ADHD": "#E91E63",             # pink
    "ASD + ADHD": "#8E24AA",       # purple
    "ADHD + ASD behavior": "#BA68C8",   # light purple
    "No ASD/ADHD": "#9E9E9E"       # gray
}

########################## Functions ##########################

def diagnosis_labels(diagnosis_string):
    """
    Parse diagnosis column and determine diagnostic group
    
    Args:
        diagnosis_string: String containing comma or semicolon separated diagnoses
        
    Returns:
        String indicating diagnostic group category
    """
    # Handle missing or empty values
    if pd.isna(diagnosis_string) or diagnosis_string == "" or diagnosis_string == "None":
        return "No ASD/ADHD"
    
    # Replace semicolons with commas and split
    diagnosis_string = str(diagnosis_string).replace(";", ",")
    diagnoses = [d.strip().lower() for d in diagnosis_string.split(",")]
    
    # Check for ASD, autism_behavior, and ADHD
    has_asd = any(d in ["autism", "asd"] for d in diagnoses)
    has_asd_behavior = any(d in ["autism_behavior", "autistic_behav", "autistic behavior"] for d in diagnoses)
    has_adhd = any(d in ["adhd", "attention_deficit_hyperactivity", "attention deficit hyperactivity"] for d in diagnoses)
    
    # Determine diagnostic group
    if has_asd and has_adhd:
        return "ASD + ADHD"
    elif has_asd:
        return "ASD"
    elif has_asd_behavior and has_adhd:
        return "ADHD + ASD behavior"
    elif has_adhd:
        return "ADHD"
    else:
        return "No ASD/ADHD"


def parse_individual_diagnoses(diagnosis_string):
    """
    Parse all individual diagnoses from a diagnosis string
    
    Args:
        diagnosis_string: String containing comma or semicolon separated diagnoses
        
    Returns:
        List of individual diagnosis strings
    """
    # Handle missing or empty values
    if pd.isna(diagnosis_string) or diagnosis_string == "" or diagnosis_string == "None":
        return []
    
    # Replace semicolons with commas and split
    diagnosis_string = str(diagnosis_string).replace(";", ",")
    diagnoses = [d.strip().lower() for d in diagnosis_string.split(",")]
    
    # Return non-empty diagnoses
    return [d for d in diagnoses if d != ""]


def create_pie_chart(cluster_data, cluster_label, output_dir):
    """
    Create and save pie chart for a specific cluster
    
    Args:
        cluster_data: DataFrame filtered to specific cluster
        cluster_label: Label of the cluster
        output_dir: Directory to save the plot
    """
    total_n = len(cluster_data)
    
    # Count diagnosis groups
    diag_counts = cluster_data['diagnosis_group'].value_counts().reset_index()
    diag_counts.columns = ['diagnosis_group', 'count']
    diag_counts['percentage'] = (diag_counts['count'] / total_n) * 100
    
    # Filter out zero counts
    diag_counts = diag_counts[diag_counts['count'] > 0]
    
    if len(diag_counts) == 0:
        print(f"  Warning: No data for cluster {cluster_label}")
        return
    
    # Create labels with count and percentage
    diag_counts['label'] = diag_counts.apply(
        lambda x: f"{int(x['count'])} ({x['percentage']:.1f}%)", axis=1
    )
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = [DIAGNOSIS_COLORS.get(dg, '#999999') for dg in diag_counts['diagnosis_group']]
    
    wedges, texts, autotexts = ax.pie(
        diag_counts['count'],
        labels=diag_counts['label'],
        colors=colors,
        autopct='',
        startangle=90,
        textprops={'color': 'white', 'weight': 'bold', 'fontsize': 12}
    )
    
    # Create legend
    legend_labels = [f"{dg}" for dg in diag_counts['diagnosis_group']]
    ax.legend(
        wedges, legend_labels,
        title="Diagnosis",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=10
    )
    
    ax.set_title(
        f"Cluster {cluster_label} (n = {total_n}) - Diagnostic Distribution",
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}diagnosis_pie_chart_cluster_{cluster_label}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def create_bar_plot(cluster_data, cluster_label, output_dir, top_n=10):
    """
    Create bar plot showing top diagnoses for a cluster
    
    Args:
        cluster_data: DataFrame with diagnosis frequencies for cluster
        cluster_label: Label of the cluster
        output_dir: Directory to save the plot
        top_n: Number of top diagnoses to show
    """
    if len(cluster_data) == 0:
        return
    
    # Separate "No Diagnosis" from others
    no_diag = cluster_data[cluster_data['diagnosis_label'] == 'No Diagnosis']
    others = cluster_data[cluster_data['diagnosis_label'] != 'No Diagnosis'].nlargest(top_n, 'count')
    
    # Combine
    plot_data = pd.concat([others, no_diag]).drop_duplicates(subset=['diagnosis_label'])
    
    if len(plot_data) == 0:
        return
    
    # Sort by count
    plot_data = plot_data.sort_values('count', ascending=True)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(
        range(len(plot_data)),
        plot_data['count'],
        color='#1976D2',
        alpha=0.8
    )
    
    # Add labels
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        ax.text(
            row['count'] + 0.5,
            i,
            f"{int(row['count'])} ({row['percentage']:.1f}%)",
            va='center',
            fontsize=9
        )
    
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data['diagnosis_label'], fontsize=10)
    ax.set_xlabel('Count (Percentage)', fontsize=11)
    ax.set_title(
        f"Top {min(top_n, len(plot_data))} Diagnoses in Cluster {cluster_label} "
        f"(n = {plot_data['total_participants'].iloc[0]})",
        fontsize=13,
        fontweight='bold'
    )
    
    # Extend x-axis for labels
    ax.set_xlim(0, plot_data['count'].max() * 1.2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}diagnosis_frequency_cluster_{cluster_label}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def create_heatmap(diagnosis_freq_by_cluster, output_dir, top_n=20):
    """
    Create heatmap showing diagnosis frequencies across all clusters
    
    Args:
        diagnosis_freq_by_cluster: DataFrame with diagnosis frequencies
        output_dir: Directory to save the plot
        top_n: Number of top diagnoses to include
    """
    # Get top diagnoses overall (excluding "No Diagnosis")
    top_diagnoses = (
        diagnosis_freq_by_cluster[diagnosis_freq_by_cluster['diagnosis_label'] != 'No Diagnosis']
        .groupby('diagnosis_label')['count']
        .sum()
        .nlargest(top_n)
        .index
        .tolist()
    )
    
    # Create pivot table
    heatmap_data = (
        diagnosis_freq_by_cluster[diagnosis_freq_by_cluster['diagnosis_label'].isin(top_diagnoses)]
        .pivot(index='diagnosis_label', columns='cluster', values='percentage')
        .fillna(0)
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 12))
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap=sns.light_palette("#1976D2", as_cmap=True),
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Percentage (%)'},
        ax=ax
    )
    
    ax.set_title(
        'Diagnosis Frequency Heatmap by Cluster (%)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Diagnosis', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}diagnosis_frequency_heatmap.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    # Also save as PDF
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap=sns.light_palette("#1976D2", as_cmap=True),
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Percentage (%)'},
        ax=ax
    )
    ax.set_title(
        'Diagnosis Frequency Heatmap by Cluster (%)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Diagnosis', fontsize=12)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}diagnosis_frequency_heatmap.pdf",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


########################## Main Analysis ##########################

def main():
    """Main analysis pipeline"""
    print("\n" + "="*80)
    print(" "*20 + "DIAGNOSIS DISTRIBUTION ANALYSIS")
    print("="*80 + "\n")
    
    # Import dataset
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns\n")
    
    ########################### Preprocessing ############################
    print("="*80)
    print("PREPROCESSING")
    print("="*80 + "\n")
    
    # Convert cluster to categorical
    df['cluster'] = df['cluster'].astype('category')
    
    # Apply diagnosis_labels function to create diagnosis_group
    df['diagnosis_group'] = df['diagnosis'].apply(diagnosis_labels)
    df['diagnosis_group'] = pd.Categorical(df['diagnosis_group'])
    
    print(f"Clusters: {sorted(df['cluster'].unique())}")
    print(f"Diagnosis groups: {sorted(df['diagnosis_group'].unique())}\n")
    
    ########################## Pie Charts for Individual Clusters ##########################
    print("="*80)
    print("CREATING PIE CHARTS FOR INDIVIDUAL CLUSTERS")
    print("="*80 + "\n")
    
    clusters = sorted(df['cluster'].unique())
    
    for cluster in clusters:
        print(f"Creating pie chart for Cluster {cluster}...")
        cluster_data = df[df['cluster'] == cluster]
        create_pie_chart(cluster_data, cluster, OUTPUT_DIR)
    
    print(f"\n✓ Pie charts saved to {OUTPUT_DIR}\n")
    
    ########################## Diagnosis Frequency Analysis by Cluster ##########################
    print("="*80)
    print("DIAGNOSIS FREQUENCY ANALYSIS BY CLUSTER")
    print("="*80 + "\n")
    
    # Parse individual diagnoses for each participant
    df['individual_diagnoses'] = df['diagnosis'].apply(parse_individual_diagnoses)
    df['has_diagnosis'] = df['individual_diagnoses'].apply(len) > 0
    
    # Count participants with no diagnosis per cluster
    no_diagnosis_counts = (
        df[~df['has_diagnosis']]
        .groupby('cluster')
        .size()
        .reset_index(name='count')
    )
    no_diagnosis_counts['individual_diagnoses'] = 'no_diagnosis'
    no_diagnosis_counts['diagnosis_label'] = 'No Diagnosis'
    
    # Create long format for participants with diagnoses
    diagnosis_rows = []
    for idx, row in df[df['has_diagnosis']].iterrows():
        for diag in row['individual_diagnoses']:
            if diag and diag != "":
                diagnosis_rows.append({
                    'cluster': row['cluster'],
                    'individual_diagnoses': diag
                })
    
    diagnosis_long = pd.DataFrame(diagnosis_rows)
    
    # Count frequencies of each diagnosis per cluster
    if len(diagnosis_long) > 0:
        diagnosis_freq_by_cluster = (
            diagnosis_long
            .groupby(['cluster', 'individual_diagnoses'])
            .size()
            .reset_index(name='count')
        )
    else:
        diagnosis_freq_by_cluster = pd.DataFrame(columns=['cluster', 'individual_diagnoses', 'count'])
    
    # Add "no_diagnosis" counts
    diagnosis_freq_by_cluster = pd.concat([
        diagnosis_freq_by_cluster,
        no_diagnosis_counts[['cluster', 'individual_diagnoses', 'count']]
    ], ignore_index=True)
    
    # Calculate percentages per cluster
    cluster_totals = df.groupby('cluster').size().reset_index(name='total_participants')
    
    diagnosis_freq_by_cluster = diagnosis_freq_by_cluster.merge(cluster_totals, on='cluster')
    diagnosis_freq_by_cluster['percentage'] = (
        diagnosis_freq_by_cluster['count'] / diagnosis_freq_by_cluster['total_participants'] * 100
    ).round(1)
    
    # Create readable diagnosis labels
    diagnosis_freq_by_cluster['diagnosis_label'] = diagnosis_freq_by_cluster['individual_diagnoses'].apply(
        lambda x: 'No Diagnosis' if x == 'no_diagnosis' else x.replace('_', ' ').title()
    )
    
    # Sort by cluster and count
    diagnosis_freq_by_cluster = diagnosis_freq_by_cluster.sort_values(['cluster', 'count'], ascending=[True, False])
    
    # Print summary table
    for cluster in clusters:
        cluster_data = diagnosis_freq_by_cluster[diagnosis_freq_by_cluster['cluster'] == cluster]
        total_n = cluster_data['total_participants'].iloc[0]
        sum_counts = cluster_data['count'].sum()
        
        print(f"CLUSTER {cluster} (n = {total_n} participants)")
        print("-" * 60)
        print(cluster_data[['diagnosis_label', 'count', 'percentage']].to_string(index=False))
        print(f"\nSum of diagnosis counts: {sum_counts} (should equal {total_n})")
        if sum_counts == total_n:
            print("✓ Counts match total participants")
        else:
            print("⚠ WARNING: Counts do not match total participants!")
        print("\n")
    
    # Save frequency tables
    diagnosis_freq_by_cluster_save = diagnosis_freq_by_cluster[
        ['cluster', 'diagnosis_label', 'count', 'total_participants', 'percentage']
    ]
    diagnosis_freq_by_cluster_save.to_csv(
        f"{OUTPUT_DIR}diagnosis_frequency_by_cluster_long.csv",
        index=False
    )
    
    # Create wide format
    diagnosis_freq_wide = (
        diagnosis_freq_by_cluster[['cluster', 'diagnosis_label', 'count']]
        .pivot(index='diagnosis_label', columns='cluster', values='count')
        .fillna(0)
        .astype(int)
    )
    # Sort by total count
    diagnosis_freq_wide['Total'] = diagnosis_freq_wide.sum(axis=1)
    diagnosis_freq_wide = diagnosis_freq_wide.sort_values('Total', ascending=False).drop('Total', axis=1)
    diagnosis_freq_wide.to_csv(f"{OUTPUT_DIR}diagnosis_frequency_by_cluster_wide.csv")
    
    print("Frequency tables saved to:")
    print("  - diagnosis_frequency_by_cluster_long.csv (long format)")
    print("  - diagnosis_frequency_by_cluster_wide.csv (wide format)\n")
    
    ########################## Create Bar Plots ##########################
    print("="*80)
    print("CREATING BAR PLOTS FOR TOP DIAGNOSES PER CLUSTER")
    print("="*80 + "\n")
    
    top_n_diagnoses = 10
    for cluster in clusters:
        print(f"Creating bar plot for Cluster {cluster}...")
        cluster_data = diagnosis_freq_by_cluster[diagnosis_freq_by_cluster['cluster'] == cluster]
        create_bar_plot(cluster_data, cluster, OUTPUT_DIR, top_n=top_n_diagnoses)
    
    print(f"\n✓ Bar plots saved to {OUTPUT_DIR}\n")
    
    ########################## Create Heatmap ##########################
    print("="*80)
    print("CREATING DIAGNOSIS FREQUENCY HEATMAP")
    print("="*80 + "\n")
    
    create_heatmap(diagnosis_freq_by_cluster, OUTPUT_DIR, top_n=20)
    
    print(f"✓ Heatmap saved to {OUTPUT_DIR}diagnosis_frequency_heatmap.png")
    print(f"✓ Heatmap saved to {OUTPUT_DIR}diagnosis_frequency_heatmap.pdf\n")
    
    ########################## Summary Statistics ##########################
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80 + "\n")
    
    if len(diagnosis_long) > 0:
        total_unique_diagnoses = diagnosis_long['individual_diagnoses'].nunique()
        print(f"Total unique diagnoses across all clusters: {total_unique_diagnoses}\n")
    
    diagnosis_summary = (
        diagnosis_freq_by_cluster
        .groupby('diagnosis_label')
        .agg({
            'count': 'sum',
            'cluster': 'count'
        })
        .rename(columns={'count': 'total_count', 'cluster': 'present_in_clusters'})
        .sort_values('total_count', ascending=False)
        .reset_index()
    )
    
    print("Top 20 most common diagnoses overall (excluding 'No Diagnosis'):")
    diagnosis_summary_filtered = diagnosis_summary[diagnosis_summary['diagnosis_label'] != 'No Diagnosis']
    print(diagnosis_summary_filtered.head(20).to_string(index=False))
    
    # Show "No Diagnosis" separately
    no_diag_summary = diagnosis_summary[diagnosis_summary['diagnosis_label'] == 'No Diagnosis']
    if len(no_diag_summary) > 0:
        print("\nParticipants with No Diagnosis:")
        print(no_diag_summary.to_string(index=False))
    
    print("\n" + "="*80)
    print(" "*25 + "ANALYSIS COMPLETE!")
    print("="*80 + "\n")
    print(f"All outputs saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()

