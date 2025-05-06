import pandas as pd
import os
import matplotlib.pyplot as plt
from IPython.display import FileLink
import seaborn as sns

class DataPreprocessingPipeline:
    def __init__(self, root_dir):
        """Initialize the pipeline with root directory."""
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, 'Data/')
        self._setup_file_paths()
        
    def _setup_file_paths(self):
        """Setup all file paths used in the pipeline."""
        # Original Data CSVs from REDCAP
        self.original_demog_genetics_data = os.path.join(self.root_dir, 'Data/Genetics/Input/Q1K report EEG_NDD_génétique.csv')
        self.original_dia_cogn_data = os.path.join(self.root_dir, 'Data/Diagnosis + Cogn Tests/Q1K-Dia_cogn.csv')
        self.original_eeg_rs_data = os.path.join(self.root_dir, 'Data/EEG/Q1K_concatenated_features_RS.csv')
        
        # Input Files for CNV prediction
        self.genetics_only_data = os.path.join(self.root_dir, 'Data/Genetics/Input/CNV-Analysis.csv')
        self.hg38_input_data = os.path.join(self.root_dir, 'Data/Genetics/Input/CNV-Analysis-Hg38.tsv')
        self.hg19_input_data = os.path.join(self.root_dir, 'Data/Genetics/Input/CNV-Analysis-Hg19.tsv')
        self.hg18_input_data = os.path.join(self.root_dir, 'Data/Genetics/Input/CNV-Analysis-Hg18.tsv')
        
        # Calculated CNVs files
        self.cnv_prediction_hg19_data = os.path.join(self.root_dir, 'Data/Genetics/Input/cnvprediction-hg19-output.csv')
        self.cnv_prediction_hg38_data = os.path.join(self.root_dir, 'Data/Genetics/Input/cnvprediction-hg38-output.csv')
        
        # Index to Participant Code Map
        self.id_map = os.path.join(self.root_dir, 'Data/Genetics/Input/sample-id-map.csv')
        
        # Output Files
        self.preprocessed_data_path = os.path.join(self.root_dir, 'Data/Final/GENIAL-DB-preprocessed-RS.csv')

    def import_raw_data(self):
        """Import all raw data files."""
        self.demog_genetics_df = pd.read_csv(self.original_demog_genetics_data)
        self.dia_cogn_df = pd.read_csv(self.original_dia_cogn_data)
        self.eeg_rs_features_df = pd.read_csv(self.original_eeg_rs_data)
        self.cnv_hg19_df = pd.read_csv(self.cnv_prediction_hg19_data)
        self.cnv_hg38_df = pd.read_csv(self.cnv_prediction_hg38_data)
        self.id_map_df = pd.read_csv(self.id_map)

    def clean_demographic_data(self):
        """Clean and standardize demographic data."""
        # Strip column names
        self.demog_genetics_df.columns = self.demog_genetics_df.columns.str.strip()
        
        # Define column mapping
        self.column_mapping = {
            'Enter in the box participant\'s EEG code as written here :  [intake_arm1][q1k_relative_idgenerated_1] [intake_arm1][q1k_proband_id_1]': 'ParticipantID',
            'Was EEG attempted?': 'EEG_attempted',
            'EEG site:': 'EEG_site',
            'Birthdate': 'Birthdate',
            'EEG date': 'EEG_date',
            'Age at EEG (years)': 'EEG_age',
            'Sex at birth:': 'Sex_at_birth',
            'Unknown - Specify:': 'diag_unknown_specify',
            'Other - Specify:': 'diag_other_specify',
            'Medication taken the morning of the EEG': 'medication_at_EEG',
            'Resting state with Rio done?': 'RS_Rio_done',
            'Participant\'s code for resting state with Rio :': 'RS_Rio_code',
            'Resting state done?': 'RS_done',
            'Participant\'s code for resting state :': 'RS_code',
            'Tone Oddball done?': 'TO_done',
            'Participant\'s code for TO': 'TO_code',
            'GO done?': 'GO_done',
            'Participant\'s code for GO:': 'GO_code',
            'VEP done?': 'VEP_done',
            'Participant\'s code for VEP:': 'VEP_code',
            'AEP done?': 'AEP_done',
            'Participant\'s code for AEP :   Choose version A or B': 'AEP_code',
            'Randomization file used (A or B)': 'AEP_randomization_file',
            'NSP done?': 'NSP_done',
            'Participant\'s code for NSP:': 'NSP_code',
            'VS done?': 'VS_done',
            'Participant\'s code for VS:': 'VS_code',
            'MMN Oddball done?': 'MMN_done',
            'Participant\'s code for MMN': 'MMN_code',
            'Result aCGH/ LP-WGS': 'Genetic_test_result',
            'Genetic status of the participant:': 'Genetic_status',
            'Affected chromosome:': 'Affected_chromosome',
            'Full proximal boundary (e.g., 2960000):': 'Proximal_boundary',
            'Full distal boundary (e.g., 3020000):': 'Distal_boundary',
            'Please indicate the Human Genome Version used': 'Genome_version',
            'Single gene testing:': 'Single_gene_testing',
            'Fragile X': 'Fragile_X',
            'Exome / Panel testing:': 'Exome_panel_testing',
            'Diagnosis (choice=Control (no genetic or neurodev disorder))': 'diag_control',
            'Diagnosis (choice=Neurodevelopmental disorder)': 'diag_neurodev',
            'Diagnosis (choice=Genetic carrier)': 'diag_genetic_carrier',
            'Diagnosis (choice=Unknown (under investigation, suspected))': 'diag_unknown',
            'Diagnosis (choice=Other (non neurodevelopmental diagnosis))': 'diag_other',
            'Inheritance (choice=De novo)': 'inheritance_denovo',
            'Inheritance (choice=Mothers inherited)': 'inheritance_mothers_inherited',
            'Inheritance (choice=Fathers inherited)': 'inheritance_fathers_inherited',
            'Inheritance (choice=Unknown)': 'inheritance_unknown',
            'Inheritance (choice=Mosaic)': 'inheritance_mosaic'
        }
        
        # Rename columns
        self.demog_genetics_df = self.demog_genetics_df.rename(columns=self.column_mapping)
        
        # Add family member type
        self.demog_genetics_df['ParticipantID'] = self.demog_genetics_df['ParticipantID'].astype('str')
        self.demog_genetics_df['family_member_type'] = self.demog_genetics_df['ParticipantID'].apply(self._categorize_family_member_type)
        
        # Convert diagnosis and inheritance columns to binary
        diag_cols = [col for col in self.demog_genetics_df.columns if col.startswith('diag_')]
        inheritance_cols = [col for col in self.demog_genetics_df.columns if col.startswith('inheritance_')]
        for col in diag_cols + inheritance_cols:
            self.demog_genetics_df[col] = self.demog_genetics_df[col].map({'Checked': 1, 'Unchecked': 0})

    def _categorize_family_member_type(self, id_value):
        """Determine the family member type based on the ID."""
        last_part = id_value.split('_')[-1]
        if last_part == 'P':
            return 'Proband'
        elif last_part.startswith('S') and last_part[1:].isdigit():
            return 'Sibling'
        elif last_part.startswith('F') and last_part[1:].isdigit():
            return 'Father'
        elif last_part.startswith('M') and last_part[1:].isdigit():
            return 'Mother'
        elif last_part.startswith('C') and last_part[1:].isdigit():
            return 'Child'
        elif last_part.startswith('O') and last_part[1:].isdigit():
            return 'Other'
        else:
            return pd.NA

    def process_cnv_data(self):
        """Process CNV (Copy Number Variation) data."""
        # Merge participantID to the genetic data
        self.cnv_hg19_df = self.cnv_hg19_df.merge(self.id_map_df, on='ID', how='left')
        self.cnv_hg38_df = self.cnv_hg38_df.merge(self.id_map_df, on='ID', how='left')
        
        # Merge hg19 and hg38 dataframes
        self.cnv_df = pd.concat([self.cnv_hg19_df, self.cnv_hg38_df], axis=0)
        
        # Clean up participant IDs
        self.cnv_df['ParticipantID'] = self.cnv_df['ParticipantID'].astype(str).str.strip()
        self.cnv_df.columns = self.cnv_df.columns.str.strip()

    def process_eeg_data(self, is_rio=False):
        """Process EEG data, optionally including Rio data."""
        # Clean up column names
        self.eeg_rs_features_df.columns = self.eeg_rs_features_df.columns.str.strip()
        
        # Split into RS and RSRIO dataframes
        self.rs_df = self.eeg_rs_features_df[self.eeg_rs_features_df['ID'].str.contains('_RS_')].copy()
        self.rsrio_df = self.eeg_rs_features_df[self.eeg_rs_features_df['ID'].str.contains('_RSRIO_')].copy()
        
        # Clean up participant IDs
        self.rs_df['ParticipantID'] = self.rs_df['ID'].str.replace(r'_RS_.*', '', regex=True)
        self.rsrio_df['ParticipantID'] = self.rsrio_df['ID'].str.replace(r'_RSRIO_.*', '', regex=True)
        
        # Rename EEG feature columns
        self.rs_df = self.rs_df.rename(columns={col: f"EEG_{col}" for col in self.rs_df.columns if col not in ['ID', 'ParticipantID']})
        self.rsrio_df = self.rsrio_df.rename(columns={col: f"EEG_{col}" for col in self.rsrio_df.columns if col not in ['ID', 'ParticipantID']})
        
        # Select appropriate dataset based on is_rio parameter
        self.rs_data = self.rsrio_df if is_rio else self.rs_df

    def merge_all_data(self):
        """Merge all processed data sources."""
        # Merge demographic and CNV data
        self.merged_df = self.demog_genetics_df.merge(
            self.cnv_df[['ParticipantID', 'NVIQ_CIupr', 'ORASD_upr', 'SRS_CIupr', 'PdN_CIupr', 'sum_LOEUF_complete']],
            on='ParticipantID',
            how='left'
        )
        
        # Merge with diagnosis and cognitive data
        self.merged_df = self.merged_df.merge(self.dia_cogn_df, on="ParticipantID", how="left")
        
        # Merge with EEG data
        self.merged_df = self.merged_df.merge(self.rs_data, on="ParticipantID", how="left")
        
        # Clean up merged dataframe
        self.merged_df = self.merged_df.loc[:, ~self.merged_df.columns.duplicated()]
        self.merged_df = self.merged_df.drop(columns=['record_id', 'redcap_event_name', 'redcap_repeat_instrument', 
                                                     'redcap_repeat_instance', 'eeg_participant_code'])
        
        # Strip string values
        self.merged_df = self.merged_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    def export_data(self):
        """Export the processed data to CSV."""
        self.merged_df.to_csv(self.preprocessed_data_path, index=False)
        return FileLink(self.preprocessed_data_path)

    def run_pipeline(self, is_rio=False):
        """Run the complete data preprocessing pipeline."""
        self.import_raw_data()
        self.clean_demographic_data()
        self.process_cnv_data()
        self.process_eeg_data(is_rio)
        self.merge_all_data()
        return self.export_data()

# Example usage:
if __name__ == "__main__":
    root_dir = "/Users/emmanuelle.coutu-nadeau/Library/Mobile Documents/com~apple~CloudDocs/UdeM/MSc Psycho/LABO NED - Personal Drive/Code/GENiAL/"
    pipeline = DataPreprocessingPipeline(root_dir)
    pipeline.run_pipeline(is_rio=False)