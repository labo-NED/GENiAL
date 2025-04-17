%% Statistical Analysis of EEG Features in Relation to Genetic Diagnosis
% This script performs statistical analysis on EEG features including:
% 1. Data preprocessing (normalization, handling missing values)
% 2. Multiple regression analysis with covariates (age, sex)
% 3. Statistical testing and results table generation

%% Set up paths
root_dir = '/Users/emmanuelle.coutu-nadeau/Library/Mobile Documents/com~apple~CloudDocs/UdeM/MSc Psycho/LABO NED - Personal Drive/Code/GENiAL';
preprocessed_data_path = fullfile(root_dir, 'Data', 'Final', 'GENIAL-DB-preprocessed-V2.csv');
output_dir = fullfile(root_dir, 'Results', 'Statistical_Analysis');

% Create output directory if it doesn't exist
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Load and prepare data
fprintf('Loading and preparing data...\n');

% Load the preprocessed data
data = readtable(preprocessed_data_path);

% Identify EEG feature columns
column_names = data.Properties.VariableNames;
eeg_cols = startsWith(column_names, 'EEG_');
non_numeric_cols = {'EEG_attempted', 'EEG_site', 'EEG_date', 'EEG_age', 'EEG_Age', 'EEG_Sex'};
eeg_cols = eeg_cols & ~ismember(column_names, non_numeric_cols);
eeg_feature_names = column_names(eeg_cols);

% Convert sex to binary
sex_binary = double(strcmp(data.Sex_at_birth, 'Male'));

% Create diagnostic groups
% 0: Control (diag_control = 1)
% 1: Neurodev only (diag_neurodev = 1 and diag_genetic_carrier = 0)
% 2: Genetic carrier (diag_genetic_carrier = 1)
diagnostic_group = zeros(height(data), 1);
diagnostic_group(data.diag_neurodev == 1 & data.diag_genetic_carrier == 0) = 1;
diagnostic_group(data.diag_genetic_carrier == 1) = 2;

% Print group sizes
fprintf('\nDiagnostic group sizes:\n');
fprintf('Controls: %d\n', sum(diagnostic_group == 0));
fprintf('Neurodev only: %d\n', sum(diagnostic_group == 1));
fprintf('Genetic carriers: %d\n', sum(diagnostic_group == 2));

