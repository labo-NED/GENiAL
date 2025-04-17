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

%% Normalize data
fprintf('\nNormalizing data...\n');

% Extract EEG features
eeg_features = table2array(data(:, eeg_cols));

% Report NaN statistics
fprintf('\nMissing data statistics:\n');
nan_counts = sum(isnan(eeg_features), 1);
total_samples = size(eeg_features, 1);
for i = 1:length(eeg_feature_names)
    fprintf('%s: %d/%d (%.1f%%) missing values\n', ...
        eeg_feature_names{i}, nan_counts(i), total_samples, ...
        nan_counts(i)/total_samples*100);
end

% Z-score normalization (handling NaN values)
eeg_features_norm = zeros(size(eeg_features));
for i = 1:size(eeg_features, 2)
    valid_idx = ~isnan(eeg_features(:, i));
    if sum(valid_idx) > 0  % Only normalize if there are valid values
        eeg_features_norm(valid_idx, i) = zscore(eeg_features(valid_idx, i));
    end
end

%% Run multiple regression analysis
fprintf('\nRunning multiple regression analysis...\n');

% Initialize results structure
n_features = length(eeg_feature_names);
results = struct('Feature', cell(n_features, 1), ...
                'R_squared', [], ...
                'Adj_R_squared', [], ...
                'F_statistic', [], ...
                'F_pvalue', [], ...
                'Neurodev_coef', [], ...
                'Neurodev_pvalue', [], ...
                'Genetic_coef', [], ...
                'Genetic_pvalue', [], ...
                'Age_coef', [], ...
                'Age_pvalue', [], ...
                'Sex_coef', [], ...
                'Sex_pvalue', [], ...
                'N_samples', []);  % Add field for number of samples used

% Create design matrix
X = [ones(height(data), 1), data.EEG_age, sex_binary, ...
     diagnostic_group == 1, diagnostic_group == 2];

% Check for multicollinearity
corr_matrix = corr(X(:,2:end)); % Exclude intercept
if any(abs(corr_matrix(tril(ones(size(corr_matrix)),-1)==1)) > 0.9)
    warning('High multicollinearity detected in predictors. Results may be unreliable.');
end

% Process all features, handling missing values case by case
for i = 1:n_features
    % Get current feature
    y = eeg_features_norm(:, i);
    valid_idx = ~isnan(y);
    
    % Skip if not enough valid samples
    if sum(valid_idx) < size(X, 2)
        fprintf('Warning: Not enough valid samples for feature %s (%d/%d). Skipping.\n', ...
            eeg_feature_names{i}, sum(valid_idx), size(X, 2));
        continue;
    end
    
    y_valid = y(valid_idx);
    X_valid = X(valid_idx, :);
    
    % Fit model
    [b, bint, r, rint, stats] = regress(y_valid, X_valid);
    
    % Store results
    results(i).Feature = eeg_feature_names{i};
    results(i).N_samples = sum(valid_idx);
    results(i).R_squared = stats(1);
    results(i).F_statistic = stats(2);
    results(i).F_pvalue = stats(3);
    results(i).Neurodev_coef = b(4);
    results(i).Genetic_coef = b(5);
    results(i).Age_coef = b(2);
    results(i).Sex_coef = b(3);
    
    % Calculate p-values
    df = length(y_valid) - length(b);
    t_stats = b ./ sqrt(diag(inv(X_valid'*X_valid))*stats(4));
    p_values = 2 * (1 - tcdf(abs(t_stats), df));
    
    results(i).Neurodev_pvalue = p_values(4);
    results(i).Genetic_pvalue = p_values(5);
    results(i).Age_pvalue = p_values(2);
    results(i).Sex_pvalue = p_values(3);
end

%% Apply FDR correction
% Implement Benjamini-Hochberg FDR correction
function p_adj = bh_fdr(p)
    m = length(p);
    [~, idx] = sort(p);
    p_adj = zeros(size(p));
    p_adj(idx) = min(1, p(idx) .* m ./ (1:m)');
end

neurodev_pvals = [results.Neurodev_pvalue];
genetic_pvals = [results.Genetic_pvalue];

% Apply FDR correction
neurodev_pvals_adj = bh_fdr(neurodev_pvals);
genetic_pvals_adj = bh_fdr(genetic_pvals);

for i = 1:n_features
    results(i).Neurodev_pvalue_adj = neurodev_pvals_adj(i);
    results(i).Genetic_pvalue_adj = genetic_pvals_adj(i);
end

%% Save results
fprintf('\nSaving results...\n');

% Convert results structure to table
results_table = struct2table(results);

% Save to CSV
writetable(results_table, fullfile(output_dir, 'regression_results.csv'));

% Create and save summary report
fid = fopen(fullfile(output_dir, 'analysis_summary.txt'), 'w');
fprintf(fid, 'Statistical Analysis Summary\n');
fprintf(fid, '==========================\n\n');
fprintf(fid, 'Total features analyzed: %d\n', n_features);
fprintf(fid, 'Features significant for Neurodevelopmental Group: %d\n', ...
    sum([results.Neurodev_pvalue_adj] < 0.05));
fprintf(fid, 'Features significant for Genetic group: %d\n\n', ...
    sum([results.Genetic_pvalue_adj] < 0.05));

% Write significant features details
sig_neurodev = [results.Neurodev_pvalue_adj] < 0.05;
sig_genetic = [results.Genetic_pvalue_adj] < 0.05;

if any(sig_neurodev)
    fprintf(fid, '\nSignificant Features for Neurodevelopmental Group:\n');
    fprintf(fid, '--------------------------------------------\n');
    for i = find(sig_neurodev)
        fprintf(fid, '\nFeature: %s\n', results(i).Feature);
        fprintf(fid, 'Coefficient: %.4f\n', results(i).Neurodev_coef);
        fprintf(fid, 'P-value (FDR corrected): %.4e\n', results(i).Neurodev_pvalue_adj);
        fprintf(fid, 'R-squared: %.4f\n', results(i).R_squared);
        fprintf(fid, 'Covariates:\n');
        fprintf(fid, '  Age: coef = %.4f, p = %.4e\n', results(i).Age_coef, results(i).Age_pvalue);
        fprintf(fid, '  Sex: coef = %.4f, p = %.4e\n', results(i).Sex_coef, results(i).Sex_pvalue);
    end
end

if any(sig_genetic)
    fprintf(fid, '\nSignificant Features for Genetic Carrier Group:\n');
    fprintf(fid, '------------------------------------------\n');
    for i = find(sig_genetic)
        fprintf(fid, '\nFeature: %s\n', results(i).Feature);
        fprintf(fid, 'Coefficient: %.4f\n', results(i).Genetic_coef);
        fprintf(fid, 'P-value (FDR corrected): %.4e\n', results(i).Genetic_pvalue_adj);
        fprintf(fid, 'R-squared: %.4f\n', results(i).R_squared);
        fprintf(fid, 'Covariates:\n');
        fprintf(fid, '  Age: coef = %.4f, p = %.4e\n', results(i).Age_coef, results(i).Age_pvalue);
        fprintf(fid, '  Sex: coef = %.4f, p = %.4e\n', results(i).Sex_coef, results(i).Sex_pvalue);
    end
end

fclose(fid);

fprintf('\nAnalysis complete! Results saved in: %s\n', output_dir); 