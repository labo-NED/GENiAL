%% 1. Load Data
data = readtable('Data/Final/GENIAL-DB-preprocessed.csv'); % Enter final dataset filepath

%% 2. Preview Data
disp('Data Summary:');
summary(data);
disp('First few rows:');
head(data);

%% 3. Recode Diagnosis (binary: 'Abnormal' = genetic, 'Normal' = non-genetic, other = 'NA')
% Initialize column with NaNs
data.GeneticDiagnosis = nan(height(data), 1);

% Assign binary labels
data.GeneticDiagnosis(strcmp(data.GeneticStatus, 'Abnormal')) = 1;     % Genetic
data.GeneticDiagnosis(strcmp(data.GeneticStatus, 'Normal')) = 0;       % Non-genetic

% % Remove rows with 'NA' GeneticDiagnosis
% data_clean = data(~isnan(data.GeneticDiagnosis), :);
% 
% %% 4. Outlier Detection & Removal
% % Identify EEG variables
% eeg_vars = data_clean.Properties.VariableNames(contains(data_clean.Properties.VariableNames, 'EEG_'));
% 
% % Z threshold
% zThresh = 3;
% rowsToRemove = false(height(data_clean),1);
% for i = 1:length(eeg_vars)
%     z = zscore(data_clean.(eeg_vars{i}));
%     rowsToRemove = rowsToRemove | abs(z) > zThresh;
% end
% data_clean(rowsToRemove, :) = [];
% 
% %% 5. Normalize EEG Features (z-score across each feature)
% for i = 1:length(eeg_vars)
%     data_clean.(eeg_vars{i}) = normalize(data_clean.(eeg_vars{i}));
% end
% 
% %% 6A. Logistic Regression
% X = data_clean{:, eeg_vars};
% y = data_clean.GeneticDiagnosis;
% 
% % Add intercept column
% X_with_intercept = [ones(size(X,1),1) X];
% [B, dev, stats] = mnrfit(X, double(y) + 1);
% disp('Logistic Regression Coefficients:');
% disp(B);
% disp('p-values:');
% disp(stats.p);
