% === Load feature.mat ===
load('/Volumes/NED_Backup3/COMBINED_Q1K_BC_2s/GENIAL/feature.mat');

% === Load any EEG file to get channel information ===
sample_set = fullfile('/Volumes/NED_Backup3/COMBINED_Q1K_BC_2s/GENIAL', feature.name{1});
chan_labels = {EEG.chanlocs.labels};
n_channels = length(chan_labels);

% === Band labels for periodic PSD (AUC) ===
band_labels = {'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'};

% === Extract SubjectID and Task from filenames ===
SubjectID = cell(size(feature.name));

for i = 1:numel(feature.name)
    filename = feature.name{i};
    
    % Pattern for Q1K files: Q1K_HSJ_XXXX-YYYY_P_RSRio_... or Q1K_HSJ_XXXX-YYYY_P_RS_...
    q1k_pattern = '^(Q1K_HSJ_\d+-\d+_P)_((?i)RSRIO|RS)(?:_|$)';
    q1k_tokens = regexp(filename, q1k_pattern, 'tokens');
    
    % Pattern for BC files: BC_2017_XXXXX_YYYYYY_P_... -> extract BC_XXXXX
    bc_pattern = '^BC_2017_(\d{5})_';
    bc_tokens = regexp(filename, bc_pattern, 'tokens');
    
    if ~isempty(q1k_tokens)
        SubjectID{i} = q1k_tokens{1}{1};
    elseif ~isempty(bc_tokens)
        SubjectID{i} = ['BC_' bc_tokens{1}{1}];
    else
        % Fallback: try to extract just the subject part before RS/RSRio
        fallback_pattern = '^(.*?)_((?i)RSRIO|RS)(?:_|$)';
        fallback_tokens = regexp(filename, fallback_pattern, 'tokens');
        if ~isempty(fallback_tokens)
            SubjectID{i} = fallback_tokens{1}{1};
        else
            SubjectID{i} = filename; % Use full filename if no pattern matches
        end
    end
end

% === Initialize result table ===
T = table(SubjectID(:), 'VariableNames', {'SubjectID'});
nSub = size(feature.offset, 1);

fprintf('Processing %d participants with %d channels each...\n', nSub, n_channels);

% === Compute global (all-channel) feature means ===

% Aperiodic features - average across all channels
T.Aperiodic_Offset = mean(feature.offset, 2, 'omitnan');
T.Aperiodic_Exponent = mean(feature.exponent, 2, 'omitnan');

% Absolute band power - average across all channels
T.Average_Delta_Power = mean(feature.PSD(:, :, 1), 2, 'omitnan');
T.Average_Theta_Power = mean(feature.PSD(:, :, 2), 2, 'omitnan');
T.Average_Alpha_Power = mean(feature.PSD(:, :, 3), 2, 'omitnan');
T.Average_Beta_Power = mean(feature.PSD(:, :, 4), 2, 'omitnan');
T.Average_Gamma_Power = mean(feature.PSD(:, :, 5), 2, 'omitnan');

% Periodic PSD (AUC) per band - average across all channels
for b = 1:length(band_labels)
    label = band_labels{b};
    T.(['Average_PeriodicPSD_' label]) = mean(feature.periodic_PSD(:, :, b), 2, 'omitnan');
end

% Relative band power - average across all channels
T.Average_RelDelta_Power = mean(feature.relative_PSD(:, :, 1), 2, 'omitnan');
T.Average_RelTheta_Power = mean(feature.relative_PSD(:, :, 2), 2, 'omitnan');
T.Average_RelAlpha_Power = mean(feature.relative_PSD(:, :, 3), 2, 'omitnan');
T.Average_RelBeta_Power = mean(feature.relative_PSD(:, :, 4), 2, 'omitnan');
T.Average_RelGamma_Power = mean(feature.relative_PSD(:, :, 5), 2, 'omitnan');

% Hurst exponent - average across all channels
% T.Average_Hurst = mean(feature.hurst, 1, 'omitnan')';

% === Display summary statistics ===
fprintf('\n=== Global Feature Summary ===\n');
fprintf('Number of participants: %d\n', nSub);
fprintf('Number of channels: %d\n', n_channels);
fprintf('Features computed:\n');
fprintf('  - Aperiodic: Aperiodic_Offset, Aperiodic_Exponent\n');
fprintf('  - Average Absolute Power: Delta, Theta, Alpha, Beta, Gamma\n');
fprintf('  - Average Periodic PSD: Delta, Theta, Alpha, Beta, Gamma\n');
fprintf('  - Average Relative Power: Delta, Theta, Alpha, Beta, Gamma\n');
% fprintf('  - Average Hurst Exponent\n');

% === Export CSV ===
out_path = '/Volumes/NED_Backup3/COMBINED_Q1K_BC_2s/GENIAL/features_global_summary.csv';
writetable(T, out_path);
fprintf('\n✅ Exported global feature summary to:\n%s\n', out_path);

% === Display first few rows as preview ===
fprintf('\n=== Preview of exported data ===\n');
fprintf('First 5 rows:\n');
disp(T(1:min(5, height(T)), :));

% === Optional: Create a copy in the local project directory ===
local_path = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/EEG/GENIAL/RS-2s/features_global_summary.csv';
local_dir = fileparts(local_path);

% Create directory if it doesn't exist
if ~exist(local_dir, 'dir')
    mkdir(local_dir);
    fprintf('📁 Created directory: %s\n', local_dir);
end

try
    writetable(T, local_path);
    fprintf('✅ Also saved local copy to:\n%s\n', local_path);
catch ME
    fprintf('⚠️  Could not save local copy: %s\n', ME.message);
end
