% === Load feature.mat ===
load('/Volumes/NED_Backup3/Q1K_Preprocessed_2s_Happe/5 - processed/feature.mat');

% === Load any EEG file to define spatial ROIs ===
sample_set = fullfile('/Volumes/NED_Backup3/Q1K_Preprocessed_Happe/5 - processed', feature.name{1});
chan_labels = {EEG.chanlocs.labels};
theta = [EEG.chanlocs.theta];

% === Define ROIs by angle (theta in degrees) ===
ROIs = struct();
ROIs.Frontal    = chan_labels(theta >= -90 & theta <= 90);
ROIs.Central    = chan_labels(abs(theta) < 30);
ROIs.Parietal   = chan_labels((theta > 90 & theta <= 135) | (theta < -90 & theta >= -135));
ROIs.Occipital  = chan_labels(theta > 135 | theta < -135);
ROIs.Temporal_L = chan_labels(theta > 30 & theta <= 90);
ROIs.Temporal_R = chan_labels(theta < -30 & theta >= -90);

% === Convert channel names to indices ===
ROI_indices = struct();
roi_names = fieldnames(ROIs);
for i = 1:numel(roi_names)
    roi = roi_names{i};
    ROI_indices.(roi) = find(ismember(chan_labels, ROIs.(roi)));
end

% === Band labels for periodic PSD (AUC) ===
band_labels = {'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'};

% === Initialize result table ===
T = table(feature.name(:), 'VariableNames', {'SubjectID'});
nSub = size(feature.offset, 1);

% === Loop through ROIs and compute feature means ===
for i = 1:numel(roi_names)
    roi = roi_names{i};
    idx = ROI_indices.(roi);

    % Aperiodic
    T.([roi '_Offset'])   = mean(feature.offset(:, idx), 2, 'omitnan');
    T.([roi '_Exponent']) = mean(feature.exponent(:, idx), 2, 'omitnan');

    % Periodic PSD (AUC) per band
    for b = 1:length(band_labels)
        label = band_labels{b};
        T.([roi '_PeriodicPSD_' label]) = mean(feature.periodic_PSD(:, idx, b), 2, 'omitnan');
    end

    % Relative theta (optional)
    T.([roi '_RelThetaPSD']) = mean(feature.relative_PSD(:, idx, 2), 2, 'omitnan');
end

% === Export CSV ===
out_path = '/Volumes/NED_Backup3/Q1K_Preprocessed_2s_Happe/5 - processed/features_ROI_summary.csv';
writetable(T, out_path);
fprintf('✅ Exported ROI-wise feature summary to:\n%s\n', out_path);
