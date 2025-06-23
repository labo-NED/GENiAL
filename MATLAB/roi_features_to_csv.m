% === Load feature.mat ===
load('/Volumes/NED_Backup3/Q1K_Preprocessed_2s_Happe/5 - processed/feature.mat');

% === Load any EEG file to define spatial ROIs ===
sample_set = fullfile('/Volumes/NED_Backup3/Q1K_Preprocessed_Happe/5 - processed', feature.name{1});
chan_labels = {EEG.chanlocs.labels};
theta = [EEG.chanlocs.theta];

% === Define ROIs by channel names ===
F  = {'Fp1','Fp2','F3','Fz','F4'};
Cz = {'FC1','FC2','C3','Cz','C4','CP1','CP2'};
TR = {'F8','FC6','T8'};
TL = {'F7','FC5','T7'};
PR = {'CP6','P4','P8'};
PL = {'CP5','P3','P7'};
Oz = {'Pz','O1','Oz','O2'};
WB = {chan_labels{:}}; % All channels

ROIs = struct();
ROIs.F  = F;
ROIs.Cz = Cz;
ROIs.TR = TR;
ROIs.TL = TL;
ROIs.PR = PR;
ROIs.PL = PL;
ROIs.Oz = Oz;
ROIs.WB = WB;

% === Convert channel names to indices ===
ROI_indices = struct();
roi_names = fieldnames(ROIs);
for i = 1:numel(roi_names)
    roi = roi_names{i};
    ROI_indices.(roi) = find(ismember(chan_labels, ROIs.(roi)));
end

% === Band labels for periodic PSD (AUC) ===
band_labels = {'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'};

% === Extract SubjectID and Task from filenames ===
SubjectID = cell(size(feature.name));
Task = cell(size(feature.name));
expr = '^(.*?)_((?i)RSRIO|RS)(?:_|$)';

for i = 1:numel(feature.name)
    tokens = regexp(feature.name{i}, expr, 'tokens');
    if ~isempty(tokens)
        SubjectID{i} = tokens{1}{1};
        Task{i} = upper(tokens{1}{2});
    else
        SubjectID{i} = '';
        Task{i} = '';
    end
end

% === Initialize result table ===
T = table(SubjectID(:), Task(:), 'VariableNames', {'SubjectID', 'Task'});
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
