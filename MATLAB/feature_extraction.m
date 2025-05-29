clear;
clc;
eeglab; close;

%% --- Load EEG Files ---
directory = '/Volumes/NED_Backup3/Q1K_Preprocessed_2s_Happe/5 - processed'; 
listing = dir(fullfile(directory, '*.set'));
file_names = {listing.name};
names = file_names(~startsWith(file_names, '._')); % Remove macOS hidden files

f_range = [1, 40];
settings = struct('peak_width_limits', [0.5, 18], 'max_n_peaks', 10);

%% Loop through each participant
for sub = 1:length(names)
    id = names{sub};
    EEG = pop_loadset(fullfile(directory, id)); % Load EEG file
    EEG_label = {EEG.chanlocs.labels};
    Fs = EEG.srate;

    for ch = 1:EEG.nbchan

        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % %% --- Multiscale Entropy (MSE) & Complexity Index (DISABLED) ---
        % m = 2; r = 0.5; scales = 1:20;
        % [mean_mse(ch,:,sub), ~, ~, scales] = get_mse(squeeze(EEG.data(ch,:,:)), m, r, scales);
        %
        % ci = squeeze(mean_mse(ch,:,sub));
        % CI(sub,ch) = trapz(scales, ci);
        % CI_lowScale(sub,ch) = trapz(scales(1:floor(end/2)), ci(1:floor(end/2)));
        % CI_highScale(sub,ch) = trapz(scales(floor(end/2)+1:end), ci(floor(end/2)+1:end));
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for tr = 1:size(EEG.data, 3)
            signal = squeeze(EEG.data(ch,:,tr));

            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % %% --- Sample Entropy (DISABLED) ---
            % sampe(tr) = sampen(signal, 2, 0.5);
            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % %% --- Hurst Exponent ---
            hurst1(tr) = estimate_hurst_exponent(signal);

            %% --- Power Spectral Density (Welch) ---
            window = hamming(1000);
            N = length(signal)/2;
            noverlap = length(window)/2;
            [pxx1(tr,:), f] = pwelch(signal, window, noverlap, N, Fs);
        end

        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % SE(sub,ch) = mean(sampe);
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        hurst(ch,sub) = mean(hurst1);
        
        pxx = mean(pxx1, 1);
        pxx_t(ch,sub,:) = pxx;

        %% --- FOOOF Spectral Decomposition ---
        fooof_results = fooof(f', squeeze(pxx)', f_range, settings, true);
        fooof_results_t{sub,ch} = fooof_results;

        %% --- Absolute Band Power ---
        f1 = find(f>=1 & f<=3.5);     % Delta
        f2 = find(f>=4 & f<=7);       % Theta
        f3 = find(f>=7.5 & f<=12.5);  % Alpha
        f4 = find(f>=13 & f<=30);     % Beta
        f5 = find(f>=31 & f<=57);     % Gamma (low)
        f6 = find(f>=62 & f<=80);     % Gamma (high)
        f7 = find(f>=0.5 & f<=80);    % Full spectrum

        PSD(sub,ch,1) = mean(pxx(f1));
        PSD(sub,ch,2) = mean(pxx(f2));
        PSD(sub,ch,3) = mean(pxx(f3));
        PSD(sub,ch,4) = mean(pxx(f4));
        PSD(sub,ch,5) = mean(pxx(f5));
        PSD(sub,ch,6) = mean(pxx(f6));
        PSD(sub,ch,7) = mean(pxx(f7));

        %% --- Relative Band Power ---
        for band = 1:size(PSD,3)-1
            relative_PSD(sub,ch,band) = PSD(sub,ch,band) / sum(PSD(sub,ch,:));
        end

        %% --- FOOOF Components ---
        aperiodic_component = fooof_results.ap_fit;
        full_spectrum = fooof_results.power_spectrum;
        periodic_component = full_spectrum - aperiodic_component;

        aperiodic_component_t(sub,ch,:) = aperiodic_component;
        full_spectrum_t(sub,ch,:) = full_spectrum;
        periodic_component_t(sub,ch,:) = periodic_component;
        fooofed_spectrum_t(sub,ch,:) = fooof_results.fooofed_spectrum;

        offset(sub,ch) = fooof_results.aperiodic_params(1);
        exponent(sub,ch) = fooof_results.aperiodic_params(2);

        %% --- Periodic Band Power (AUC and Mean) ---
        periodic_PSD(sub,ch,1) = trapz(periodic_component(f1));
        periodic_PSD(sub,ch,2) = trapz(periodic_component(f2));
        periodic_PSD(sub,ch,3) = trapz(periodic_component(f3));
        periodic_PSD(sub,ch,4) = trapz(periodic_component(f4));
        periodic_PSD(sub,ch,5) = trapz(periodic_component(f5(f5<=40)));

        periodic_PSD_m(sub,ch,1) = mean(periodic_component(f1));
        periodic_PSD_m(sub,ch,2) = mean(periodic_component(f2));
        periodic_PSD_m(sub,ch,3) = mean(periodic_component(f3));
        periodic_PSD_m(sub,ch,4) = mean(periodic_component(f4));
        periodic_PSD_m(sub,ch,5) = mean(periodic_component(f5(f5<=40)));
    end
end

%% --- Save Extracted Features (Without Entropy) ---
clear feature
feature.aperiodic_component_t = aperiodic_component_t;
feature.fooof_results_t = fooof_results_t;
feature.fooofed_spectrum_t = fooofed_spectrum_t;
feature.full_spectrum_t = full_spectrum_t;
feature.offset = offset;
feature.exponent = exponent;
feature.periodic_component_t = periodic_component_t;
feature.periodic_PSD = periodic_PSD;
feature.periodic_PSD_m = periodic_PSD_m;
feature.PSD = PSD;
feature.relative_PSD = relative_PSD;
feature.APF_fooof = APF_fooof;
feature.APF_fooof_ROI = APF_fooof_ROI;
feature.sgf_t = sgf_t;
feature.hurst = hurst';
feature.pxx_t = pxx_t;
feature.name = names;

% Optional: Remove or comment out undefined fields if not computed yet
% feature.SE = SE;
% feature.CI = CI;
% feature.CI_lowScale = CI_lowScale;
% feature.CI_highScale = CI_highScale;
% feature.MSE = mean_mse;

save(fullfile(directory, 'feature.mat'), 'feature');
