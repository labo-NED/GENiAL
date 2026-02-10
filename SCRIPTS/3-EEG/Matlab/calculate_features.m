% Sample code to calculate features from EEG data
% Author: Saeideh Davoodi
% Date: Feb-09 2026

clear
clc
eeglab; close

directory = '';   % directory that contains the data
listing = dir(fullfile(directory,'*.set'));


names = {listing.name};

f_range = [1, 55];
settings = struct('peak_width_limits', [0.5, 18], ...
    'max_n_peaks', 10);




for sub = 1:length(names)
    id = names{sub};
    EEG = pop_loadset([directory,'\',id]);
    EEG_label = {EEG.chanlocs.labels};
    Fs = EEG.srate;

    for ch = 1:EEG.nbchan

        % m,r,scales must be defined; set get_mse.m m=2, r=0.15, and scales=[1:40]
        
        [mean_mse(ch,:,sub),~, ~, scales]= get_mse(squeeze(EEG.data(ch,:,:)),m,r,scales);

        ci = squeeze(mean_mse(ch,:,sub));
        CI(sub,ch) = trapz(scales,ci);
        CI_lowScale(sub,ch) = trapz(scales(1:floor(length(scales)/2)),ci(1:floor(length(scales)/2)));
        CI_highScale(sub,ch) = trapz(scales(floor(length(scales)/2)+1:length(scales)),ci(floor(length(scales)/2)+1:length(scales)));
        for tr = 1:size(EEG.data,3)
            % Hurst exponent
            hurst1(tr) = estimate_hurst_exponent(squeeze(EEG.data(ch,:,tr)));
            signal = squeeze(EEG.data(ch,:,tr));

            % sample entropy
            % m,r must be defined; set m=2, r=0.15
            sampe(tr) = sampen(squeeze(EEG.data(ch,:,tr)),m,r);

            % power spectra
            nfft = length (signal); %signal number of points
            window = hamming(1000);
            N = nfft/2; %signal number of points
            noverlap = length(window)/2;
            [pxx1(tr,:),f]= pwelch(signal,window,noverlap,N,Fs); %
        end

        SE(sub,ch) = mean(sampe);
        hurst(ch,sub) = mean(hurst1);
        pxx = mean(pxx1,1);
        pxx_t(ch,sub,:) = pxx;
        fooof_results = fooof(f', squeeze(pxx)', f_range, settings, true);
        fooof_results_t{sub,ch} = fooof_results;

        %compute band power
        f1 = find(f>=1 & f<=3.5);
        PSD(sub,ch,1) = mean(pxx(f1(1):f1(end)));

        f2 = find(f>=4 & f<=7);
        PSD(sub,ch,2) = mean(pxx(f2(1):f2(end)));

        f3 = find(f>=7.5 & f<=12.5);
        PSD(sub,ch,3) = mean(pxx(f3(1):f3(end)));

        f4 = find(f>=13 & f<=30);
        PSD(sub,ch,4) = mean(pxx(f4(1):f4(end)));

        f5 = find(f>=31 & f<=57);
        PSD(sub,ch,5) = mean(pxx(f5(1):f5(end)));

        f6 = find(f>=62 & f<=80);
        PSD(sub,ch,6) = mean(pxx(f6(1):f6(end)));

        f7 = find(f>=0.5 & f<=80);
        PSD(sub,ch,7) = mean(pxx(f7(1):f7(end)));


        for band = 1:size(PSD,3)-1
            relative_PSD(sub,ch,band) = PSD(sub,ch,band)/sum(PSD(sub,ch,:));
        end

        aperiodic_component = fooof_results.ap_fit;
        full_spectrum  = fooof_results.power_spectrum;
        periodic_component  = full_spectrum-aperiodic_component;

        aperiodic_component_t(sub,ch,:) = aperiodic_component;
        full_spectrum_t(sub,ch,:) = full_spectrum;
        periodic_component_t(sub,ch,:) = periodic_component;
        fooofed_spectrum_t(sub,ch,:) = fooof_results.fooofed_spectrum;

        offset(sub,ch) = fooof_results.aperiodic_params(1);
        exponent(sub,ch) = fooof_results.aperiodic_params(2);

        % periodic PSD as the area under the PSD curve
        periodic_PSD(sub,ch,1) = trapz(periodic_component(f1(1):f1(end)));

        periodic_PSD(sub,ch,2) = trapz(periodic_component(f2(1):f2(end)));

        periodic_PSD(sub,ch,3) = trapz(periodic_component(f3(1):f3(end)));

        periodic_PSD(sub,ch,4) = trapz(periodic_component(f4(1):f4(end)));

        periodic_PSD(sub,ch,5) = trapz(periodic_component(f5(1):55));


        % periodic PSD as the average
        periodic_PSD_m(sub,ch,1) = mean(periodic_component(f1(1):f1(end)));

        periodic_PSD_m(sub,ch,2) = mean(periodic_component(f2(1):f2(end)));

        periodic_PSD_m(sub,ch,3) = mean(periodic_component(f3(1):f3(end)));

        periodic_PSD_m(sub,ch,4) = mean(periodic_component(f4(1):f4(end)));

        periodic_PSD_m(sub,ch,5) = mean(periodic_component(f5(1):55));



    end

end


clear feature

feature.aperiodic_component_t = aperiodic_component_t;
feature.APF_fooof = APF_fooof;
feature.APF_fooof_ROI = APF_fooof_ROI;
feature.exponent = exponent;
feature.fooof_results_t = fooof_results_t;
feature.fooofed_spectrum_t = fooofed_spectrum_t;
feature.full_spectrum_t = full_spectrum_t;
feature.hurst = hurst';
feature.offset = offset;
feature.periodic_component_t = periodic_component_t;
feature.periodic_PSD = periodic_PSD;
feature.periodic_PSD_m = periodic_PSD_m;
feature.PSD = PSD;
feature.sgf_t = sgf_t;
feature.relative_PSD = relative_PSD;
feature.CI = CI;
feature.CI_lowScale = CI_lowScale;
feature.CI_highScale = CI_highScale;
feature.SE = SE;
feature.MSE = mean_mse;
feature.name = names;

save(directory ,'feature')