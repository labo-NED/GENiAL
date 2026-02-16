clear
clc
eeglab; close

directory = '';   % directory that contains the data
listing = dir(fullfile(directory,'*.set'));


names = {listing.name};

for sub = 1:length(names)
    id = names{sub};
    EEG = pop_loadset([directory,'\',id]);
    EEG_segment_num(sub ) = size(EEG.data,3);
end


%%%%%%%%%%%%%%%%%%%%%% Choose one of them %%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% if 5s data ---> <30s means <6 segments

segments_lower_than_30s = find(EEG_segment_num<6);
subjects_lower_than_30s = names(segments_lower_than_30s);

%%%% if 2s data ---> <30s means <15 segments

segments_lower_than_30s = find(EEG_segment_num<15);
subjects_lower_than_30s = names(segments_lower_than_30s);