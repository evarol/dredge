function [ap_output_file, lfp_output_file] = Intarpolate_and_align_AP_2022(data_file)
%function [output_file] = Intarpolate_and_align_AP_2022(name, data_file, num_of_channels, distortion, channels, start_point, end_point)

% no output - this will save a new bin file and additional information in
% a mat files

%%%
%NOT READY TO USE YET 
% addpath('D:\Git_repos\matlab\CorticalNeuropixelProcessingPipeline\util');
PITCH = 20;
%%% Load data:
load(data_file);
orig_map = load(channel_map_file);
global_timestamps = load(timestamps_file);
num_of_raw_channels = double(num_of_raw_channels);
first_AP_sample = find(global_timestamps.AP_timestamp >= AP_timestamp(1),1,'first');
first_LFP_sample = find(global_timestamps.LFP_timestamp >= LFP_timestamp(1),1,'first');

num_AP_samples = length(p_csd_AP_Fs);
num_LFP_samples = length(p_csd_lfp_Fs);

AP_memmap = memmapfile(ap_file,'Format', {'int16', [num_of_raw_channels, num_AP_samples], 'data'}, 'Offset', first_AP_sample*num_of_raw_channels*2);
LF_memmap = memmapfile(lfp_file, 'Format',{'int16', [num_of_raw_channels, num_LFP_samples], 'data'}, 'Offset', first_LFP_sample*num_of_raw_channels*2);


columns = round(384/length(unique(orig_map.ycoords)));    % should be either 2 or 4 (either use two columns or 4 columns)
if columns == 1
    [channels{1}(:,2),channels{1}(:,1)] = sort(orig_map.ycoords);
elseif columns == 2
    [~,temp] = sort(orig_map.ycoords);
    channels{1}(:,1) = temp(orig_map.chanMap(orig_map.xcoords == 11 | orig_map.xcoords == 27));
    channels{1}(:,2) = orig_map.ycoords(channels{1}(:,1));
    channels{2}(:,1) = temp(orig_map.chanMap(orig_map.xcoords == 43 | orig_map.xcoords == 59));
    channels{2}(:,2) = orig_map.ycoords(channels{2}(:,1));
    
else
    error('Verify channel map - and adapt algorithm accordingly')
end 




% remove median to maintain maximum avilable data in 384 channels:
p_csd_AP_Fs = p_csd_AP_Fs - median(p_csd_lfp_Fs);
p_csd_lfp_Fs = p_csd_lfp_Fs - median(p_csd_lfp_Fs);


% open new file:
[filepath,~,~] = fileparts(data_file);

output_folder = [filepath,'\','aligned_AP_to_kilosort\'];
if ~exist(output_folder,'dir')
    mkdir(output_folder)
end
[~,lfp_file_name,ext] = fileparts(lfp_file);
lfp_output_file = [output_folder, lfp_file_name ,'_aligned.bin'];
if exist(lfp_output_file, 'file')
    answer = questdlg('File already exists, Overwrite?',lfp_file_name,'No');
    if strcmp(answer, 'No')
       warning('no permission to overwrite')
       return
    end
end

[~,ap_file_name,ext] = fileparts(ap_file);
ap_output_file = [output_folder, ap_file_name ,'_aligned.bin'];
if exist(ap_output_file, 'file')
    answer = questdlg('File already exists, Overwrite?',ap_file_name,'No');
    if strcmp(answer, 'No')
       warning('no permission to overwrite')
       return
    end
end
%% LFP:
%%% Get channel stats for zscoring:
disp(['zscore LFP']);
for ch = 1:384
    ch_data = double(LF_memmap.Data.data(ch,:));
    ch_stat(ch,1) = mean(ch_data);
    ch_stat(ch,2) = std(rmoutliers(ch_data - ch_stat(ch,1),'percentiles',[10 90]));
end
% apply temporal gaussiam smoothing:
winsize = 5;
gauss_win = gausswin(winsize)/sum(gausswin(winsize));

fid_lfp_write = fopen(lfp_output_file,'w');

for sample = 1 : num_LFP_samples 
    if mod(sample, round(num_LFP_samples/10)) == 0
        disp(['LFP Progress: ',num2str(round(sample/num_LFP_samples*100)),'%']);
    end
    try
        sample_data = double(LF_memmap.Data.data(1:384,sample-(winsize-1)/2:sample+(winsize-1)/2));
        sample_data = sample_data*gauss_win;
    catch
        sample_data = double(LF_memmap.Data.data(1:384,sample));
    end
    % zscore correction:
    sample_data = (sample_data - ch_stat(:,1)) ./ ch_stat(:,2);
     
    % interpolation:
    aligned_sample_data = zeros(size(sample_data));
    for col = 1:columns
        cur_ch = channels{col}(:,1);
        y_val = channels{col}(:,2);
        cur_data = sample_data(cur_ch);
        actual_y = y_val - PITCH * (p_csd_lfp_Fs(sample));
        new_data = interp1(actual_y, cur_data, y_val,'spline',0);
        
        out_of_bound = find(y_val>max(actual_y) | y_val<min(actual_y));
        
        %assign random values to out of bound data - to avoid 'jumps' or
        %explosion of unassigned data:
        new_data(out_of_bound) = cur_data(randperm(size(cur_data,1),size(out_of_bound,1))); 
        
        %merge columns:
        aligned_sample_data(channels{col}(:,1)) = new_data;
    end
    
    % Convert to int16:
    aligned_sample_data = int16(aligned_sample_data*800);
    
    % add orig. TTL value:
    if num_of_raw_channels == 385
        aligned_sample_data(385) = LF_memmap.Data.data(385,sample);
    end
    
    %write to file:
    fwrite(fid_lfp_write, aligned_sample_data, 'int16' );
end
fclose(fid_lfp_write)
copyfile(data_file, output_folder);
disp('LFP Done!')

%% AP:
%%% Get channel stats for zscoring:
disp(['zscore AP']);

for ch = 1:384
    ch_data = double(AP_memmap.Data.data(ch,:));
    ch_stat(ch,1) = mean(ch_data);
    ch_stat(ch,2) = std(rmoutliers(ch_data - ch_stat(ch,1),'percentiles',[10 90]));
end
% apply temporal gaussiam smoothing:
winsize = 9;
gauss_win = gausswin(winsize)/sum(gausswin(winsize));

fid_ap_write = fopen(ap_output_file,'w');

for sample = 1 : num_AP_samples 
    if mod(sample, round(num_AP_samples/10)) == 0
        disp(['AP Progress: ',num2str(round(sample/num_AP_samples*100)),'%']);
    end
    try
        sample_data = double(AP_memmap.Data.data(1:384,sample-(winsize-1)/2:sample+(winsize-1)/2));
        sample_data = sample_data*gauss_win;
    catch
        sample_data = double(AP_memmap.Data.data(1:384,sample));
    end
    % zscore correction:
    sample_data = (sample_data - ch_stat(:,1)) ./ ch_stat(:,2);
     
    % interpolation:
    aligned_sample_data = zeros(size(sample_data));
    for col = 1:columns
        cur_ch = channels{col}(:,1);
        y_val = channels{col}(:,2);
        cur_data = sample_data(cur_ch);
        actual_y = y_val - PITCH * (p_csd_AP_Fs(sample));
        new_data = interp1(actual_y, cur_data, y_val,'spline',0);
        
        out_of_bound = find(y_val>max(actual_y) | y_val<min(actual_y));
        
        %assign random values to out of bound data - to avoid 'jumps' or
        %explosion of unassigned data:
        new_data(out_of_bound) = cur_data(randperm(size(cur_data,1),size(out_of_bound,1))); 
        
        %merge columns:
        aligned_sample_data(channels{col}(:,1)) = new_data;
    end
    
    % Convert to int16:
    aligned_sample_data = int16(aligned_sample_data*800);
    
    % add orig. TTL value:
    if num_of_raw_channels == 385
        aligned_sample_data(385) = AP_memmap.Data.data(385,sample);
    end
    
    %write to file:
    fwrite(fid_ap_write, aligned_sample_data, 'int16' );
end
copyfile(data_file, output_folder);
fclose(fid_ap_write);
disp('AP Done!')
