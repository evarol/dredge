
%% pt02:
clear all
addpath([pwd,'\lib\global_timing']);
mfilename('fullpath')
load('T:\IntraOp_Micro\Neuropixelpt02\file2_g0\file2_g0_imec0\ExtractedMatlabData\APChannelPerRecording_385.mat')
AP_data = dataArray;
AP_timestamp = [1:length(AP_data)]/30000;
load('T:\IntraOp_Micro\Neuropixelpt02\file2_g0\file2_g0_imec0\ExtractedMatlabData\LFPChannelPerRecording_385.mat')
LFP_data = dataArray;

daq_bin = 'D:\Neuropixel\Neuropixelpt02\file2_g0\file2_g0_t0.nidq.bin';
fid = fopen(daq_bin,'r');
data = fread(fid,[9,Inf],'int16');
fclose(fid);
DAQ_data = data(1,:);

[DAQ_timestamp] = align_TTL_timing(AP_data, AP_timestamp, DAQ_data);
[LFP_timestamp] = align_TTL_timing(AP_data, AP_timestamp, LFP_data);


save('D:\Neuropixel\Neuropixelpt02\file2_g0\Global_timestamp.mat','AP_timestamp','DAQ_timestamp','LFP_timestamp');
save('D:\Neuropixel\global_alignment\pt02_Global_timestamp.mat','AP_timestamp','DAQ_timestamp','LFP_timestamp');


