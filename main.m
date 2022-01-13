clear all
clc
close all

%% params
threshold=0.7; %correlation threshold to poor discard pairwise registrations
subsampling_rate=8; %subsampling rate
num_sequential=20; %number of sequential registrations to always have
temporal_lambda=0; %amount of temporal smoothness
num_bins=1000; %number of time bins to divide data into
%% read data
data=h5read('Pt02_2.h5','/raster');
% data=h5read('mg29_lfpraster.h5','/raster')';

%% filter, normalize and resize data
for i=1:100
    data=medfilt2(data);
end
for t=1:20
data=zscore(data,[],1);
data=zscore(data,[],2);
end
data=imresize(data,[size(data,1) num_bins]);

%% register data (data_reg) and output position/motion estimates (p)
[data_reg,p]=DCreg(data,threshold,subsampling_rate,num_sequential,temporal_lambda);

%% visualize un-registered + registered raster
figure('units','normalized','outerposition',[0 0 1 1/3])
subplot(1,3,1)
imagesc(data);
xlabel('Time bins')
ylabel('Channels')
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
title('Unregistered')
subplot(1,3,2)
imagesc(data_reg);
xlabel('Time bins')
ylabel('Channels')
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
title('Registered')
colormap(othercolor('BuDRd_12'));
subplot(1,3,3)
plot(p,'.');
xlabel('Time bins')
ylabel('Displacement')
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
title('Displacement estimate');