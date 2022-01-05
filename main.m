clear all
clc
close all

%% params
threshold=0.7; %correlation threshold to poor pairwise registrations
subsampling_rate=8; %subsampling rate - reduce this increase speed and decrease quality
num_sequential=4; %number of sequential registrations to always have i.e. register t=5 to t=1,2,3,4 and 6,7,8,9 -- increase this to improve quality and decrease speed
num_bins=1000; %number of time bins to divide data into - resizing the samples to larger time bins for speed
%% read data
data=h5read('mg29_lfpraster.h5','/raster')';
data=imresize(data,[size(data,1) num_bins]);

%% register data (data_reg) and output position/motion estimates (p)
[data_reg,p]=DCreg(data,threshold,subsampling_rate,num_sequential);

%% visualize un-registered + registered raster
figure
subplot(1,2,1)
imagesc(data);
xlabel('Time bins')
ylabel('Channels')
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
title('Unregistered')
subplot(1,2,2)
imagesc(data_reg);
xlabel('Time bins')
ylabel('Channels')
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
title('Registered')
colormap(othercolor('BuDRd_12'));