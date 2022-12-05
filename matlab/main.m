clear
clc
close all


%% params
threshold=0.95; %linkage probability to threshold on (0 to 1)
temporal_lambda=0; %amount of temporal smoothness
num_bins=1000; %number of time bins to divide data into
%% read data
% data=h5read('../Pt02_2.h5','/raster');
data=h5read('../Pt03.h5','/raster')';

%% filter, normalize and resize data
data=imresize(data,[size(data,1) num_bins]);
% 

%% register data (data_reg) and output position/motion estimates (p)
tic;[data_reg_corr,p_corr,D_corr,C_corr,D_unthresholded_corr,C_raw_corr]=dredge(data,threshold,temporal_lambda,'corr');corr_time=toc;
% tic;[data_reg_mi,p_mi,D_mi,C_mi,D_unthresholded_mi,C_raw_mi]=dredge(data,threshold,temporal_lambda,'mi');mi_time=toc;
disp(['DREDGE-corr time: ' num2str(corr_time)]);
disp(['DREDGE-mi time: ' num2str(mi_time)]);

%% visualize un-registered + registered raster
figure('units','normalized','outerposition',[0 0 1 1/3])
subplot(1,4,1)
imagesc(data,[quantile(data(:),0.01) quantile(data(:),0.99)]);
xlabel('Time bins')
ylabel('Channels')
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
title('Unregistered')
hold on
plot(100+p_corr,'g.');
plot(100+p_mi,'m.');
legend('DREDGE-corr','DREDGE-mi');
subplot(1,4,2)
imagesc(data_reg_corr,[quantile(data(:),0.01) quantile(data(:),0.99)]);
xlabel('Time bins')
ylabel('Channels')
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
title('Registered (DREDGE-corr)')
colormap(flipud(colormap(gray)));
subplot(1,4,3)
imagesc(data_reg_mi,[quantile(data(:),0.01) quantile(data(:),0.99)]);
xlabel('Time bins')
ylabel('Channels')
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
title('Registered (DREDGE-mi)')
colormap(flipud(colormap(gray)));
subplot(1,4,4)
hold on
plot(p_corr,'.');
plot(p_mi,'.');
legend('DREDGE-corr','DREDGE-mi');
xlabel('Time bins')
ylabel('Displacement')
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
title('Displacement estimate');


