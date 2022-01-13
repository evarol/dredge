function [data_reg,p,applied]=DCreg_unsigned(data,threshold,subsampling_rate,num_sequential,temporal_lambda,applied)
%% decentralized registration
D=nan(size(data,2));% pre-allocate displacement + correlation matrices
C=nan(size(data,2));


% pairwise subsampled displacement estimation
for i=1:size(data,2)
    for t=i:size(data,2)
        if or(rand(1)<subsampling_rate*log(size(data,2))/size(data,2),abs(i-t)<=num_sequential)
            [x,c]=myXCORR(data(:,i)',data(:,t)');
            [C(i,t),idx]=max(abs(c));
            D(i,t)=x(idx);
            D(t,i)=-x(idx);
            C(t,i)=C(i,t);
            if mod(t,100)==0
                figure(1)
                imagesc(D(1:i,:));drawnow
            end
            
        end
    end
end


figure(2)
subplot(2,2,1)
imagesc(D);colorbar;axis square
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
xlabel('Time bins');
ylabel('Time bins')
title('Subsampled Displacement Matrix');
subplot(2,2,2)
imagesc(C);colorbar;axis square
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
xlabel('Time bins');
ylabel('Time bins')
title('Subsampled XCorr Matrix');
colormap(othercolor('BuDRd_12'));
% remove displacement estimates with poor correlation results
D(C<threshold)=nan;
C(C<threshold)=nan;

% re-visualize displacement + correlation matrices after outlier removal
figure(3)
subplot(2,2,3)
imagesc(D,[-5 5]);colorbar;axis square
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
xlabel('Time bins');
ylabel('Time bins')
title('Outlier Removed Subsampled Displacement Matrix');
subplot(2,2,4)
imagesc(C);colorbar;axis square
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
xlabel('Time bins');
ylabel('Time bins')
title('Outlier Removed Subsampled XCorr Matrix');

%compute centralized position estimates
try
    p=psolver_TF(D,temporal_lambda);
catch
    p=zeros(size(data,2));
end

% visualize position estimates
figure(4)
plot(p,'.','MarkerSize',10,'LineWidth',2);
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
xlabel('Time bins');
ylabel('Displacement');
title('Displacement estimate');

% de-shift raster to motion correct it
for t=1:size(data,2)
    data_reg(:,t)=mycircshift(data(:,t),-round(p(t)));
    applied(:,t)=mycircshift(applied(:,t),-round(p(t)));
    t
end

% visualize un-registered + registered raster
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

end