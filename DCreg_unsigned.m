function [data_reg,p]=DCreg_unsigned(data,threshold,subsampling_rate,num_sequential)
%% decentralized registration
Dp=nan(size(data,2));% pre-allocate displacement + correlation matrices
Cp=nan(size(data,2));

S=zeros(size(data,2));
% pairwise subsampled displacement estimation
for i=1:size(data,2)
    for t=1:size(data,2)
        if or(rand(1)<subsampling_rate*log(size(data,2))/size(data,2),abs(i-t)<=num_sequential)
            S(i,t)=1;
            [x,c]=CXCORR(data(:,i)',data(:,t)');
            [Cp(i,t),idx]=max(c);
            Dp(i,t)=x(idx);
            if mod(t,100)==0
                figure(1)
                imagesc(Dp(1:i,:));drawnow
            end
            
        end
    end
end

%% decentralized registration
Dn=nan(size(data,2));% pre-allocate displacement + correlation matrices
Cn=nan(size(data,2));


% pairwise subsampled displacement estimation
for i=1:size(data,2)
    for t=1:size(data,2)
        if S(i,t)==1
            [x,c]=CXCORR(data(:,i)',-data(:,t)');
            [Cn(i,t),idx]=max(c);
            Dn(i,t)=x(idx);
            if mod(t,100)==0
                figure(1)
                imagesc(Dn(1:i,:));drawnow
            end
            
        end
    end
end
D=nan(size(data,2));
D(Cp>=Cn)=Dp(Cp>=Cn);
D(Cn>=Cp)=Dn(Cn>=Cp);
C=nan(size(data,2));
C(Cp>=Cn)=Cp(Cp>=Cn);
C(Cn>=Cp)=Cn(Cn>=Cp);
% visualize displacement + correlation matrices
figure(2)
subplot(2,2,1)
imagesc(Dp);colorbar;axis square
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
xlabel('Time bins');
ylabel('Time bins')
title('Subsampled Displacement+ Matrix');
subplot(2,2,2)
imagesc(Cp);colorbar;axis square
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
xlabel('Time bins');
ylabel('Time bins')
title('Subsampled XCorr+ Matrix');
colormap(othercolor('BuDRd_12'));
subplot(2,2,3)
imagesc(Dn);colorbar;axis square
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
xlabel('Time bins');
ylabel('Time bins')
title('Subsampled Displacement- Matrix');
subplot(2,2,4)
imagesc(Cn);colorbar;axis square
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
xlabel('Time bins');
ylabel('Time bins')
title('Subsampled XCorr- Matrix');
colormap(othercolor('BuDRd_12'));


figure(3)
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

% compute centralized position estimates
try;p=psolver(D);catch;p=zeros(size(data,2));end


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
end

end