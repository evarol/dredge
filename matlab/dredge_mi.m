function [data_reg,p]=dredge_mi(data,threshold,temporal_lambda)
%% decentralized registration
D=nan(size(data,2));% pre-allocate displacement + correlation matrices
C=nan(size(data,2));


% pairwise subsampled displacement estimation
V_data=var(data,[],1);
M_data=(data-mean(data,1));
for i=1:size(data,2)
    [x,c]=mat_XMI(V_data,M_data,data(:,i));
    [C(i,:),idx]=max(c,[],1);
    D(i,:)=-x(idx);
    if mod(i,10)==0
        figure(1)
        subplot(1,2,1);imagesc(D(1:i,:));drawnow
        subplot(1,2,2);imagesc(C(1:i,:));drawnow
    end
end
Dnew=D;
Cnew=C;
for i=1:size(data,2)
    for j=1:size(data,2)
        if C(i,j)>C(j,i)
            Dnew(j,i)=-D(i,j);
            Dnew(i,j)=D(i,j);
            Cnew(i,j)=C(i,j);
            Cnew(j,i)=C(i,j);
        else
                        Dnew(j,i)=D(j,i);
            Dnew(i,j)=-D(j,i);
            Cnew(i,j)=C(j,i);
            Cnew(j,i)=C(j,i);
        end
    end
end
C=Cnew;clear Cnew
D=Dnew;clear Dnew
            
C(isinf(C))=nan;

% for i=1:size(data,2)
%     for j=1:size(data,2)
%         [x,c]=XMI(data(:,i),data(:,j));
%         [C(i,j),idx]=max(c,[],1);
%         D(i,j)=x(idx);
%         if mod(i,10)==0
%             figure(1)
%             subplot(1,2,1);imagesc(D(1:i,:));drawnow
%             subplot(1,2,2);imagesc(C(1:i,:));drawnow
%         end
%     end
% end
% C=cdf('normal',CLR(C),0,1);
C=cdf('normal',nanzscore(C,[],'all'),0,1);
% visualize displacement + correlation matrices
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
figure(2)
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
try
    p=psolver_TF(D,temporal_lambda);
catch
    p=zeros(size(data,2));
end

% visualize position estimates
figure(3)
plot(p,'.','MarkerSize',10,'LineWidth',2);
set(gca,'FontWeight','bold','FontSize',15,'TickLength',[0 0]);set(gcf,'Color','w');
xlabel('Time bins');
ylabel('Displacement');
title('Displacement estimate');

% de-shift raster to motion correct it
data_reg=data;
for t=1:size(data,2)
    data_reg(:,t)=imtranslate(data(:,t),[0 -round(p(t))]);
end