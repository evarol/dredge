function [data_reg,p,D,C,D_unthresholded,C_raw]=dredge(data,threshold,temporal_lambda,type)
%% decentralized registration
D=nan(size(data,2));% pre-allocate displacement + correlation matrices
C=nan(size(data,2));


%% pairwise subsampled displacement estimation
V_data=var(data,[],1);
M_data=(data-mean(data,1));
F_data=fft(M_data);
for i=1:size(data,2)
    if strcmpi(type,'mi')
        [x,c]=mat_XMI(V_data,M_data,data(:,i));
    elseif strcmpi(type,'corr')
        [x,c]=mat_XC(V_data,M_data,data(:,i));
    elseif strcmpi(type,'circ_corr')
        [x,c]=mat_CXC(V_data,M_data,data(:,i));
    elseif strcmpi(type,'circ_mi')
        [x,c]=mat_CXMI(V_data,M_data,data(:,i));
    elseif strcmpi(type,'fft_corr')
        [x,c]=mat_FFTXC(F_data,F_data(:,i));
    else
        disp('Unrecognized type');
        return
    end
%     for g=1:size(c,2)
%     c(:,g)=medfilt1(c(:,g),20);
%     end
    [C(i,:),idx]=max(c,[],1);
    D(i,:)=-x(idx);
    if mod(i,10)==0
        figure(1)
        subplot(1,2,1);imagesc(D(1:i,:));drawnow
        subplot(1,2,2);imagesc(C(1:i,:));drawnow
    end
end

Cnew=nan(size(C));
Dnew=nan(size(D));
for i=1:size(C,1)
    for j=i:size(C,2)
        [Cnew(i,j),idx]=max([C(i,j) C(j,i)],[],2);
        Cnew(j,i)=Cnew(i,j);
        Dnew(i,j)=D(i,j)*(idx==1) - D(j,i)*(idx==2);
        Dnew(j,i)=-Dnew(i,j);
    end
end
        


C=Cnew;D=Dnew;clear Cnew Ct Dnew Ct
% Dnew=D;
% Cnew=C;
% for i=1:size(data,2)
%     for j=1:size(data,2)
%         if C(i,j)>C(j,i)
%             Dnew(j,i)=-D(i,j);
%        `       Cnew(j,i)=C(i,j);
%         else
%             Dnew(j,i)=D(j,i);
%             Dnew(i,j)=-D(j,i);
%             Cnew(i,j)=C(j,i);
%             Cnew(j,i)=C(j,i);
%         end
%     end
% end
% C=Cnew;clear Cnew
% D=Dnew;clear Dnew
C(sub2ind(size(C),1:size(C,1),1:size(C,1)))=nan;
C(isinf(C))=nan;

%% Estimate linkage probability
C_raw=C;
C=cdf('normal',nanzscore(C,[],'all'),0,1);

%% remove displacement pairs with low linkage probability
D_unthresholded=D;
D(C<threshold)=nan;
D(sub2ind(size(D),1:size(D,1),1:size(D,1)))=nan;

%% compute centralized position estimates
try
    p=psolver_TF(D,temporal_lambda);
catch
    p=zeros(size(data,2));
end


%% de-shift raster to motion correct it
data_reg=data;
for t=1:size(data,2)
    data_reg(:,t)=imtranslate(data(:,t),[0 -round(p(t))]);
end