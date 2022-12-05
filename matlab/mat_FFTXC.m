function [x,c]=mat_FFTXC(FA,Fb)
% Va=var(A,[],1);
% Ma=(A-mean(A,1));
c=zeros(length(Fb),size(FA,2));
x=-length(Fb):length(Fb);%lags
for i=1:size(FA,2)
    [output,~,Cmax] = dftregistration(FA(:,i),Fb,1);
    c(x==output(3),i)=Cmax;
end