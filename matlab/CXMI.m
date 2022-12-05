function [x,c]=CXMI(a,b)
% Va=var(A,[],1);
% Ma=(A-mean(A,1));
c=zeros(length(b),1);
for k=1:length(b)
    c(k,1)=gauss_mi(a,b,0);
    b=[b(end);b(1:end-1)]; %circular shift
end
x=[0:length(b)-1]; %lags
x=min(abs([x' -(length(b)-x')]),[],2).*(-sign(x'-length(b)/2));

idx=find(x==0);
for i=1:length(idx)
    try
        x(idx)=x(idx+1)-1;
    catch
        x(idx)=x(idx-1)+1;
    end
end