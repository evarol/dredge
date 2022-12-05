function [x,c]=mat_CXC(Va,Ma,b)


% Va=var(A,[],1);
% Ma=(A-mean(A,1));
c=zeros(length(b),size(Va,2));
x=-length(b):length(b);%lags
for k=1:length(x)
    c(k,:)=mat_corr(Va,Ma,b,1);
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
end