function [x,c]=mat_XC(Va,Ma,b)
% Va=var(A,[],1);
% Ma=(A-mean(A,1));
c=zeros(length(b),size(Va,2));
x=-length(b):length(b);%lags
for k=1:length(x)
    c(k,:)=mat_corr(Va,Ma,fastShift(b,x(k)),1);
end

end