function [x,c]=XMI(a,b)
c=zeros(length(b),1);
x=-length(b):length(b);%lags
for k=1:length(x)
    c(k,1)=gauss_mi(a,fastShift(b,x(k)),0);
end

end