function [x,c]=CXCORR(a,b)
na=norm(a);
nb=norm(b);
a=a/na; %normalization
b=b/nb;
for k=1:length(b)
    c(k)=a*b';
    b=[b(end),b(1:end-1)]; %circular shift
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