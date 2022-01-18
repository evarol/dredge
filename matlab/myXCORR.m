function [x,c]=myXCORR(a,b)
a=a/norm(a);
b=b/norm(b);
[c,x]=xcorr(a,b);
end