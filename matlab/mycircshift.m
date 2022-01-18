function out=mycircshift(x,s)
s=-s;
if s==0
    out=x;
elseif s>0
    
    out=[x(s:end);x(1:s-1)];
elseif s<0
    s=-s;
    out=[x(end-s:end);x(1:end-s-1)];
end
end