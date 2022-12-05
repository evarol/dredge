function x=fastShift(x,shift)

x = circshift(x,shift);
N = numel(x);
ix = (1:N) - shift;
tf = ix < 1 | ix > N;
x(tf) = 0 ;
end