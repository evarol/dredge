function mi=mat_gauss_mi(Vx,Mx,y,flag)
% x is the matrix
% y is a vector

% Vx=var(X,[],1);
% Mx=(X-mean(X,1));
numerator=Vx*var(y);

if flag==1 %N-1 normalization
    detcovXy=Vx*var(y,0) - (sum(Mx.*(y-mean(y)),1)/(size(Mx,1)-1)).^2;
else
    detcovXy=Vx*var(y,0) - (mean(Mx.*(y-mean(y)),1)).^2;
end

mi = 0.5*log(numerator./detcovXy);
end