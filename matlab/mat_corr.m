function c=mat_corr(Vx,Mx,y,flag)
% x is the matrix
% y is a vector

% Vx=var(X,[],1);
% Mx=(X-mean(X,1));
numerator=mean(Mx.*(y-mean(y)),1);
denominator=sqrt(Vx.*var(y,flag));

c = numerator./denominator;
end