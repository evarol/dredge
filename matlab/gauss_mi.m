function mi=gauss_mi(x,y,flag)

if flag==1
    mi=0.5*log((var(x)*var(y))/det(cov([x y])));
else
    detCov=var(x)*var(y)-mean((x-mean(x)).*(y-mean(y))).^2;
    mi=0.5*log((var(x)*var(y))/detCov);
end

end