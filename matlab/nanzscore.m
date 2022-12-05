function z=nanzscore(x,flag,dim)

z=(x-nanmean(x,dim))./nanstd(x,flag,dim);

end
