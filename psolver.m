function p=psolver(Dy,sigma)
if nargin<2
    robust=0;
else
    robust=1;
end


[I,J,~]=find(~isnan(Dy));
S=~isnan(Dy);
V=Dy(S==1);
M=sparse((1:size(I,1))',I,ones(size(I)));
N=sparse((1:size(I,1))',J,ones(size(I)));
A=M-N;


%% non-robust regression

if robust==0
    p=lsqr(A,V);obj=[];
    
else
    %% robust regression
    idx=(1:size(A,1))';
    pold=nan(size(Dy,2),1);
    for t=1:20
        p=lsqr(A(idx,:),V(idx,:));
        pold=p;
        idx=find(abs(zscore(A*p-V))<=sigma);
    end
end
end