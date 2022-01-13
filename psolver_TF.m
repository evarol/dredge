function p=psolver_TF(Dy,lambda,sigma)
%ICASSP '21 solver with temporal trend filtering
%Dy - input T X T displacement matrix
%lambda - temporal trend filter lambda
%sigma - robust regression standard deviation
% Solves: \min_p||D - 1p^T - p1^T|| + lambda*\sum_t||p_{t-1}+p_{t+1}-2p_t||^2
if nargin<3
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
D=diag(2*ones(1,size(Dy,1))) + diag(-ones(1,size(Dy,1)-1),1) + diag(-ones(1,size(Dy,1)-1),-1);

%% non-robust regression

if robust==0
    p=lsqr([A;lambda*D],[V;zeros(size(D,1),size(V,2))]);
    
else
    %% robust regression
    idx=(1:size(A,1))';
    for t=1:20
        p=lsqr([A(idx,:);lambda*D],[V(idx,:);zeros(size(D,1),size(V,2))]);
        idx=find(abs(zscore(A*p-V))<=sigma);
    end
end
end


