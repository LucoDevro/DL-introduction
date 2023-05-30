function[z,V,D] = doPCA(X,q)
covM = cov(X');
if nargin < 2
    q = size(covM,2);
end
[V,D] = eigs(covM,q);
D = diag(D);
z = V'*X;
end