function[err] = PCAcomprAndReconErr(X,q)
[z,V,~] = doPCA(X',q);
X_hat = (V*z)';
err = sqrt(mean(mean((X-X_hat).^2)));
end