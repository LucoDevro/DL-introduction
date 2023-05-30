clear; clc; close all
%%
load threes.mat -ascii
colormap('gray')
%%
mean3 = mean(threes,1);
imagesc(reshape(mean3,16,16),[0,1]);
savefig('mean3.fig')
%%
covM = cov(threes);
[V,D] = eigs(covM, size(covM,2));
figure
plot(diag(D))
%%
threes_comp=cell(4,1);
threes_hat_c=cell(4,1);
for c=1:4
    [z,v,d] = doPCA(threes',c);
    threes_comp{c}=z;
    threes_hat_c{c} = (v*z)';
end
%%
ii=randi([1, size(threes,1)],1,4);
for k=1:length(ii)
    i=ii(k);
    figure
    for j=1:4
        subplot(2,2,j)
        imagesc(reshape(threes_hat_c{j}(i,:),16,16),[0,1]);
        title(num2str(j)+" PCs")
    end
    savefig("threes_PC_comparison_repl_"+num2str(k)+".fig")
end
%%
max_PCA=256;
rmse_c = zeros(max_PCA,1);
for q=1:max_PCA
    rmse_c(q) = PCAcomprAndReconErr(threes,q);
end
figure
plot(rmse_c)
xlabel('PCs')
ylabel('Reconstruction RMSE')
savefig("RMSEvsnrPCA.fig")
%%
evl = diag(D);
evl_sum = cumsum(evl);
figure
hold on
yyaxis left
ylabel("Reconstruction RMSE")
plot(rmse_c)
yyaxis right
plot(evl_sum)
ylabel("Sum of N-i largest eigenvalues")
xlabel('PCs')
savefig("RMSEvsEvlSum.fig")