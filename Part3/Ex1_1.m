clear; clc; close all
%%
X = randn(50,500);
%%
rmse_random = zeros(size(X,1),1);
for d=1:size(X,1)
    [Z,V,D] = doPCA(X,d);
    X_hat = V*Z;
    rmse_random(d) = sqrt(mean(mean((X-X_hat).^2)));
end
figure(1)
plot(rmse_random)
xlabel('Retained dimensions')
ylabel('Reconstruction RMSE')
title('Random data')
savefig('Ex1_randomData.fig')
%%
load choles_all
rmse_corr = zeros(size(p,1),1);
for d=1:size(p,1)
    [Z,V,D] = doPCA(p,d);
    p_hat = V*Z;
    rmse_corr(d) = sqrt(mean(mean((p-p_hat).^2)));
end
figure(2)
plot(rmse_corr)
xlabel('Retained dimensions')
ylabel('Reconstruction RMSE')
title('choles\_all')
savefig('Ex1_cholesAll.fig')
%%
figure(3)
hold on
yyaxis left
plot(rmse_random)
ylabel('Reconstruction RMSE')
yyaxis right
plot(rmse_corr)
xlabel('Retained dimensions')
ylabel('Reconstruction RMSE')
legend('Random', 'Correlated')