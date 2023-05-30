function [Preds, err] = MLPtrainer(trainData_org, predictData_org, l, n)
z=size(predictData_org,1);
%%
mu = mean(trainData_org);
sigma = std(trainData_org);
trainData0 = (trainData_org-mu)/sigma;
[trainData, trainTarget] = getTimeSeriesTrainData(trainData0, l);
%%
net = feedforwardnet(n, 'trainlm');
net.trainParam.epochs = 250;
net.trainParam.showWindow = false;
%%
net = train(net, trainData, trainTarget);
%%
horizon = trainTarget(end-l+1:end)';
preds = zeros(z,1);
for i=1:z
    preds(i) = sim(net, horizon);
    horizon = [horizon(2:end);preds(i)];
end
%%
Preds = preds*sigma + mu;
err = sqrt(sum((Preds-predictData_org).^2)/z);
end