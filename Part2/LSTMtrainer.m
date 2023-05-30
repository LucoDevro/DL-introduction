function [Preds, err] = LSTMtrainer(trainData0, predictData0, l, n)
%%
mu = mean(trainData0);
sigma = std(trainData0);
trainData = (trainData0-mu)/sigma;
%%
z = length(predictData0);

[trainData, trainTarget] = getTimeSeriesTrainData(trainData, l);

numFeatures = l;
numResponses = 1;
numHiddenUnits = n;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'Verbose',0, ...
    'Plots','none', ...
    'ExecutionEnvironment','cpu');
%%
net = trainNetwork(trainData,trainTarget,layers, options);

%%
net = predictAndUpdateState(net,trainData);

horizon = trainTarget(end-l+1:end)';
preds = zeros(z,1);
for i=1:z
    [net, preds(i)] = predictAndUpdateState(net, horizon);
    horizon = [horizon(2:end); preds(i)];
end
%%
Preds = preds*sigma+mu;
err = sqrt(sum((Preds-predictData0).^2)/z);
end