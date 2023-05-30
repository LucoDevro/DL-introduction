clear
close all
%nntraintool('close');
nnet.guis.closeAllViews();

% Neural networks have weights randomly initialized before training.
% Therefore the results from training are different each time. To avoid
% this behavior, explicitly set the random number generator seed.
rng('default')

% Load the training data into memory
load('digittrain_dataset.mat');

% other parameters
trials = 10;

%% Initialising datasets
% test
imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;
load('digittest_dataset.mat');
xTest = zeros(inputSize,numel(xTestImages));
for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end

% training for finetuning
xTrain = zeros(inputSize,numel(xTrainImages));
for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
end

%% Defining layers
% first layers
layer1_100N_400E = @(input)...
(trainAutoencoder(input,100, ...
    'ShowProgressWindow', false,...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false));
layer1_100N_200E = @(input)...
(trainAutoencoder(input,100, ...
    'ShowProgressWindow', false,...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false));
layer1_200N_200E = @(input)...
(trainAutoencoder(input,200, ...
    'ShowProgressWindow', false,...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false));
layer1_50N_200E = @(input)...
(trainAutoencoder(input,50, ...
    'ShowProgressWindow', false,...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false));

% Second layers
layer2_50N_100E = @(input)...
(trainAutoencoder(input,50, ...
    'ShowProgressWindow', false,...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false));
layer2_50N_200E = @(input)...
(trainAutoencoder(input,50, ...
    'ShowProgressWindow', false,...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false));
layer2_100N_200E = @(input)...
(trainAutoencoder(input,100, ...
    'ShowProgressWindow', false,...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false));
layer2_25N_200E = @(input)...
(trainAutoencoder(input,25, ...
    'ShowProgressWindow', false,...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false));

%% Deep nets configurations
conf1 = {layer1_100N_400E, layer2_50N_100E};
conf2 = {layer1_100N_200E, layer2_50N_100E};
conf3 = {layer1_100N_200E, layer2_50N_200E};
conf4 = {layer1_200N_200E, layer2_100N_200E};
conf5 = {layer1_200N_200E, layer2_50N_200E};
conf6 = {layer1_100N_200E, layer2_25N_200E};
conf7 = {layer1_200N_200E, layer2_100N_200E, layer2_50N_200E};
conf8 = {layer1_100N_200E};
conf9 = {layer1_200N_200E};
conf10 = {layer1_50N_200E};
conf11 = {layer1_100N_400E};

conf = {conf1, conf2, conf3, conf4, conf5, conf6, conf7, conf8, conf9, conf10, conf11};

%% Build and train deep nets
nets = cell(length(conf),trials);
for c=1:length(conf)
    for t=1:trials
        nets{c,t} = buildDeepNet(conf{c},xTrainImages,tTrain);
    end
end

%% Test deep nets
classAcc_c = zeros(length(conf),trials);
for c=1:size(nets,1)
    for t=1:size(nets,2)
        y = nets{c,t}(xTest);
        classAcc_c(c,t) = 100*(1-confusion(tTest,y));
    end
end
mean_classAcc = mean(classAcc_c,2);

%% Test fine-tuned deep nets
classAcc_finetuned_c = zeros(length(conf),trials);
for c=1:size(nets,1)
    for t=1:size(nets,2)
        net = nets{c,t};
        net.trainParam.showWindow = false;
        finetuned_net = train(net,xTrain,tTrain);
        y = finetuned_net(xTest);
        classAcc_finetuned_c(c,t) = 100*(1-confusion(tTest,y));
    end
end
mean_classAcc_finetuned = mean(classAcc_finetuned_c,2);

%% Compare with normal neural network (1 hidden layers)
classAcc_1L = zeros(1,trials);
for t=1:trials
    net = patternnet(100);
    net.trainParam.showWindow = false;
    net.trainParam.epochs = 200;
    net.trainParam.min_grad = 10^(-9);
    net = train(net,xTrain,tTrain);
    y = net(xTest);
    classAcc_1L(t) = 100*(1-confusion(tTest,y));
end
mean_classAcc_1L = mean(classAcc_1L);

%% Compare with normal neural network (2 hidden layers)
classAcc_2L = zeros(1,trials);
for t=1:trials
    net = patternnet([100 50]);
    net.trainParam.showWindow = false;
    net.trainParam.epochs = 200;
    net.trainParam.min_grad = 10^(-9);
    net = train(net,xTrain,tTrain);
    y = net(xTest);
    classAcc_2L(t) = 100*(1-confusion(tTest,y));
end
mean_classAcc_2L = mean(classAcc_2L);

%% deepnet builder helper function
function deepnet = buildDeepNet(conf,input,tTrain)
feat = input;
layers = cell(1,length(conf));
for l=1:length(conf)
    layers{l} = feval(conf{l},feat);
    feat = encode(layers{l},feat);
end
softnet = trainSoftmaxLayer(feat,tTrain,'MaxEpochs',400, 'ShowProgressWindow',false);
deepnet = layers{1};
for n=2:length(layers)
    deepnet = stack(deepnet,layers{n});
end
deepnet = stack(deepnet,softnet);
end
