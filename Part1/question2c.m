%% Initialising environment
clear
clc
close all

%% Loading and creating data
load('Data_Problem1_regression.mat')
Tnew = (6*T1+6*T2+5*T3+3*T4+2*T5)/(6+6+5+3+2);
data = [X1,X2,Tnew];

%% Splitting the dataset
all_ind = randsample(length(X1),3000);
train_ind = all_ind(1:1000);
vali_ind = all_ind(1001:2000);
test_ind = all_ind(2001:3000);
train_data = data(train_ind,:);
vali_data = data(vali_ind,:);
test_data = data(test_ind,:);

%% Plotting the training set
x1 = linspace(min(train_data(:,1)), max(train_data(:,1)));
x2 = linspace(min(train_data(:,2)), max(train_data(:,2)));
[x1m, x2m] = meshgrid(x1,x2);
interpol = scatteredInterpolant(train_data(:,1), train_data(:,2), train_data(:,3));
zm = interpol(x1m,x2m);

figure(1)
mesh(x1m,x2m,zm);
hold on
plot3(train_data(:,1), train_data(:,2), train_data(:,3),'.')

%% Just a test with a double layer of 25 neurons
% neurons = [25 25];
% net0 = feedforwardnet(neurons, 'trainlm');
% net0.trainParam.epochs = 100;
% net0.divideFcn = 'divideind';
% net0.divideParam.trainInd = train_ind;
% net0.divideParam.valInd = vali_ind;
% net0.divideParam.testInd = test_ind;
% 
% X = [X1';X2'];
% [net0, net0_tr] = train(net0,X,Tnew');
% 
% figure(2)
% plotperform(net0_tr)

%% Tuning the number of layers
H = 60; % 60 because that has many divisors
layers = 5;
trials = 10;
mses_layers = zeros(trials,layers);

X = [X1';X2'];

for k=1:trials
    for l=1:layers
        neurons = H/l;
        net = feedforwardnet(repmat(neurons,1,l), 'trainlm');
        net.trainParam.epochs = 200;
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = train_ind;
        net.divideParam.valInd = vali_ind;
        net.divideParam.testInd = test_ind;

        [net, tr] = train(net,X,Tnew');
        mses_layers(k,l) = tr.best_tperf;
    end
end
average_mses_layers = mean(mses_layers,1);
[~,min_mses_layers] = min(mses_layers,[],2);
% four layers

%% Tuning the number of neurons in a layer
L = 4;
trials = 10;
neurons = 6:3:30;
mses_neurons = zeros(trials,length(neurons));

for k=1:trials
    for n=1:length(neurons)
        net = feedforwardnet(repmat(neurons(n),1,L), 'trainlm');
        net.trainParam.epochs = 200;
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = train_ind;
        net.divideParam.valInd = vali_ind;
        net.divideParam.testInd = test_ind;
        
        [net, tr] = train(net,X,Tnew');
        mses_neurons(k,n) = tr.best_tperf;
    end
end
average_mses_neurons = mean(mses_neurons,1);
[~,min_mses_neurons] = min(mses_neurons,[],2);

%% Training final net: 4 layers of 12 neurons
L = 4;
N = 12;
neurons = repmat(N,1,L);
net = feedforwardnet(neurons, 'trainlm');
net.trainParam.epochs = 200;
net.divideFcn = 'divideind';
net.divideParam.trainInd = train_ind;
net.divideParam.valInd = vali_ind;
net.divideParam.testInd = test_ind;

[net,tr] = train(net,X,Tnew');
figure(3)
plotperform(tr)

%% Plotting surface of the test set
Y = net(test_data(:,1:2)');
x1t = linspace(min(test_data(:,1)), max(test_data(:,1)));
x2t = linspace(min(test_data(:,2)), max(test_data(:,2)));
[x1tm, x2tm] = meshgrid(x1t,x2t);
interpol_data = scatteredInterpolant(test_data(:,1), test_data(:,2), test_data(:,3));
zm_test = interpol_data(x1tm,x2tm);
interpol_net = scatteredInterpolant(test_data(:,1), test_data(:,2), Y');
zm_net = interpol_net(x1tm,x2tm);

figure(4)
mesh(x1tm,x2tm,zm_test);
hold on
plot3(test_data(:,1), test_data(:,2), test_data(:,3),'.')

figure(5)
mesh(x1tm,x2tm,zm_net);

%% Plotting difference surface
interpol_diff = scatteredInterpolant(test_data(:,1), test_data(:,2), test_data(:,3)-Y');
zm_diff = interpol_diff(x1tm,x2tm);

figure(6)
mesh(x1tm,x2tm,zm_diff);