%% Overparametrised comparison of trainbr and trainlm with noisy data

%%
clear
clc
close all

%% Colours
colours = [
0 0.4470 0.7410
0.8500 0.3250 0.0980
0.9290 0.6940 0.1250
0.4940 0.1840 0.5560];

%% Data creation
dx=0.05;
x=0:dx:3*pi;
t=sin(x.^2);
tn=sin(x.^2)+0.2*randn(size(x));

trials=50;
H=300;

%% Container initialisation
algs = {'trainlm', 'trainbr','trainlm', 'trainbr'};
A = length(algs);

mses = zeros(trials, A);
regress = mses;
epochs = mses;

%% Initialising nets
nets = cell(trials,A);
for k=1:trials
    for n=1:A
        net = feedforwardnet(H,algs{n});
        net.trainParam.showWindow = false;
        net.trainParam.epochs = 100;
        if n~=1
            net.iw{1,1}=nets{1}.iw{1,1};
            net.iw{2,1}=nets{1}.iw{2,1};
            net.b{1}=nets{1}.b{1};
            net.b{2}=nets{1}.b{2};
        end
        nets{k,n} = net;
    end
end

%% Train
nets_tr = cell(trials,A);
for k=1:trials
    ys = zeros(length(x),A);
    perfs = zeros(1, A);
    regrs = perfs;
    best_epochs = perfs;
    for n=1:A
        if n<3
            tx = t;
        else
            tx = tn;
        end
        [net, tr] = train(nets{k,n},x,tx);
        y = net(x);
        perf = tr.best_tperf;
        regr = regression(y,tx);
        best_ep = tr.best_epoch;
        
        nets{k,n} = net;
        nets_tr{k,n} = tr;
        ys(:,n) = y;
        perfs(n) = perf;
        regrs(n) = regr;
        best_epochs(n) = best_ep;
    end
    mses(k,:) = perfs;
    regress(k,:) = regrs;
    epochs(k,:) = best_epochs;
end

av_mse = mean(mses,1);
std_mse = std(mses,0,1);
av_regress = mean(regress,1);
std_regress = std(regress,0,1);
av_best_epoch = mean(epochs,1);
std_best_epoch = std(epochs,0,1);

%% Plotting metrics
figure

subplot(1,3,1)
hold on
b = bar(av_mse');
b.FaceColor = 'flat';
b.CData = colours;
er = errorbar(av_mse,std_mse);
er.Color = [0 0 0];
er.LineStyle = 'None';
xticklabels({'trainlm','trainbr','Ntrainlm','Ntrainbr'})
xtickangle(45)
title('test MSE')
hold off
xticks(1:size(av_mse,2))

subplot(1,3,2)
hold on
b = bar(abs(av_regress'));
b.FaceColor = 'flat';
b.CData = colours;
er = errorbar(av_regress,std_regress);
er.Color = [0 0 0];
er.LineStyle = 'None';
xticklabels({'trainlm','trainbr','Ntrainlm','Ntrainbr'})
xtickangle(45)
ylim([0 1])
title('Regression')
hold off
xticks(1:size(av_regress,2))

subplot(1,3,3)
hold on
b = bar(av_best_epoch');
b.FaceColor = 'flat';
b.CData = colours;
er = errorbar(av_best_epoch,std_best_epoch);
er.Color = [0 0 0];
er.LineStyle = 'None';
xticklabels({'trainlm','trainbr','Ntrainlm','Ntrainbr'})
xtickangle(45)
title('Required epochs')
hold off
xticks(1:size(av_best_epoch,2))
