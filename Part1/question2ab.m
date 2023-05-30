%% Setting up
clear
clc
close all

%% Colours
colours = [
0 0.4470 0.7410
0.8500 0.3250 0.0980
0.9290 0.6940 0.1250
0.4940 0.1840 0.5560
0.4660 0.6740 0.1880
0.3010 0.7450 0.9330];

%% Data creation
dx=0.05;
x=0:dx:3*pi;
t=sin(x.^2);

trials=50;
H=30;

%% Container initialisation
algs = {'traingd','traingdx','traincgf','trainbfg','trainlm','trainbr'};
A = length(algs);

mse1 = zeros(trials, A);
mse14 = zeros(trials, A);
mse85 = zeros(trials, A);

regress1 = zeros(trials, A);
regress14 = zeros(trials, A);
regress85 = zeros(trials, A);

best_epoch1 = zeros(trials, A);
best_epoch14 = zeros(trials, A);
best_epoch85 = zeros(trials, A);

%% Initialising nets
nets = cell(trials,A);
for k=1:trials
    for n=1:A
        net = feedforwardnet(H,algs{n});
        net.trainParam.showWindow = false;
        net.trainParam.epochs = 1;
        if n~=1
            net.iw{1,1}=nets{1}.iw{1,1};
            net.iw{2,1}=nets{1}.iw{2,1};
            net.b{1}=nets{1}.b{1};
            net.b{2}=nets{1}.b{2};
        end
        nets{k,n} = net;
    end
end

%% 1 epoch
nets_tr = cell(trials,A);
for k=1:trials
    ys = zeros(length(x),A);
    perfs = zeros(1, A);
    regrs = perfs;
    best_epochs = perfs;
    for n=1:A
        [net, tr] = train(nets{k,n},x,t);
        y = net(x);
        perf = tr.best_tperf;
        regr = regression(y,t);
        best_ep = tr.best_epoch;
        
        nets{k,n} = net;
        nets_tr{k,n} = tr;
        ys(:,n) = y;
        perfs(n) = perf;
        regrs(n) = regr;
        best_epochs(n) = best_ep;
    end
    mse1(k,:) = perfs;
    regress1(k,:) = regrs;
    best_epoch1(k,:) = best_epochs;
end

av_mse1 = mean(mse1,1);
std_mse1 = std(mse1,0,1);
av_regress1 = mean(regress1,1);
std_regress1 = std(regress1,0,1);
av_best_epoch1 = mean(best_epoch1,1);
std_best_epoch1 = std(best_epoch1,0,1);

%% 15 epochs
nets_tr = cell(trials,A);
for k=1:trials
    ys = zeros(length(x),A);
    perfs = zeros(1, A);
    regrs = perfs;
    best_epochs = perfs;
    for n=1:A
        net = nets{k,n};
        net.trainParam.epochs = 14;
        [net, tr] = train(net,x,t);
        y = net(x);
        perf = tr.best_tperf;
        regr = regression(y,t);
        best_ep = tr.best_epoch;
        
        nets{k,n} = net;
        nets_tr{k,n} = tr;
        ys(:,n) = y;
        perfs(n) = perf;
        regrs(n) = regr;
        best_epochs(n) = best_ep;
    end
    mse14(k,:) = perfs;
    regress14(k,:) = regrs;
    best_epoch14(k,:) = best_epochs;
end

av_mse14 = mean(mse14,1);
std_mse14 = std(mse14,0,1);
av_regress14 = mean(regress14,1);
std_regress14 = std(regress14,0,1);
av_best_epoch14 = mean(best_epoch14,1);
std_best_epoch14 = std(best_epoch14,0,1);

%% 100 epochs
nets_tr = cell(trials,A);
for k=1:trials
    ys = zeros(length(x),A);
    perfs = zeros(1, A);
    regrs = perfs;
    best_epochs = perfs;
    for n=1:A
        net = nets{k,n};
        net.trainParam.epochs = 85;
        [net, tr] = train(net,x,t);
        y = net(x);
        perf = tr.best_tperf;
        regr = regression(y,t);
        best_ep = tr.best_epoch;
        
        nets{k,n} = net;
        nets_tr{k,n} = tr;
        ys(:,n) = y;
        perfs(n) = perf;
        regrs(n) = regr;
        best_epochs(n) = best_ep;
    end
    mse85(k,:) = perfs;
    regress85(k,:) = regrs;
    best_epoch85(k,:) = best_epochs;
end

av_mse85 = mean(mse85,1);
std_mse85 = std(mse85,0,1);
av_regress85 = mean(regress85,1);
std_regress85 = std(regress85,0,1);
av_best_epoch85 = mean(best_epoch85,1);
std_best_epoch85 = std(best_epoch85,0,1);

%% Plotting fits for 100 epochs
figure(5)
hold on
plot(x,t,'kx');
for i=1:size(ys,2)
    plot(x,ys(:,i))
end
title('Fitted values (100 epochs)');
legend([{'target'},algs],'Location','northwest');

%% Creating noisy data
dx=0.05;
x=0:dx:3*pi;
t=sin(x.^2)+0.2*randn(size(x));

%% Initialising containers for noisy data
mse100_noise = zeros(trials, A);
regress100_noise = zeros(trials, A);
best_epoch100_noise = zeros(trials, A);

%% Initialising nets for noisy data
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

%% 100 epochs for noisy data
nets_tr = cell(trials,A);
for k=1:trials
    ys = zeros(length(x),A);
    perfs = zeros(1, A);
    regrs = perfs;
    best_epochs = perfs;
    for n=1:A
        [net, tr] = train(nets{k,n},x,t);
        y = net(x);
        perf = tr.best_tperf;
        regr = regression(y,t);
        best_ep = tr.best_epoch;
        
        nets{k,n} = net;
        nets_tr{k,n} = tr;
        ys(:,n) = y;
        perfs(n) = perf;
        regrs(n) = regr;
        best_epochs(n) = best_ep;
    end
    mse100_noise(k,:) = perfs;
    regress100_noise(k,:) = regrs;
    best_epoch100_noise(k,:) = best_epochs;
end

av_mse100_noise = mean(mse100_noise,1);
std_mse100_noise = std(mse100_noise,0,1);
av_regress100_noise = mean(regress100_noise,1);
std_regress100_noise = std(regress100_noise,0,1);
av_best_epoch100_noise = mean(best_epoch100_noise,1);
std_best_epoch100_noise = std(best_epoch100_noise,0,1);

%% Plotting MSE by epoch
figure(1)

subplot(1,3,1)
hold on
b = bar(av_mse1);
b.FaceColor = 'flat';
b.CData = colours;
er = errorbar(av_mse1,std_mse1);
er.Color = [0 0 0];
er.LineStyle = 'None';
xticklabels(algs)
xtickangle(45)
ylim([0,8])
title('MSE 1 epoch')
hold off
xticks(1:size(av_mse1,2))

subplot(1,3,2)
hold on
b = bar(av_mse14);
b.FaceColor = 'flat';
b.CData = colours;
er = errorbar(av_mse14,std_mse14);
er.Color = [0 0 0];
er.LineStyle = 'None';
xticklabels(algs)
xtickangle(45)
ylim([0,8])
title('MSE 15 epochs')
hold off
xticks(1:size(av_mse14,2))

subplot(1,3,3)
hold on
b = bar(av_mse85);
b.FaceColor = 'flat';
b.CData = colours;
er = errorbar(av_mse85,std_mse85);
er.Color = [0 0 0];
er.LineStyle = 'None';
xticklabels(algs)
xtickangle(45)
ylim([0,8])
title('MSE 100 epochs')
hold off
xticks(1:size(av_mse85,2))

%% Plotting regression by epoch
figure(2)

subplot(1,3,1)
hold on
b = bar(abs(av_regress1));
b.FaceColor = 'flat';
b.CData = colours;
er = errorbar(abs(av_regress1),std_regress1);
er.Color = [0 0 0];
er.LineStyle = 'None';
xticklabels(algs)
xtickangle(45)
ylim([0,1.01])
title('Regression 1 epoch')
hold off
xticks(1:size(av_regress1,2))

subplot(1,3,2)
hold on
b = bar(abs(av_regress14));
b.FaceColor = 'flat';
b.CData = colours;
er = errorbar(abs(av_regress14),std_regress14);
er.Color = [0 0 0];
er.LineStyle = 'None';
xticklabels(algs)
xtickangle(45)
ylim([0,1.01])
title('Regression 15 epochs')
hold off
xticks(1:size(av_regress14,2))

subplot(1,3,3)
hold on
b = bar(abs(av_regress85));
b.FaceColor = 'flat';
b.CData = colours;
er = errorbar(abs(av_regress85),std_regress85);
er.Color = [0 0 0];
er.LineStyle = 'None';
xticklabels(algs)
xtickangle(45)
ylim([0,1.01])
title('Regression 100 epochs')
hold off
xticks(1:size(av_regress85,2))

%% Plotting convergence by epoch
figure(3)
hold on
b = bar(av_best_epoch85);
b.FaceColor = 'flat';
b.CData = colours;
er = errorbar(av_best_epoch85,std_best_epoch85);
er.Color = [0 0 0];
er.LineStyle = 'None';
xticklabels(algs)
xtickangle(45)
ylim([0,100])
title('Average number of epochs')
hold off
xticks(1:size(av_best_epoch85,2))

%% Plotting metrics for noisy data
figure(4)

subplot(1,3,1)
hold on
bdata = [av_mse85',av_mse100_noise'];
berr = [std_mse85',std_mse100_noise'];
b = bar(bdata);
xticklabels(algs)
xtickangle(45)
title('MSE')
legend('noiseless', 'noisy', 'Location', 'southwest')
for k=1:size(bdata,2)
    xpos = b(k).XData + b(k).XOffset;
    er = errorbar(xpos,bdata(:,k),berr(:,k), 'handleVisibility', 'off');
    er.Color = [0 0 0];
    er.LineStyle = 'None';
end
hold off
xticks(1:size(bdata,1))

subplot(1,3,2)
hold on
bdata = [abs(av_regress85'),abs(av_regress100_noise')];
berr = [std_regress85',std_regress100_noise'];
b = bar(bdata);
xticklabels(algs)
xtickangle(45)
title('Regression')
legend('noiseless', 'noisy', 'Location', 'southwest')
for k=1:size(bdata,2)
    xpos = b(k).XData + b(k).XOffset;
    er = errorbar(xpos,bdata(:,k),berr(:,k), 'handleVisibility', 'off');
    er.Color = [0 0 0];
    er.LineStyle = 'None';
end
hold off
xticks(1:size(bdata,1))

subplot(1,3,3)
hold on
bdata = [av_best_epoch85'+15,av_best_epoch100_noise'];
berr = [std_best_epoch85',std_best_epoch100_noise'];
b = bar(bdata);
xticklabels(algs)
xtickangle(45)
title('Required epochs')
legend('noiseless', 'noisy', 'Location', 'southwest')
for k=1:size(bdata,2)
    xpos = b(k).XData + b(k).XOffset;
    er = errorbar(xpos,bdata(:,k),berr(:,k), 'handleVisibility', 'off');
    er.Color = [0 0 0];
    er.LineStyle = 'None';
end
hold off
xticks(1:size(bdata,1))

%% Plotting fits for 100 epochs
figure(6)
hold on
plot(x,t,'kx');
for i=1:size(ys,2)
    plot(x,ys(:,i))
end
title('Fitted values (100 epochs)');
legend([{'target'},algs],'Location','northwest');
