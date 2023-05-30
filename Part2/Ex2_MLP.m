clear; clc; close all

trainData_org = load("lasertrain.dat");
predictData_org = load("laserpred.dat");

%%
ll = 5:5:50;
nn = 20:5:80;
pars = combvec(ll, nn);

trials = 10;

Err = zeros(trials,size(pars,2));
Preds_c=cell(trials,size(pars,2));

for t=1:trials
    disp(t)
    for i=1:size(pars, 2)
        l = pars(1,i);
        n = pars(2,i);
        disp([l,n])
        [Preds, err] = MLPtrainer(trainData_org, predictData_org, l, n);
        Preds_c{t,i} = Preds;
        Err(t,i) = err;
    end
end

%%
MeanErr = mean(Err,1);

%%
tbl = table(pars(1,:)', pars(2,:)', MeanErr', 'VariableNames', {'Lag size','Number of neurons','RMSE'});
h = heatmap(tbl, 'Lag size', 'Number of neurons', 'ColorVariable', 'RMSE');
h.title("Mean prediction RMSE")

%%
[meanErr_min, idx] = min(tbl{:,3});
l_min = tbl{idx,1};
n_min = tbl{idx,2};
[err_min, t_min] = min(Err(:,idx));
Preds_min = Preds_c{t_min,idx};

%%
figure
hold on
plot(predictData_org, 'b-')
plot(Preds_min, 'r')
xlabel('Time point [-]')
ylabel('Output [-]')
legend('Measured', 'Predicted', 'Location', 'northwest')