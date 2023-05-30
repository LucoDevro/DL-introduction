%%%%%%%%%%%
% rep3.m
% A script which generates n random initial points for
% and visualise results of simulation of a 3d Hopfield network net
%%%%%%%%%%
%%
clear; clc; close all;

%% Random initial states
figure(1)
T = [1 1 1; -1 -1 1; 1 -1 -1]'; % three attractors for a three-neuron network. This will fit without any unwanted equilibrium.
net = newhop(T);
n=100;
s=500;
records = cell(1,n);
P = {[-1 -1 -1]', [1 1 -1]', [-1 1 1]'};
for i=1:n+3
    a={rands(3,1)};                         % generate an initial point                   
    if i>n
        a=P(i-n);
    end
    [y,Pf,Af] = sim(net,{1 s},{},a);        % simulation of the network  for 50 timesteps
    record=[cell2mat(a) cell2mat(y)];       % formatting results
    records{i} = record;
    start=cell2mat(a);                      % formatting results 
    plot3(start(1,1),start(2,1),start(3,1),'bx',record(1,:),record(2,:),record(3,:),'r');  % plot evolution
    hold on
    plot3(record(1,s),record(2,s),record(3,s),'gO');  % plot the final point with a green circle
end
grid on;
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 3d Hopfield model');

% lang blijven zitten op een punt net buiten het evenwichtspunt, tot
% plotseling naar een attractor te schieten (normaal of numerieke
% onstabiliteit?)

%% Collecting endstates
endstates = unique(cell2mat(cellfun(@(x) x(:,end), records, 'UniformOutput', false))','row')';

%% Required number of iterations
converged_states = zeros(1,length(records));
for r=1:length(records)
    found = false;
    for s=1:size(records{r},2)
        for e=1:size(endstates,2)
            if ismember(records{r}(:,s), endstates(:,e), 'rows')
                converged_states(r) = s;
                found = true;
                break
            end
        end
        if found
            break
        end
    end
end

%% Plotting convergence
figure(2)
histogram(converged_states)
ylabel('Counts')
xlabel('Required number of iterations')