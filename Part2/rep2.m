%%%%%%%%%%%
% rep2.m
% A script which generates n random initial points 
%and visualises results of simulation of a 2d Hopfield network 'net'
%%%%%%%%%%
clear
clc
close all

%% Random initial states
T = [1 1; -1 -1; 1 -1]'; % 3 attractors in a 2 neuron network? -> that's asking for unwanted attractor states
net = newhop(T);
n=100;
figure(1)
records=cell(1,n);
for i=1:n
    a={rands(2,1)};                     % generate an initial point 
    [y,Pf,Af] = sim(net,{1 50},{},a);   % simulation of the network for 50 timesteps              
    record=[cell2mat(a) cell2mat(y)];   % formatting results  
    records{i} = record;
    start=cell2mat(a);                  % formatting results 
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r'); % plot evolution
    hold on;
    plot(record(1,50),record(2,50),'gO');  % plot the final point with a green circle
end
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 2d Hopfield model');

%% Boundary initial states
% Everything on the boundary between two quadrants will be in an unstable
% equilibrium
P = [ -1    -0.5  0 0.5 1   0   0       0   0   0;...
      0     0     0 0   0   -1  -0.5    0   0.5 1];
for i=1:size(P,2)
   a = {P(:,i)};
   [y,Pf,Af] = sim(net,{1 50},{},a);
   record=[cell2mat(a) cell2mat(y)];
   records{end+1} = record;
   start=cell2mat(a);
   plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r')
   hold on
   plot(record(1,end),record(2,end),'gO');
end
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 2d Hopfield model');
xline(0, '--', 'handleVisibility', 'off')
yline(0, '--', 'handleVisibility', 'off')

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
figure(3)
histogram(converged_states)
ylabel('Counts')
xlabel('Required number of iterations')