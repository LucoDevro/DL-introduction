clear; clc; close all
for noise=2:2:10
    disp(noise)
    for iter=logspace(1,5,5)
        disp(iter)
        hopdigit(noise, iter);
    end
end