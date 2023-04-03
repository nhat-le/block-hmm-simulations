%% Plotting the performance
addpath('/Users/minhnhatle/Dropbox (MIT)/Jazayeri/NoisyMutualInhibition/PlotTools')

load("/Users/minhnhatle/Documents/Sur/MatchingSimulations/processed_data/expdata/122221b_figshare/simdata/EGreedyqlearningAgent-withCorr-doublesigmoid-prob0.00to1.00.mat");
assert(nblocks == 1000);

produce_heatmap(efflist, prewlst, pswitchlst, 'clim', [0.5,1], 'legendname', 'Efficiency', ...
    'x_label', '$P_{rew}$', 'y_label', '$P_{switch}$', 'ytickvalues', 0:0.1:0.4);
plot([0.55, 0.99], [0.01, 0.45], 'k--', 'LineWidth', 2)
plot([0.55, 0.7456, 0.99], [0.01, 0.1986, 0.45], 'kx', 'MarkerSize', 10, 'LineWidth', 2);


%% Plotting the switch offset
produce_heatmap(-PLoffsetlist, prewlst, pswitchlst, 'clim', [0, 10], ...
'legendname', 'Offset', ...
'x_label', '$P_{rew}$', 'y_label', '$P_{switch}$', 'ytickvalues', 0:0.1:0.4);
plot([0.55, 0.99], [0.01, 0.45], 'k--', 'LineWidth', 2)
plot([0.55, 0.7456, 0.99], [0.01, 0.1986, 0.45], 'kx', 'MarkerSize', 10, 'LineWidth', 2);



%% Plotting the switch slope
produce_heatmap(PLslopelist, prewlst, pswitchlst, 'clim', [0, 3], 'legendname', 'Slope', ...
'x_label', '$P_{rew}$', 'y_label', '$P_{switch}$', 'ytickvalues', 0:0.1:0.4);
plot([0.55, 0.99], [0.01, 0.45], 'k--', 'LineWidth', 2)
plot([0.55, 0.7456, 0.99], [0.01, 0.1986, 0.45], 'kx', 'MarkerSize', 10, 'LineWidth', 2);


%% Plotting the lapse rate (exploration)
produce_heatmap(LapseR, prewlst, pswitchlst, 'clim', [0, 0.5], 'legendname', 'Lapse', ...
'x_label', '$P_{rew}$', 'y_label', '$P_{switch}$', 'ytickvalues', 0:0.1:0.4);
plot([0.55, 0.99], [0.01, 0.45], 'k--', 'LineWidth', 2)
plot([0.55, 0.7456, 0.99], [0.01, 0.1986, 0.45], 'kx', 'MarkerSize', 10, 'LineWidth', 2);

    

