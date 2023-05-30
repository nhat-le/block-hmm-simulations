%% Plotting the performance
addpath('/Users/minhnhatle/Dropbox (MIT)/Jazayeri/NoisyMutualInhibition/PlotTools')

paths = pathsetup('matchingsim');
version = '121021';

load('simdata/EGreedyqlearningAgent-withCorr-doublesigmoid-prob0.00to1.00-092321.mat');
% load("/Users/minhnhatle/Documents/Sur/MatchingSimulations/processed_data/expdata/122221b_figshare/simdata/EGreedyinf-basedAgent-withCorr-doublesigmoid-30by30-prob0.00to1.00.mat");
assert(nblocks == 1000);

produce_heatmap(efflist, epslst, gammalst, 'clim', [0.5,1], 'legendname', 'Efficiency', ...
'x_label', '$\epsilon$', 'y_label', '$\gamma$', 'vertline', 0.1, 'horline', 1.2);

%% Plotting the switch offset
produce_heatmap(-PLoffsetlist, epslst, gammalst, 'clim', [0 10], 'legendname', 'Offset', ...
'x_label', '$\epsilon$','y_label', '$\gamma$', 'vertline', 0.24, 'horline', 1.22);


%% Plotting the switch slope
produce_heatmap(PLslopelist, epslst, gammalst, 'clim', [0, 3], 'legendname', 'Slope', ...
'x_label', '$\epsilon$', 'y_label', '$\gamma$', 'vertline', 0.24, 'horline', 1.22);

%% Plotting the lapse rate (exploration)
produce_heatmap(LapseR, epslst, gammalst, 'clim', [0, 0.5], 'legendname', 'Lapse', ...
'x_label', '$\epsilon$', 'y_label', '$\gamma$', 'vertline', 0.24, 'horline', 1.22);

   