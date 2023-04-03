function [out,opts] = load_data(opts)
% Load features of behavioral simulation
%%
% load('/Users/minhnhatle/Documents/Sur/MatchingSimulations/processed_data/expdata/122221b_figshare/simdata/EGreedyqlearningAgent-withCorr-doublesigmoid-prob0.00to1.00.mat', 'efflist', 'LapseL', 'PLslopelist', 'PLoffsetlist', 'epslst', 'gammalst');
load('simdata/EGreedyQLearningAgent-withCorr-doublesigmoid-prob0.00to1.00-092321.mat', 'efflist', 'LapseL', 'PLslopelist', 'PLoffsetlist', 'epslst', 'gammalst');

Qeff_flat = reshape(efflist, [], 1);
Qlapse_flat = reshape(LapseL, [], 1);
Qslope_flat = reshape(PLslopelist, [], 1);
Qoffset_flat = reshape(PLoffsetlist, [], 1);

out.Qdim = size(efflist);

load('simdata/EGreedyinf-basedAgent-withCorr-doublesigmoid-prob0.00to1.00-092321.mat', 'efflist', 'LapseL', 'PLslopelist', 'PLoffsetlist', 'prewlst', 'pswitchlst');

IBeff_flat = reshape(efflist, [], 1);
IBlapse_flat = reshape(LapseL, [], 1);
IBslope_flat = reshape(PLslopelist, [], 1);
IBoffset_flat = reshape(PLoffsetlist, [], 1);

% Filter outliers
if isfield(opts, 'clipmode') && opts.clipmode == 1
    IBoffset_flat(IBoffset_flat < -20) = 3; %-20;
    Qoffset_flat(Qoffset_flat < -20) = 3; %-20;
elseif isfield(opts, 'clipmode') && opts.clipmode == 2
    IBoffset_flat(IBoffset_flat < -20) = -20; %-20;
    Qoffset_flat(Qoffset_flat < -20) = -20; %-20;
end


% Parse the outputs
out.IBdim = size(efflist);

% TODO: change this order to be consistent with the decoding
out.features = [IBeff_flat IBlapse_flat IBslope_flat IBoffset_flat;
    Qeff_flat Qlapse_flat Qslope_flat Qoffset_flat];

out.features_norm = (out.features - nanmean(out.features, 1)) ./ nanstd(out.features, [], 1);

out.prewlst = prewlst;
out.pswitchlst = pswitchlst;
out.gammalst = gammalst;
out.epslst = epslst;

end