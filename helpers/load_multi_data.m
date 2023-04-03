function [out,opts] = load_multi_data(opts)
% Load features of behavioral simulation

%% parse options
if ~isfield(opts, 'rootdir')
    opts.rootdir = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/simdata';
end

if ~isfield(opts, 'filestem')
    opts.filestem = 'svmresults_from_pickle_092221_prob%.2f.mat';
end

if ~isfield(opts, 'prob'); opts.prob = 1; end
if ~isfield(opts, 'outliermode'); opts.outliermode = 1; end

if ~isfield(opts, 'savepath')
    opts.savepath = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/PaperFigures/decodeFigs';
end

%%
filedir = sprintf(['%s/' opts.filestem], opts.rootdir, 1-opts.prob);
load(filedir);


% load(filedir, 'efflist', 'LapseL', 'PLslopelist', 'PLoffsetlist', 'epslst', 'gammalst');

Qeff_flat = reshape(efflist, [], 1);
Qlapse_flat = reshape(LapseL, [], 1);
Qslope_flat = reshape(PLslopelist, [], 1);
Qoffset_flat = reshape(PLoffsetlist, [], 1);

out.Qdim = size(efflist);

filedir = sprintf(['%s/' opts.filestem{2}], opts.rootdir, 1-opts.prob, opts.prob);
load(filedir, 'efflist', 'LapseL', 'PLslopelist', 'PLoffsetlist', 'prewlst', 'pswitchlst');

IBeff_flat = reshape(efflist, [], 1);
IBlapse_flat = reshape(LapseL, [], 1);
IBslope_flat = reshape(PLslopelist, [], 1);
IBoffset_flat = reshape(PLoffsetlist, [], 1);

% Filter outliers
IBoffset_flat(IBoffset_flat < -20) = -3; %-20;
Qoffset_flat(Qoffset_flat < -20) = -3; %-20;


% Parse the outputs
out.IBdim = size(efflist);

out.features = [IBeff_flat IBlapse_flat IBslope_flat IBoffset_flat;
    Qeff_flat Qlapse_flat Qslope_flat Qoffset_flat];

out.features_norm = (out.features - nanmean(out.features, 1)) ./ nanstd(out.features, [], 1);

out.prewlst = prewlst;
out.pswitchlst = pswitchlst;
out.gammalst = gammalst;
out.epslst = epslst;

end