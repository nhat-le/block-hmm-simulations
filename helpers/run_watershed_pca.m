function [idx,out] = run_watershed_pca(opts)
rng(opts.seed);

%% form the feature vectors
[out,opts] = load_data(opts);

%
[~,~,V] = svd(out.features_norm);
Y = out.features_norm * V;
Y = Y(:, 1:2);
% D = pdist(out.features_norm);
% D = squareform(D);
% [Y,e] = cmdscale(D,2);

% figure;
% plot(Y(:,1), Y(:,2), '.');


%%
f = figure;
h = histogram2(Y(:,1), Y(:,2), opts.nbinhist);
vals = h.Values;
binX = h.XBinEdges;
binY = h.YBinEdges;

% Lowpass filter
vals = conv2(double(vals), double(ones(opts.kernelsize, opts.kernelsize)), 'same');
vals = imhmin(-vals, opts.imhmin);
close(f)

% Watershed!
L = watershed(vals);
Lfill = remove_borders(L);

%
% labels = assign_labels(Y, L, binX, binY, 'Lfill', Lfill);
labels = assign_labels(Y, L, binX, binY, 'nearest');

nLlabels = unique(Lfill);
fprintf('unique class labels: %d\n', numel(nLlabels));
idx = labels;

out.Y = Y;

end