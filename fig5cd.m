%% Build the options for the run
% where the Python simulation results are stored (model-free/inf-based
% simulations)
opts = struct;
opts.rootdir = "simdata";
opts.expfolder = '092321';
opts.prob = 1;
opts.clipmode = 2;
opts = makeparams_tsne(opts);

% script operations
opts.save = 0;
opts.plotfeatures = 0;
opts.show_trans_functions = 0;
opts.show_refs = 0; %whether to show ref markers
opts.method = 'tsne'; %'multidim-watershed or 'watershed', 'watershed-tsne', or 'dbscan', or 'gmm'


colors = brewermap(12, 'Paired');
colors2 = brewermap(9, 'Set1');
opts.cmap = colors([1,5,7,11,9,3,2,4,6,8,10],:); %permute to align with MATLAB default..
opts.cmap2 = colors2([2,1,5,6,4,3,7,8,9],:);
opts.cmap2(4,:) = [255, 238, 21] / 255;

%% form the feature vectors
rng(opts.seed)
% rng(8)

[out,opts] = load_data(opts);


D = pdist(out.features_norm);
D = squareform(D);


Y = tsne(out.features_norm, 'Perplexity', opts.perplexity, 'Distance', 'euclidean', ...
    'Options', opts);
h = histogram2(Y(:,1), Y(:,2), opts.nbinhist);
vals = h.Values;
binX = h.XBinEdges;
binY = h.YBinEdges;
close(gcf);


% Lowpass filter
vals = conv2(double(vals), double(ones(opts.kernelsize, opts.kernelsize)), 'same');
vals = imhmin(-vals, opts.imhmin);

% Watershed!
L = watershed(vals);
Lfill = remove_borders(L);

figure;
subplot(131)
imagesc(vals')
colormap gray
caxis([-200 100])
axis xy

subplot(132)
valsold = vals;
vals(L == 0) = 100;
imagesc(vals')
caxis([-200 100])
axis xy


subplot(133)
imagesc(Lfill', 'AlphaData', -valsold' / -min(valsold(:)) * 3);
colormap jet
axis xy


%%
% perform dbscan segmentation
idx = assign_labels(Y, L, binX, binY, 'nearest');


% post-processing



fprintf('unique class labels: %d\n', numel(unique(idx)));
labels_name = unique(idx)';

idx = group_idx(idx, opts.groups);
idx = rotate_idx(idx, opts.rotations);
out.idx = idx;

all_transfuncs = [];
xvals = 1:15;

for i = 1:size(out.features, 1)
    mu = -out.features(i, 4);
    sigma = out.features(i, 3);
    lapse = out.features(i, 2);
    transfunc = mathfuncs.sigmoid(xvals, mu, sigma, lapse);
    all_transfuncs(end+1,:) = transfunc;
    
end

figure;
opts.kclust = max(idx);
for clustid = 1:max(idx)
    subplot(2, ceil(opts.kclust/2), clustid)
    hold on
    transfunc_sub = all_transfuncs(idx == clustid, :)';
    plot(transfunc_sub(:,:), 'k', 'LineWidth', 0.25);
    plot(mean(transfunc_sub, 2), 'r', 'Color', 'r', 'LineWidth', 2);
    
    ylim([0, 1])
    mymakeaxis('x_label', 'Trials in block', 'y_label', 'P(Correct)', ...
        'yticks', 0:0.2:1, 'xticks', 0:5:15);
end
clustfig = gcf;


%
[idxQ, idxIB] = reshapeidx(idx, out);
[idxQorder, idxIBorder] = reshapeidx(1:numel(idx), out);
opts.N = numel(unique(idx));

cmap = opts.cmap(1:opts.N,:);
% cmap = brewermap(opts.N, 'Blues');
cmap = [cmap; 0.2 0.2 0.2];
%%
h = produce_heatmap(idxIB, out.prewlst, out.pswitchlst, 'clim', [0.5 opts.N+1.5], 'legendname', 'Performance regime', ...
    'x_label', '$P_{rew}$', 'y_label', '$P_{switch}$', 'ytickvalues', 0:0.1:0.4, ...
    'cmap', cmap, 'limits', [0.5, opts.N + 0.5], 'font_size', 25);
hold on
if opts.show_refs
    plot(out.prewlst(end), out.pswitchlst(end), 'ko', 'MarkerSize', 20, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 2);
end
ibfig = gcf;
%%

produce_heatmap(idxQ, out.epslst, out.gammalst, 'clim', [0.5 opts.N+1.5], 'legendname', 'Performance regime', ...
    'x_label', '$\epsilon$', 'y_label', '$\gamma$', 'cmap', cmap, 'limits', [0.5, opts.N + 0.5],...
    'font_size', 25);
hold on
if opts.show_refs
    plot(out.epslst(5), out.gammalst(9), 'ko', 'MarkerSize', 20, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 2);
end
qfig = gcf;


%% Visualize tsne embedding
[~,~,V] = svd(out.features_norm);
out.V = V;
out.Y = Y;

colors = opts.cmap2;
% colors = brewermap(7, 'Set1');
% colors = colors([2,1,5,6,4,3,7],:);

figure()
    
hold on
for i = 1:numel(unique(idx))
    col2use = colors(i,:);
      
    plot(out.Y(idx == i, 1), out.Y(idx == i, 2), 'o', ...
        'MarkerFaceColor', col2use, 'MarkerEdgeColor', 'w', 'MarkerSize', 10)
end

if opts.show_refs
    %plot landmark points
    plot(out.Y(150, 1), out.Y(150, 2), 'kx', 'MarkerSize', 10, 'LineWidth', 1)
    plot(out.Y(259, 1), out.Y(259, 2), 'kx', 'MarkerSize', 10, 'LineWidth', 1)
end

mymakeaxis('x_label', 'z_1', 'y_label', 'z_2', 'font_size', 25)

tsnefig = gcf;



%% Save if requested

if opts.save  
    % Save options
    filename = 'opts_prob1.0_tsne.mat';
    opts.fig = [];
    if ~exist(filename, 'file')
        save(filename, 'opts', 'out');
    else
        fprintf('File exists, skipping save: %s\n', filename)
    end
    
    fprintf('Files saved!\n');
    close all

end







