function [idx,out] = run_watershed_kmeans(opts, out)
rng(opts.seed);

%% form the feature vectors
[out,opts] = load_data(opts);

all_transfuncs = [];
xvals = 1:15;

for i = 1:size(out.features, 1)
    mu = -out.features(i, 4);
    sigma = out.features(i, 3);
    lapse = out.features(i, 2);
    transfunc = mathfuncs.sigmoid(xvals, mu, sigma, lapse);
    all_transfuncs(end+1,:) = transfunc;
    
end

idx = kmeans(all_transfuncs, opts.kclust, 'Distance', 'cityblock');
idx = rotate_idx(idx, opts.rotations);
out.idx = idx;

idx = labels;

out.Y = Y;

end