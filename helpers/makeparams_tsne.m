function opts = makeparams_tsne(opts)
% Parameters for tsne
opts.perplexity = 30;
opts.nbinhist = 25;
opts.imhmin = 3;
opts.kernelsize = 4;
opts.seed = 5;
opts.MaxIter = 1200;
opts.groups = {[3,5]};
opts.rotations = {[4,1,6,3], [2,5]};

end