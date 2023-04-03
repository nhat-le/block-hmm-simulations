function [all_aggparams, aggmeans_all, statesFlat, features_flat, MCC] = apply_model(aggparams, aggmeans, opts)
% Load the model 
load(fullfile(opts.svmmodelpath, opts.model_name))

all_aggparams = cell2mat(aggparams);
Mdl = Models{opts.mdltype}{opts.mdlid};
MCC = MCCs_all{opts.mdltype}(opts.mdlid);
offsetFlat = all_aggparams(1,:)';
slopesFlat = all_aggparams(2,:)';
lapseFlat = all_aggparams(3,:)';
effFlat = all_aggparams(4,:)';

% Note: no normalization since normalization already handled in svm Mdl
features_flat = [-offsetFlat slopesFlat lapseFlat effFlat];

features_flat(features_flat < -20) = -20; 

% features_flat(4, 2) = 0.07;
statesFlat = Mdl.predict(features_flat);

% Grouping and plotting the transition function by decoded states
aggmeans_all = cell2mat(aggmeans');
end





