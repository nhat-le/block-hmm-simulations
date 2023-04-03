%% Script for standardizing classification and segmentation for all probabilities

% Load raw data
load('opts_prob1.0_tsne.mat', 'opts', 'out');
res1 = out;
opts_out = opts;



%% Decoding analysis (all probabilities)
% paths = pathsetup('matchingsim');
opts.nClasses = max(res1.idx);
opts.reps = 20;
opts.method = 'knn';
opts.nNeighbors = 1;
opts.save_model = 0;


% [counts_allprob1, Mdls1, MCCs] = do_decoding(1, res1, opts);


%%
MCCs_means = [];
MCCs_stds = [];
MCCs_all = {};
Models = {};
confusions_all = {};


model_types = 2:2:30; 

seed = 14;
rng(seed);

for i = 1:numel(model_types)
    fitval = model_types(i);
    if fitval == -1
        opts.method = 'svm';
    else
        opts.method = 'knn';
        opts.nNeighbors = fitval;
    end
    [confusions, Mdl, MCCs] = do_decoding(1, res1, opts);
    confusions_all{i} = confusions;
    MCCs_means(i) = mean(MCCs);
    MCCs_stds(i) = std(MCCs);
    Models{i} = Mdl;
    MCCs_all{i} = MCCs;

end
    


%% Plot performance
meanperfs = [];
stdperfs = [];
for i = 1:numel(confusions_all)
    perfs = [];
    for j = 1:numel(confusions_all{1})
        confusion_mat = confusions_all{i}{j};
        perfs(j) = sum(diag(confusion_mat)) / sum(confusion_mat(:));
    end
    
    meanperfs(i) = mean(perfs);
    stdperfs(i) = std(perfs);
    
end

figure;
cols = paperaesthetics;
l1 = errorbar(model_types, meanperfs, stdperfs, 'o', 'MarkerFaceColor', cols.bluecol,...
    'MarkerEdgeColor', cols.bluecol);
hold on
l2 = errorbar(model_types, MCCs_means , MCCs_stds, 'o', 'MarkerFaceColor', cols.redcol,...
    'MarkerEdgeColor', cols.redcol);
% ylim([0.9, 0.95])

mymakeaxis('x_label', 'k Neighbors', 'y_label', 'Decoding performance', 'xticks', model_types,...
    'font_size', 20);
leg = legend([l1, l2], {'Accuracy', 'Matthews correlation'}, 'FontSize', 16);



%% Plot confusion matrix
figure;
idbest = 12;
cm = confusionchart(confusions_all{idbest}{1}, {'Q1', 'Q2', 'Q3', 'Q4', 'IB5', 'IB6'});
sortClasses(cm, {'Q1', 'Q2', 'Q3', 'Q4', 'IB5', 'IB6'});
% cm.RowSummary = 'row-normalized';
cm.FontSize = 20;
cm.FontName = 'helvetica';
cm.Normalization = 'row-normalized';


%%
% opts.save_model = 1;
notes = 'knn models with k = 2:2:30';
savename = 'decoding_common_Mdl_knn_svm_tsne.mat';
if opts.save_model
    if ~exist(savename, 'file')
%         save(savename, 'counts_allprob1', 'counts_allprob09', 'counts_allprob08',...
%             'counts_allprob07', 'Mdls1', 'Mdls09', 'Mdls08', 'Mdls07')
        save(savename, 'MCCs_means', 'MCCs_stds', 'Models', 'notes', 'MCCs_all', 'model_types', 'seed');
        fprintf('File saved!\n');
    else
        fprintf('File exists, skipping save...\n')
    end
end

function [confusions, Mdls, MCCs] = do_decoding(prob, res, opts)

if ~isfield(opts, 'method'); opts.method = 'knn'; end
if ~isfield(opts, 'nNeighbors'); opts.nNeighbors = 5; end


opts.prob = prob;

load('decodingresults_from_pickle_prob0.00.mat');

[idxQ, idxIB] = reshapeidx(res.idx, res);


idxQrep = repmat(idxQ, [1 1 50]);
idxIBrep = repmat(idxIB, [1 1 50]);

%unroll the matrices
idxQall = reshape(idxQrep, [], 1);
idxIBall = reshape(idxIBrep, [], 1);

IBeffall = reshape(IBeff_arr, [], 1);
IBslopeall = reshape(IBslope_arr, [], 1);
IBlapseall = reshape(IBlapse_arr, [], 1);
IBoffsetall = reshape(IBoffset_arr, [], 1);
Qeffall = reshape(Qeff_arr, [], 1);
Qslopeall = reshape(Qslope_arr, [], 1);
Qlapseall = reshape(Qlapse_arr, [], 1);
Qoffsetall = reshape(Qoffset_arr, [], 1);

Qoffsetall(Qoffsetall < -20) = -20;
IBoffsetall(IBoffsetall < -20) = -20;


features = [IBoffsetall IBslopeall IBlapseall IBeffall;
    Qoffsetall Qslopeall Qlapseall Qeffall];
% features = [IBeffall IBlapseall IBslopeall IBoffsetall;
%     Qeffall Qlapseall Qslopeall Qoffsetall];
% features_norm = (features - mean(features, 1)) ./ std(features, [], 1);

labels = [idxIBall; idxQall];

% To balance the number of examples for each class
counts = [];
for ilabel = 1:opts.nClasses
    counts(ilabel) = sum(labels == ilabel);
end

mincounts = min(counts);
filteredIDs = [];
for ilabel = 1:opts.nClasses
    label_pos = find(labels == ilabel);
    filtered_lbl_pos = randsample(label_pos, mincounts, false);
    filteredIDs = [filteredIDs; filtered_lbl_pos];
end


%shuffle
confusions = {};
Mdls = {};
MCCs = [];

for k = 1:opts.reps
%     order = randperm(numel(labels));
%     fprintf('Repetition %d\n', k);
    order = randsample(filteredIDs, numel(filteredIDs), false);
    labels_shuffled = labels(order);
    features_shuffled = features(order,:);

    %80% training, 20% testing
%     rng('shuffle');
    ntrain = floor(numel(filteredIDs) * 0.8);
    Xtrain = features_shuffled(1:ntrain,:);
    ytrain = labels_shuffled(1:ntrain);
    Xtest = features_shuffled(ntrain + 1:end,:);
    ytest = labels_shuffled(ntrain + 1:end);
    
    % TODO: check if need to transpose for SVM model, if not, do an if else
    % condition check
    
    if strcmp(opts.method, 'knn')
        Mdl = fitcknn(Xtrain,ytrain,'NumNeighbors', opts.nNeighbors,'Standardize',1);
        ypred = Mdl.predict(Xtest);
    else
%         t = templateSVM('Standardize',true, 'KernelFunction', 'rbf');

        t = templateSVM('Standardize',true,'KernelFunction','rbf', 'Standardize', 1);
        Mdl = fitcecoc(Xtrain',ytrain,'Learners',t,'ObservationsIn','columns');
        ypred = Mdl.predict(Xtest);
    end

    % Make the confusion matrix
    N = numel(unique(ypred));
%     fprintf('Number of clusters = %d\n', N);
    
    counts = confusionmat(ytest, ypred); %confusion matrix
    MCCs(k) = matthews_corr(counts');
    % Matthews correlation coefficient
    
    
    confusions{k} = counts';
    Mdls{k} = Mdl;
    
end
end