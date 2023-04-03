function [out, opts] = load_and_run(prob, opts)

if ~isfield(opts, 'usepca')
    usepca = 0;
else
    usepca = opts.usepca;
end

if ~isfield(opts, 'version')
    version = '092521';
else
    version = opts.version;
end

if ~isfield(opts, 'rotations')
    opts.rotations = {};
end

folder = fullfile('/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/processed_data/svm/configs/', version);

filenames = dir(fullfile(folder, sprintf('opts_prob%.1f*.mat', prob)));
if numel(filenames) == 0
    error('No opts files found!')
elseif numel(filenames) > 1
    filenames = dir(fullfile(folder, sprintf('opts_prob%.1f*_final.mat', prob)));
end
    

% Deprecated: files used for previous figure plotting
% switch prob
%     case 0
%         filename = 'opts_prob0.0-2021-09-25 20.52.mat';
%     case 0.1
%         filename = 'opts_prob0.1-2021-09-25 21.44.mat';
%     case 0.2
%         filename = 'opts_prob0.2-2021-09-25 21.57.mat';
%     case 0.3
%         filename = 'opts_prob0.3-2021-09-25 22.29.mat';
% end

load(fullfile(folder, filenames(1).name));
opts.save = 0;
opts.savefeatures = 0;
opts.usepca = usepca;
opts.version = version;


if strcmp(version, '092521')
    % old version, need to update the root directory
    opts.rootdir = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/processed_data/simdata';
    opts.filestem{1} = 'EGreedyQLearningAgent-withCorr-doublesigmoid-prob%.2fto%.2f-%s.mat';
    opts.filestem{2} = 'EGreedyinf-basedAgent-withCorr-doublesigmoid-prob%.2fto%.2f-%s.mat';
end
% changed 9.30.21 from run_watershed to run_watershed_pca
if usepca == 1
    [idx, out] = run_watershed_pca(opts);
elseif usepca == 0
    [idx, out] = run_watershed(opts);
else
    idx = out.idx;
end

% Rotation for idx
if usepca < 2 && isfield(opts, 'rotations')
    for i = 1:numel(opts.rotations)
        idx = myrotate(idx, opts.rotations{i});
    end
end

% if opts.usepca
%     switch prob
%         case 0
%             idx = myrotate(idx, [1,2,3,4]);
%         case 0.1
% %             idx = myrotate(idx, [6, 1, 3]);
% %             idx = myrotate(idx, [4 5]);
%         case 0.2
% %             idx = myrotate(idx, [2, 1, 4]);
% %             idx = myrotate(idx, [5, 3]);
%         case 0.3
% %             idx = myrotate(idx, [4, 1]);
% %             idx = myrotate(idx, [2, 5, 3]);
%     end
% else
%     switch prob
%         case 0
%             idx = myrotate(idx, [3, 2, 4]);
%         case 0.1
%             idx = myrotate(idx, [6, 1, 3]);
%             idx = myrotate(idx, [4 5]);
%         case 0.2
%             idx = myrotate(idx, [2, 1, 4]);
%             idx = myrotate(idx, [5, 3]);
%         case 0.3
%             idx = myrotate(idx, [4, 1]);
%             idx = myrotate(idx, [2, 5, 3]);
%     end
% end
%[idxQ, idxIB] = reshapeidx(idx, out);

out.idx = idx;

% Project X on PC space
[~,~,V] = svd(out.features_norm);
out.V = V;

end


