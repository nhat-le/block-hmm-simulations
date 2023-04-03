function info = get_animal_raw_data(animal, sessid)
% animal: str, animal name
% sess_id: int, session number
% Returns: info, a struct containing
% feedbacks, targets: information about the raw performance
% of the animal during the session

rootdir = '/Users/minhnhatle/Documents/Sur/MatchingSimulations/processed_data/expdata/122221b';
fname = sprintf('%s_all_sessions_122221b.mat', animal);
load(fullfile(rootdir, fname), 'feedbacks_cell', 'targets_cell');

%load the fitrange file
paths = pathsetup('matchingsim');
load(fullfile(paths.expdatapath, '102121', 'fitranges_122221.mat'), 'ranges', 'animals');

% Determine corresponding fit range
idx = contains(animals, animal);
assert(sum(idx) == 1);
fitrange = ranges{idx} + 1;

if sessid > numel(fitrange)
    info = nan;
else
    id = fitrange(sessid);
    info.feedbacks = feedbacks_cell{id};
    info.targets = targets_cell{id};
end