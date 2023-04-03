% load expdata/expfit_params.mat so that we have the animal list
filterval = 20; %filter values such that expoffsets_all(expoffsets_all > 20) = filterval;
files = dir('behavior_data/fitparams_session_averaged.mat');
assert(numel(files) == 1)
load(fullfile(files(1).folder, files(1).name), 'fitparams_all')
animals_ids = fields(fitparams_all);


%% Parse offset for all animals
for i = 1:numel(animals_ids)
    offsetlst = parse_offset_lst(animals_ids{i});
    offsets_all{i} = offsetlst;
end

% Pad to same length
offset_arr = pad_to_same_length(offsets_all);

%% Plot
cols = paperaesthetics;
redcol = cols.redcol;
Nsess = 30;
figure;
offsetmeans = nanmean(offset_arr(:, 1:Nsess));
offseterr = nanstd(offset_arr(:, 1:Nsess), [], 1) ./ sqrt(sum(~isnan(offset_arr(:, 1:Nsess))));

offset_wsls = 1;

hold on
errorbar(1:Nsess, offsetmeans, offseterr, 'o', ...,
    'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'Color', 'k')
plot([1, Nsess], [offset_wsls offset_wsls], '--', 'Color', 'k', 'LineWidth', 2)
text(25, 1.2, 'WSLS', 'FontSize', 20, 'Color', 'k')

ylim([1 4])

mymakeaxis('x_label', 'Session', 'y_label', '# initial errors',...
    'yticks', 1:4,'font_size', 22)



function offsetlst = parse_offset_lst(animal)
% Parse the lapse in the performance,
% as approximated by the accuracy of the last 10 trials in the block
rootdir = 'behavior_data';

fname = sprintf('%s_all_sessions.mat', animal);
load(fullfile(rootdir, fname));

%load the fitrange file
load('behavior_data/fitranges.mat');


% Determine corresponding fit range
idx = contains(animals, animal);
assert(sum(idx) == 1);
fitrange = ranges{idx} + 1;
offsetlst = [];

for i = 1:numel(fitrange)
    offsetlst(i) = parse_offset_helper(feedbacks_cell{fitrange(i)}, ...
        targets_cell{fitrange(i)});
end
end

%%


function mean_offsets = parse_offset_helper(feedbacks, targets)
% feedbacks: array of feedbacks of the session
% targets: array of targets of the session
% Returns: estimated offset: number of incorrect trials 
% at the beginning of each block
% 
blockstarts = find(diff(targets)) + 1;
blockstarts = [1 blockstarts numel(targets) + 1];
offsets = [];

for i = 1:numel(blockstarts) - 1
    fbblock = feedbacks(blockstarts(i) : blockstarts(i + 1) - 1);
    offset = find(fbblock, 1, 'first') - 1;
    if isempty(offset)
        offset = numel(fbblock);
    end
    offsets(i) = offset;

end

mean_offsets = mean(offsets);


end


function loss = lossfun(p, x, y)
A = p(1);
k = p(2);
c = p(3);
ypred = A * exp(-k*x) + c;
loss = sum((ypred - y).^2);
end


function loss = losssigmoid(p, x, y)
lapseL = p(4);
slope = p(1);
offset = p(2);
lapseR = p(3);
ypred = lapseL + (1 - lapseL - lapseR) ./ (1 + exp(-slope * (x - offset)));
loss = sum((ypred - y).^2);

end