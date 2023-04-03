%%
filterval = 20; %filter values such that expoffsets_all(expoffsets_all > 20) = filterval;
files = dir('behavior_data/fitparams_session_averaged.mat');
assert(numel(files) == 1)
load(fullfile(files(1).folder, files(1).name), 'fitparams_all')
animals_ids = fields(fitparams_all);


%% Parse lapse for all animals
for i = 1:numel(animals_ids)
    lapselst = parse_lapse_lst(animals_ids{i});
    lapses_all{i} = lapselst;
end

% Pad to same length
lapse_arr = pad_to_same_length(lapses_all);

%% Plot the late errors
cols = paperaesthetics;
redcol = cols.redcol;
Nsess = 30;
lapse_wsls = 1;
figure;
lapsemeans = nanmean(lapse_arr(:, 1:Nsess));
lapseerr = nanstd(lapse_arr(:, 1:Nsess), [], 1) ./ sqrt(sum(~isnan(lapse_arr(:, 1:Nsess))));

hold on

errorbar(1:Nsess, lapsemeans, lapseerr, 'o', ...,
    'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'Color', 'k')
plot([1, Nsess], [lapse_wsls lapse_wsls], '--', 'Color', 'k', 'LineWidth', 2)
text(25, 0.97, 'WSLS', 'FontSize', 20, 'Color', 'k')

ylim([0.5 1])

mymakeaxis('x_label', 'Session', 'y_label', 'Late performance',...
    'yticks', 0.5:0.1:1,'font_size', 22)






function lapselst = parse_lapse_lst(animal)
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
lapselst = [];

for i = 1:numel(fitrange)
    if i == 1
        disp('here')
    end
    lapselst(i) = parse_lapse_helper(feedbacks_cell{fitrange(i)}, ...
        targets_cell{fitrange(i)});
end
end



function lapse = parse_lapse_helper(feedbacks, targets)
% feedbacks: array of feedbacks of the session
% targets: array of targets of the session
% Returns: estimated lapse: mean performance of last 10 trials
% 
blockstarts = find(diff(targets)) + 1;
blockstarts = [1 blockstarts numel(targets) + 1];
Ncorr = 0;
Ntotal = 0;
for i = 1:numel(blockstarts) - 1
    startidx = max(blockstarts(i + 1) - 10, blockstarts(i));
    endidx = blockstarts(i + 1) - 1;
    if endidx - startidx + 1 < 10
        continue
    end
    Ncorr = Ncorr + sum(feedbacks(startidx:endidx));
    Ntotal = Ntotal + (endidx - startidx + 1);
end

lapse = Ncorr / Ntotal;

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