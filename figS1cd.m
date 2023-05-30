%% Analysis of side bias

%% Process L-R difference for all animals
load('behavior_data/fitranges.mat');
path = '/Users/minhnhatle/Documents/Sur/MatchingSimulations/processed_data/expdata/122221b';
% files = dir(sprintf('%s/*_all_sessions_122221b.mat', path));
files = dir('behavior_data/fitparams_session_averaged.mat');
assert(numel(files) == 1)
load(fullfile(files(1).folder, files(1).name));
animals_ids = fields(fitparams_all);
plot_window = 25;
Nsessions_max = 30;

animals_dirright_cell = {};
animals_dirleft_cell = {};
animals_diff_cell = {};




for i = 1:numel(animals_ids)
    animal_id = animals_ids{i};
    idx = find(strcmp(animals, animal_id));
    Nsessions = numel(ranges{idx});

    disp(animal_id)

    animal_dirright_all = [];
    animal_dirleft_all = [];
    for sessid = 1:min(Nsessions, Nsessions_max)
        [dirrightperf, dirleftperf] = get_left_right_performance(animal_id, sessid, plot_window);
        animal_dirright_all(sessid) = dirrightperf;
        animal_dirleft_all(sessid) = dirleftperf;
    end

    animals_dirleft_cell{i} = animal_dirleft_all;
    animals_dirright_cell{i} = animal_dirright_all;
    animals_diff_cell{i} = animal_dirleft_all - animal_dirright_all;
end

%% Do stats
animals_left_arr = pad_to_same_length(animals_dirleft_cell);
animals_right_arr = pad_to_same_length(animals_dirright_cell);

pvals = [];
for i = 1:30
    left_vals = animals_left_arr(:, i);
    right_vals = animals_right_arr(:, i);
    pvals(i) = signrank(left_vals, right_vals);
end

%% Plot mean bias across all animals
animals_diff_arr = pad_to_same_length(animals_diff_cell);

figure('Position', [440,422,768,376]);
cols = paperaesthetics;
hold on;

meanbias = nanmean(animals_diff_arr);
stdbias = nanstd(animals_diff_arr);
errorbar(1:30, meanbias, stdbias, 'o', 'LineWidth', 2, 'MarkerFaceColor', cols.bluecol)
xlim([-3, 33])
% plot(1:30, pvals)
ylim([-1, 1])
hline(0, 'k--')
mymakeaxis('x_label', 'Session', 'y_label', 'Left - right difference in performance')

%% Bias vs overall performance vs # modes
% load expdata/expfit_params.mat
filterval = 20; %filter values such that expoffsets_all(expoffsets_all > 20) = filterval;
files = dir('behavior_data/fitparams_session_averaged.mat');
assert(numel(files) == 1)
load(fullfile(files(1).folder, files(1).name));

% Load the fitrange file
load('behavior_data/fitranges.mat');

Nsess = 30;

animals_nmodes = {'fh03',...
    'f16',...
    'f21',...
    'f12',...
    'f04',...
    'e56',...
    'e35',...
    'f01',...
    'e57',...
    'e53',...
    'fh02',...
    'f17',...
    'f20',...
    'f03',...
    'f22',...
    'f11',...
    'e46',...
    'f23',...
    'fh01',...
    'f02',...
    'e54'};

nmodes = [6 5 3 6 4 6 3 6 5 2 5 2 6 6 6 6 4 3 4 3 6];

% animals_ids = fields(fitparams_all);
effs_all = {};
nmodes_sort = [];

for i = 1:numel(animals_ids)
    [offsets, slopes, lapses, effs] = parse_params(fitparams_all.(animals_ids{i}));
    
    %Trim based on fitparams
    idx = contains(animals, animals_ids{i});
    assert(sum(idx) == 1);
    fitrange = ranges{idx} + 1;
    effs_all{i} = effs(fitrange);

    % find number of modes
    nmodes_sort(i) = nmodes(find(strcmp(animals_nmodes, animals_ids{i})));

end

% Pad to same length
expeff_all = pad_to_same_length(effs_all);

bias_last_5 = [];
perf_last_5 = [];

% Find performance on the last 5 sessions for each animal
for i = 1:size(expeff_all, 1)
    perf_array = expeff_all(i,:);
    perf_array = perf_array(~isnan(perf_array));
    bias_array = animals_diff_arr(i,:);
    bias_array = bias_array(~isnan(bias_array));

    bias_last_5(i) = mean(bias_array(end-5:end));
    perf_last_5(i) = mean(perf_array(end-5:end));
    
    fprintf("Animal: %s, perf: %.2f,", animals_ids{i}, perf_last_5(i));
    fprintf(" bias: %.2f\n", bias_last_5(i));
end


% plot
figure('Position', [222,35,922,602]);
hold on
colors = [127,201,127;
        190,174,212;
        253,192,134;
        255,255,153;
        56,108,176]/ 255;
% scatter(bias_last_5(1:3), perf_last_5(1:3), [], cellstr(colors(nmodes_sort - 1)), 'filled');
% scatter(bias_last_5, perf_last_5, [], colors(nmodes_sort-1, :), 'filled');

lines = [];
for i = 1:5
    if i == 4
        edgecolor = 'k';
    else
        edgecolor = colors(i,:);
    end
    lines(i) = plot(bias_last_5(nmodes_sort == i + 1), perf_last_5(nmodes_sort == i + 1), ...
        'o', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', edgecolor);
end


hold on
text(bias_last_5 + 0.01, perf_last_5, animals_ids, 'FontSize', 14)
xlim([-0.5, 0.5])
ylim([0.3, 0.8])

mymakeaxis('x_label', 'Left - right bias', 'y_label', 'Performance', ...
    'xticks', -0.4:0.2:0.4, 'font_size', 24)

lgd = legend(lines, {'2', '3', '4', '5', '6'});
lgd.FontSize = 16;
title(lgd, 'Num. blockHMM modes');


%% stats
perf_last_5filt = perf_last_5(bias_last_5 < 0.4);
bias_last_5filt = bias_last_5(bias_last_5 < 0.4);

[r, p] = corrcoef(perf_last_5filt, bias_last_5filt);







function [dirrightperf, dirleftperf] = get_left_right_performance(animal, sessid, plot_window)
% disp(sessid)
[~, raw] = utils.get_perf_animal_from_session(animal, sessid, plot_window);

blockwise_perf = nanmean(raw, 2);

% Find which direction is first block
info = utils.get_animal_raw_data(animal, sessid);
first_block_direction = info.targets(1);

if first_block_direction == 1 %first block: left
    dirleft_performance = blockwise_perf(1:2:end);
    dirright_performance = blockwise_perf(2:2:end);
elseif first_block_direction == 0 %first block: right
    dirright_performance = blockwise_perf(1:2:end);
    dirleft_performance = blockwise_perf(2:2:end);
else
    error('invalid dir')
end

dirrightperf = nanmean(dirright_performance);
dirleftperf = nanmean(dirleft_performance);


end


function [offsets, slopes, lapses, effs] = parse_params(params)

slopesL = params.pL(:,1);
slopesR = params.pR(:,1);
offsetsL = params.pL(:,2);
offsetsR = params.pR(:,2);
lapsesL = params.pL(:,3);
lapsesR = params.pR(:,3);

slopes = nanmin([slopesL, slopesR], [], 2)';
offsets = nanmean([offsetsL, offsetsR], 2)';
lapses = nanmax([lapsesL, lapsesR], [], 2)';
effs  = params.eff;




end



