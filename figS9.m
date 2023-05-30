%% process K selection results
filepath = '/Users/minhnhatle/Documents/Sur/MatchingSimulations/processed_data/blockhmmfit/K_selection';
load(fullfile(filepath, 'blockhmm_validation_050823.mat'));

nmodes_Ksel = argmin(-ll_lst_all(:, 1:6)');


%% Load the data

files = dir('/Users/minhnhatle/Documents/block-hmm-simulations/animal_blockHMM/offset_shift3_fit/*.mat');

params_all = [];
nmodes = [];
animal_names = {};
perf_flat = [];

transfunc_all_animals = [];

for idx = 1:21
    load(fullfile(files(idx).folder, files(idx).name));

    %parse animal name
    parts = strsplit(files(idx).name, '_');
    animal_names{idx} = parts{1};

    params_all = [params_all params];
    nmodes(idx) = size(params, 2);

    % get the transition function and performance
    xvals = 3:25;
    L = numel(xvals);
    transfunc_all = [];
    for i = 1:size(params, 2)
        transfunc_all(i,:) = mathfuncs.sigmoid(xvals, params(1,i), params(2,i), params(3,i));
    end
    
    % sort in increasing order of efficiencies
    perfs = sum(transfunc_all, 2) / L;
    perf_flat = [perf_flat perfs'];

    % gather the transfunc for all animals
    xvals = 0:25;
    L = numel(xvals);
    transfunc_all_full = [];
    for i = 1:size(params, 2)
        transfunc_all_full(i,:) = mathfuncs.sigmoid(xvals, params(1,i), params(2,i), params(3,i));
    end

    transfunc_all_animals = [transfunc_all_animals; transfunc_all_full];
    
end

%%

opts.perf_thres1 = 0.65;
opts.perf_thres2 = 0.84;
opts.lapse_thres = 0.1;
offset_flat = params_all(1,:);
lapse_flat = params_all(3, :);
class_by_perf = [];
for i = 1:numel(perf_flat)
    if perf_flat(i) < opts.perf_thres1
        class_by_perf(i) = 1;
    elseif perf_flat(i) < opts.perf_thres2
        if lapse_flat(i) < 0.1
            class_by_perf(i) = 2;
        else
            class_by_perf(i) = 3;
        end
    else
        class_by_perf(i) = 4;
    end
end

early_group = find(class_by_perf == 4 & offset_flat < 3);

figure;
scatter(offset_flat - 3, lapse_flat, [], class_by_perf, 'filled');
hold on
xlim([-3 20])
plot(offset_flat(early_group) - 3, lapse_flat(early_group), 'ko', 'LineWidth', 2);

hold on
vline(0, 'k--')
cmap = brewermap(6, 'Set2');
cmap = cmap([3, 4, 6, 1],:);

colormap(gca, cmap);
mymakeaxis('x_label', 'Offset', 'y_label', 'Lapse', 'xticks', [-3 0:5:20], 'font_size', 20)


%% Grouped transition functions

figure('Position', [440,133,338,665])
for i = 1:4
    subplot(4,1,i)
    hold on
    if i == 4
        plot(-3:22, transfunc_all_animals(class_by_perf == 4 & offset_flat < 3, :)', 'r');
        plot(-3:22, transfunc_all_animals(class_by_perf == 4 & offset_flat >= 3, :)', 'k');
    else
        plot(-3:22, transfunc_all_animals(class_by_perf == i, :)', 'k');
    end
    hold on
    hline(1, 'k--')
    vline(0, 'k--')

    if i == 4
        mymakeaxis('x_label', 'Trial # in block', 'y_label', 'Performance', 'xticks', [-5, 0:10:20])
    else
        mymakeaxis('x_label', '', 'y_label', 'Performance', 'xticks', [-5, 0:10:20])
    end

end

%% histogram

histogram(perf_flat, 'BinEdges', linspace(0, 1, 30))
vline([opts.perf_thres1, opts.perf_thres2], 'k--');
mymakeaxis('x_label', 'Mode performance', 'y_label', 'Count', 'font_size', 20)


%% parse the mode evolution of "early modes" over time for each animal

files = dir('/Users/minhnhatle/Documents/block-hmm-simulations/animal_blockHMM/offset_shift3_fit/*.mat');

%note: to produce different slines corresponding to the five groups
% (sline_1, sline_2, sline_3, sline_4early, sline_4late)
% manually comment out the correct definition of early_group.

% early_group = find(class_by_perf == 4 & offset_flat < 3); %sline4_early
% early_group = find(class_by_perf == 4 & offset_flat >= 3); %sline4_late
early_group = find(class_by_perf == 1); %sline_1

nmodes_cumsum = [1 (cumsum(nmodes) + 1)];
early_group_binary = zeros(1, sum(nmodes));
early_group_binary(early_group) = 1;

animalinfo = struct;
for idx = 1:21
    % Load the behavioral file  
    load(fullfile(files(idx).folder, files(idx).name));
    
    % extract the relevant "early group(s)" of the animal
    animal_modes = early_group_binary(nmodes_cumsum(idx) : nmodes_cumsum(idx+1) - 1);
    assert(numel(animal_modes) == size(params, 2));

    zstates_transformed = zeros(1, numel(zstates));
    zstates_transformed(ismember(zstates + 1, find(animal_modes))) = 1;

    zclassified_splits = mat2cell(zstates_transformed, 1, lengths);

    animalinfo(idx).animal = animal_names{idx};
    animalinfo(idx).zclassified = zclassified_splits;
    animalinfo(idx).classes = find(animal_modes);

    fprintf('%d. Animal: %s, n early modes = %d/%d\n', idx, animal_names{idx}, ...
        sum(animal_modes), numel(animal_modes));

end


%% plot the evolution
opts.plotting = 0;
f = waitbar(0);
compositions = {};
Lwindow = 30;
for id = 1 :numel(animalinfo)
    waitbar(id / numel(animalinfo), f);
    zclassified = animalinfo(id).zclassified;
    
    composition = nan(numel(zclassified), 1);
    for i = 1:numel(zclassified)
        z_single_session = zclassified{i};
        composition(i) = sum(z_single_session) / numel(z_single_session);
    end
    
    animalinfo(id).composition = composition;
    nsess = numel(composition);
    xlimmax = min(nsess, Lwindow);

    if opts.plotting
        % Plot bar graph to show the composition
        figure('Position', [440,423,736,375], 'Name', animalinfo(id).animal);
        plot(composition)
        hold on
        xlim([0.5 xlimmax + 0.5])
        ylim([0 1])
        mymakeaxis('x_label', 'Session #', 'y_label', 'Fraction', 'xticks', 0:5:xlimmax)
    end

    Lextracted = min(Lwindow, numel(composition));
    compositions{id} = composition(1:Lextracted)';
end
close(f)

%% Plot average composition
cols = paperaesthetics;
colors = cols.colors;
figure('Position', [440,379,731,419]);
hold on
lines = [];
sline = pad_to_same_length(compositions);

N = sum(~isnan(sline));
h = errorbar(1:size(sline, 2), nanmean(sline, 1), nanstd(sline, [], 1) ./ sqrt(N), ...
    'o-', 'Color', 'k', 'MarkerFaceColor', colors(1,:), 'MarkerEdgeColor', 'k');

lines(i) = h;
%     xlim([1, 40])
ylim([0, 1])


mymakeaxis('x_label', 'Session', 'y_label', 'Fraction', 'xticks', 0:5:Lwindow, 'font_size', 22)
% l = legend(lines, {'Q1', 'Q2', 'Q3', 'Q4', 'IB5', 'IB6'});
% l.Title.String = 'Regime';
% l.Title.FontSize = 12;
% l.FontSize = 12;

%% plot all on same plot
slines_all = {sline_1, sline_2, sline_3, slines_4late, sline_4early};
cols = paperaesthetics;
colors = cols.colors;
figure('Position', [440,379,731,419]);
hold on
lines = [];
% sline = pad_to_same_length(compositions);
cmap = brewermap(6, 'Set2');
cmap = cmap([3, 4, 6, 1],:);
cmap = [cmap; 0 0 0];

for i = 1:5
    sline = slines_all{i};
    N = sum(~isnan(sline));
    h = errorbar(1:size(sline, 2), nanmean(sline, 1), nanstd(sline, [], 1) ./ sqrt(N), ...
        'o-', 'Color', cmap(i,:), 'MarkerFaceColor', cmap(i,:), 'MarkerEdgeColor', 'none');
end


mymakeaxis('x_label', 'Session', 'y_label', 'Fraction', 'xticks', 0:5:Lwindow, 'font_size', 22)
% l = legend(lines, {'Q1', 'Q2', 'Q3', 'Q4', 'IB5', 'IB6'});
% l.Title.String = 'Regime';
% l.Title.FontSize = 12;
% l.FontSize = 12;







