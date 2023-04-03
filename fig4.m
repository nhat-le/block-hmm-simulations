xvals = 0:25;

animal_names = {'e54', 'f11', 'f22', 'f03', 'f20', 'f01', 'e56', 'f12', 'fh03', ...
    'fh02', 'e57', 'f16', 'fh01', 'e46', 'f04', ...
    'f02', 'f23', 'e35', 'f21', ...
    'f17', 'e53'};

sex = {'F', 'F', 'M', 'F', 'M', 'M', 'M', 'F', 'M', ...
    'F', 'F', 'F', 'M', 'M', 'F', ...
    'F', 'M', 'F', 'M', ...
    'F', 'F'};


expdate_lst = {'122221', '113021', '113021', '113021', '113021', ...
    '113021', '122221', '113021', '113021',...
    '122221b', '122221b', '122221b', '122221b', '122221b', '122221b',...
    '122221b', '122221b', '122221b', '122221b',...
    '122221b', '122221b'};


nmodes_lst = [6 6 6 6 6 6 6 6 6, ...
    5 5 5 4 4 4 3 3 3 3 2 2];


opts.perf_thres1 = 0.65;
opts.perf_thres2 = 0.84;
opts.lapse_thres = 0.05;

assert(numel(animal_names) == numel(nmodes_lst));
assert(numel(nmodes_lst) == numel(expdate_lst));
assert(numel(nmodes_lst) == numel(sex));

% collect transition modes for all animals and all modes
transfunc_all_animals = {};
perfs_all = {};
lapses_all = {};
offsets_all = {};
slopes_all = {};
classes_all = {};
for i = 1:numel(nmodes_lst)
    disp(i)
    [transfunc_all_animals{i}, perfs_all{i}, offsets_all{i}, slopes_all{i}, lapses_all{i}, classes_all{i}] = get_transition_modes(animal_names{i}, nmodes_lst(i), expdate_lst{i}, opts);
end




%% fig 4b

histogram(cell2mat(perfs_all') / 26, 'BinEdges', linspace(0, 1, 30))
vline([opts.perf_thres1, opts.perf_thres2], 'k--');
mymakeaxis('x_label', 'Mode performance', 'y_label', 'Count', 'font_size', 20)

%% fig 4c
perf_flat = cell2mat(perfs_all') / 26;
offset_flat = cell2mat(offsets_all);
slope_flat = cell2mat(slopes_all);
lapse_flat = cell2mat(lapses_all);
classes_flat = cell2mat(classes_all);


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

figure;
scatter(offset_flat, lapse_flat, [], class_by_perf, 'filled')
xlim([0 20])

cmap = brewermap(6, 'Set2');
cmap = cmap([3, 4, 6, 1],:);

colormap(gca, cmap);
mymakeaxis('x_label', 'Offset', 'y_label', 'Lapse')

%% Fig 4d
transfunc_all_animals2 = {};
for i = 1:21
    transfunc_all_animals2{i} = transfunc_all_animals{i};
end
transfunc_arr = cell2mat(transfunc_all_animals2');

figure('Position', [440,133,338,665])
for i = 1:4
    subplot(4,1,i)
    plot(transfunc_arr(class_by_perf == i, :)', 'k');
    hold on
    hline(1, 'k--')

    if i == 4
        mymakeaxis('x_label', 'Trial # in block', 'y_label', 'Performance')
    else
        mymakeaxis('x_label', '', 'y_label', 'Performance')
    end

end




%% Fig. 4e

hmmidentities = zeros(21, 6);
curr_idx = 1;
for i = 1:21
    N = numel(perfs_all{i});
    hmmidentities(i, 1:N) = class_by_perf(curr_idx : (curr_idx + N - 1));
    curr_idx = curr_idx + N;
end

hmmidentities = flipud(hmmidentities);

cmap = brewermap(6, 'Set2');
cmap = cmap([3, 4, 6, 1],:);
cmap = [0 0 0; cmap];


figure('Position', [440,50,413,748]);
imagesc(hmmidentities);
colormap(cmap);
hold on
vline((0:6) + 0.5, 'k');
hline((0:21) + 0.5, 'k');

axis xy
mymakeaxis('x_label', 'HMM mode', ...
    'font_size', 22, 'xticks', 1:6, 'yticks', 1:21, 'yticklabels', fliplr(animal_names))

% %% Correspondence between behavior and algorithmic classifications
% hmmidentities_behavior = hmmidentities;
% 
% hmmidentities_flat = hmmidentities(hmmidentities > 0);
% hmmidentities_bflat = hmmidentities_behavior(hmmidentities_behavior > 0);
% 
% breakdowns = {};
% breakdowns{1} = hmmidentities_flat(hmmidentities_bflat == 1);
% breakdowns{2} = hmmidentities_flat(hmmidentities_bflat == 2);
% breakdowns{3} = hmmidentities_flat(hmmidentities_bflat == 3);
% breakdowns{4} = hmmidentities_flat(hmmidentities_bflat == 4);
% 
% compositions = [];
% for i = 1:4
%     compositions(i,:) = parse_composition(breakdowns{i});
% 
% end
% 
% % stacked bar charts
% figure;
% h = barh((compositions ./ sum(compositions, 1))', 'stacked');
% cmap = brewermap(6, 'Set2');
% cmap = cmap([3, 4, 6, 1],:);
% 
% for i = 1:4
%     h(i).FaceColor = cmap(i,:);
%     h(i).ShowBaseLine = 'off';
% end
% 
% mymakeaxis('x_label', '% modes', 'font_size', 25, ...
%     'yticks', 1:6, 'yticklabels', {'Q1', 'Q2', 'Q3', 'Q4', 'IB5', 'IB6'});




%% Fig. S2b
classes_males = hmmidentities(strcmp(sex, 'M'), :);
classes_females = hmmidentities(strcmp(sex, 'F'), :);
composition_males = classes_males(:);
composition_females = classes_females(:);
composition_males = composition_males(composition_males > 0);
composition_females = composition_females(composition_females > 0);

f = figure;
h1 = histogram(composition_males);
hold on
h2 = histogram(composition_females);

n_composition_males = h1.Values / sum(h1.Values);
n_composition_females = h2.Values / sum(h2.Values);
close(f)

% stacked bar
figure;
h = bar([n_composition_males; n_composition_females], 'stacked');
cmap = brewermap(6, 'Set2');
cmap = cmap([3, 4, 6, 1],:);


for i = 1:4
    h(i).FaceColor = cmap(i,:);
    h(i).ShowBaseLine = 'off';
end

mymakeaxis('y_label', '% modes', 'font_size', 25, ...
    'xticks', 1:2, 'xticklabels', {'Male', 'Female'});



%% Fig. S2a
nmodes_male = nmodes_lst(strcmp(sex, 'M'));
nmodes_female = nmodes_lst(strcmp(sex, 'F'));
male_counts = [0 2 2 0 5];
female_counts = [2 2 1 3 4];

figure;
h1 = bar((2:6) - 0.1, male_counts, 0.2, 'k');
hold on
h2 = bar((2:6) + 0.1, female_counts, 0.2, 'b');
mymakeaxis('x_label', 'Number of modes', 'y_label', 'Count', 'font_size', 25, ...
    'xticks', 2:6)
legend([h1, h2], ["Male", "Female"], 'Location', 'northwest', 'FontSize', 16);

% plot(ones(1, numel(nmodes_male)) + rand(1, numel(nmodes_male)) * 0.1 - 0.05, nmodes_male, 'o', 'MarkerFaceColor', 'r', ...
%     'MarkerEdgeColor', 'w', 'MarkerSize', 8)
% hold on
% plot(ones(1, numel(nmodes_female)) * 2 + rand(1, numel(nmodes_female)) * 0.1 - 0.05, nmodes_female, 'o', 'MarkerFaceColor', 'r', ...
%     'MarkerEdgeColor', 'w', 'MarkerSize', 8)
% 
% mymakeaxis('y_label', 'Number of modes', 'xticks', 1:2, 'xticklabels', {'Male', 'Female'})

% histogram(nmodes_male, 'BinEdges', 0.5:6.5)
% histogram(nmodes_female, 'BinEdges', 0.5:6.5)

%% Fig. 4a
animals_to_plot = {'e46', 'e53', 'e56', 'f02', 'fh02', 'fh03'};
Nrows = numel(animals_to_plot);
Ncols = 6;
figure('Position', [440,173,772,625]);
for i = 1:numel(animals_to_plot)
    animal = animals_to_plot{i};
    pos = find(contains(animal_names, animal));
    transfuncs_to_plot = transfunc_all_animals{pos};
    for j = 1:size(transfuncs_to_plot, 1)
        subplot(Nrows, Ncols, (i - 1) * Ncols + j)
        plot(transfuncs_to_plot(j,:), 'LineWidth', 2)
        hold on
        plot([0, 25], [1 1], 'k--')
        ylim([0, 1])
        if j == 1
            mymakeaxis('y_label', upper(animal), 'xticks', [0, 25]) %, 'yticklabels', {'0', '25'})
        else
            mymakeaxis('xticks', [0, 25])
        end
    end

end


function composition = parse_composition(arr)

composition = [];
for i = 1:6
    composition(i) = sum(arr == i) / numel(arr);
end


end






function [transfunc_all, perf_sort, offsets_sort, slopes_sort, lapses_sort, classes] = get_transition_modes(animal, nmodes, expdate, opts)

% rootdir = '/Users/minhnhatle/Documents/Sur/MatchingSimulations/processed_data/blockhmmfit/';
rootdir = '/Users/minhnhatle/Documents/Sur/MatchingSimulations/PaperFigures/code/expfit/plos_code/animal_blockHMM';
filename = sprintf('%s/121821bK%d/%s_hmmblockfit_%s_saved_121821bK%d.mat', ...
    rootdir, nmodes, animal, expdate, nmodes);
load(filename, 'params');
xvals = 0:25;
L = numel(xvals);

transfunc_all = [];
for i = 1:size(params, 2)
    transfunc_all(i,:) = mathfuncs.sigmoid(xvals, params(1,i), params(2,i), params(3,i));
end

% sort in increasing order of efficiencies
perfs = sum(transfunc_all, 2);
[perf_sort,idx] = sort(perfs);
transfunc_all = transfunc_all(idx, :);
lapses = params(3,:);
offsets = params(1,:);
slopes = params(2,:);
lapses_sort = lapses(idx);
offsets_sort = offsets(idx);
slopes_sort = slopes(idx);

% classify
classes = [];
for i = 1:numel(perf_sort)
    if perf_sort(i) / L < opts.perf_thres1
        classes(i) = 1;
    elseif perf_sort(i) / L < opts.perf_thres2 && lapses_sort(i) > 0.05
        classes(i) = 2;
    elseif perf_sort(i) / L < opts.perf_thres2
        classes(i) = 3;
    else
        classes(i) = 4;
    end

end

end