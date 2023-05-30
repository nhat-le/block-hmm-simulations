% load expdata/expfit_params.mat
filterval = 20; %filter values such that expoffsets_all(expoffsets_all > 20) = filterval;
files = dir('behavior_data/fitparams_session_averaged.mat');
assert(numel(files) == 1)
load(fullfile(files(1).folder, files(1).name));

% Load the fitrange file
load('behavior_data/fitranges.mat');

Nsess = 30;
cols = paperaesthetics;
bluecol = cols.bluecol;
redcol = cols.redcol;
% bluecol = [8,81,156]/255;


%%
animals_ids = fields(fitparams_all);
effs_all = {};
slopes_all = {};
lapses_all = {};
offsets_all = {};

for i = 1:numel(animals_ids)
    [offsets, slopes, lapses, effs] = parse_params(fitparams_all.(animals_ids{i}));
    
    %Trim based on fitparams
    idx = contains(animals, animals_ids{i});
    assert(sum(idx) == 1);
    fitrange = ranges{idx} + 1;
    
    effs_all{i} = effs(fitrange);
    slopes_all{i} = slopes(fitrange);
    lapses_all{i} = lapses(fitrange);
    offsets_all{i} = offsets(fitrange);
end


%% Pad to same length
expeff_all = pad_to_same_length(effs_all);
expoffsets_all = -pad_to_same_length(offsets_all);
expoffsets_all(expoffsets_all > 20) = filterval;


expslopes_all = pad_to_same_length(slopes_all);
explapses_all = pad_to_same_length(lapses_all);

%% Find performance on the last session for each animal
for i = 1:size(expeff_all, 1)
    perf_array = expeff_all(i,:);
    perf_array = perf_array(~isnan(perf_array));
    fprintf("Animal: %s, perf: %.2f\n", animals_ids{i}, mean(perf_array(end-5:end)));

end



%% Eff
% plot(expeff_all', 'Color', [55,126,184]/255)
effmeans = nanmean(expeff_all, 1);
efferr = nanstd(expeff_all, [], 1) / sqrt(size(expeff_all, 1));

% Fit the eff mean
xvals = 1:Nsess;
yvals = effmeans(1:Nsess);
p = [1, 10, 0.2, 0.5];
pfit = fminsearch(@(p) losssigmoid(p, xvals, yvals), p);
ypred = pfit(4) + (1 - pfit(4) - pfit(3)) ./ (1 + exp(-pfit(1) * (xvals - pfit(2))));
ypred2 = pfit(4) + (1 - pfit(4) - p(3)) ./ (1 + exp(-p(1) * (xvals - p(2))));


% plot(expeff_all(:, 1:Nsess)', 'k')

%'perfect' agent simulation
blocklen_lims = 15:25;
Nblocks = 20;
Nreps = 100;
effs = perfect_agent_metrics(blocklen_lims, Nblocks, Nreps);

%% split performance by sex
animal_names_sex = {'e54', 'f11', 'f22', 'f03', 'f20', 'f01', 'e56', 'f12', 'fh03', ...
    'fh02', 'e57', 'f16', 'fh01', 'e46', 'f04', ...
    'f02', 'f23', 'e35', 'f21', ...
    'f17', 'e53'};

sex = {'F', 'F', 'M', 'F', 'M', 'M', 'M', 'F', 'M', ...
    'F', 'F', 'F', 'M', 'M', 'F', ...
    'F', 'M', 'F', 'M', ...
    'F', 'F'};

% parse the sex for each animal
animal_sex = {};
for i = 1:numel(animals_ids)
    animal_sex{i} = sex{find(strcmp(animal_names_sex, animals_ids{i}))};
end

perf_males = expeff_all(strcmp(animal_sex, 'M'), :);
perf_females = expeff_all(strcmp(animal_sex, 'F'), :);

figure('Position', [440,414,738,384]);
effmeans_males = nanmean(perf_males, 1);
efferr_males = nanstd(perf_males, [], 1); %/ sqrt(size(perf_males, 1));
effmeans_females = nanmean(perf_females, 1);
efferr_females = nanstd(perf_females, [], 1); % / sqrt(size(perf_females, 1));

gap = 0.1;
hold on
l1 = errorbar((1:Nsess) - gap, effmeans_females(1:Nsess), efferr_females(1:Nsess), 'o', ...,
    'MarkerFaceColor', cols.bluecol, 'MarkerEdgeColor', cols.bluecol, 'Color', cols.bluecol);
l2 = errorbar((1:Nsess) + gap, effmeans_males(1:Nsess), efferr_males(1:Nsess), 'o', ...,
    'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'Color', 'k');

ylim([0.3, 0.9])
mymakeaxis('x_label', 'Session', 'y_label', 'Performance')
lgd = legend({'Female', 'Male'});
lgd.FontSize = 16;

%% stats
male_perf_1 = [];
male_perf_2 = [];
male_perf_3 = [];
for i = 1:size(perf_males, 1) %iterate through each male animal
    male_perf_1(end+1) = nanmean(perf_males(i, 1:10));
    male_perf_2(end+1) = nanmean(perf_males(i, 11:20));
    male_perf_3(end+1) = nanmean(perf_males(i, 21:30));
end

female_perf_1 = [];
female_perf_2 = [];
female_perf_3 = [];
for i = 1:size(perf_females, 1) %iterate through each female animal
    female_perf_1(end+1) = nanmean(perf_females(i, 1:10));
    female_perf_2(end+1) = nanmean(perf_females(i, 11:20));
    female_perf_3(end+1) = nanmean(perf_females(i, 21:30));
end

%stats
p1 = ranksum(male_perf_1(~isnan(male_perf_1)), female_perf_1(~isnan(female_perf_1)));
p2 = ranksum(male_perf_2(~isnan(male_perf_2)), female_perf_2(~isnan(female_perf_2)));
p3 = ranksum(male_perf_3(~isnan(male_perf_3)), female_perf_3(~isnan(female_perf_3)));

%% plot the bar plots with average performance
figure('Position', [440,434,712,364]);

mean_male1 = nanmean(male_perf_1);
mean_male2 = nanmean(male_perf_2);
mean_male3 = nanmean(male_perf_3);

std_male1 = nanstd(male_perf_1);
std_male2 = nanstd(male_perf_2);
std_male3 = nanstd(male_perf_3);

mean_female1 = nanmean(female_perf_1);
mean_female2 = nanmean(female_perf_2);
mean_female3 = nanmean(female_perf_3);

std_female1 = nanstd(female_perf_1);
std_female2 = nanstd(female_perf_2);
std_female3 = nanstd(female_perf_3);

gap = 0.1;
h1 = bar([1, 2, 3] + gap, [mean_male1, mean_male2, mean_male3], 0.2, 'k', 'FaceAlpha', 0.6);
hold on
h2 = bar([1, 2, 3] - gap, [mean_female1, mean_female2, mean_female3], 0.2, 'b', 'FaceAlpha', 0.6);
errorbar([1, 2, 3] + gap, [mean_male1, mean_male2, mean_male3], ...
    [std_male1, std_male2, std_male3], 'o', 'Color', 'k', ...
    'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
errorbar([1, 2, 3] - gap, [mean_female1, mean_female2, mean_female3], ...
    [std_female1, std_female2, std_female3], 'o', 'Color', 'k', ...
    'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
legend([h1, h2], {'Male', 'Female'})


mymakeaxis('x_label', 'Session group', 'y_label', 'Mean performance', 'xticks', [1,2,3], ...
    'xticklabels', {'1-10', '11-20', '21-30'}, 'font_size', 20)


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



