%% plot expected vs observed standard deviations
% files = dir('/Users/minhnhatle/Documents/Sur/MatchingSimulations/PaperFigures/code/expfit/uniform_test_sim_results3_eff/123122/*.mat');
files = dir('uniform_test_sim_results_eff/*.mat');
stdarr_all = [];
metricid = 4;
std_observed_all = [];
names = {};

out = [];
for i = 1:21
    load(fullfile(files(i).folder, files(i).name));
    if ~isfield(out, 'stdarr')
        load(fullfile(files(1).folder, files(1).name));
        fprintf("Dummy file loaded for i = %d: %s\n", i, files(i).name);
    end
    stdarr_all(:,i) = out.stdarr';
    std_observed_all(i) = nanstd(out.pfits(:, metricid));
    
    animal_name = strsplit(files(i).name, '.');
    names{i} = animal_name{1};
    clear out

end

figure('Position', [440,13,273,785]);

boxplot(stdarr_all, 'orientation','horizontal')
hold on
for i = 1:21
    plot([std_observed_all(i), std_observed_all(i)], [i - 0.3 i + 0.3], 'k', ...
        'LineWidth', 3)
end

yticks(1:21)
yticklabels(names)
xlim([0.0, 0.4])

mymakeaxis('x_label', 'Expected/observed variability', 'yticks', 1:21, 'yticklabels', names)



%% P-values
for i = 1:21
    pvals(i) = sum(stdarr_all(:, i) > std_observed_all(i)) / 100;

end

