filterval = 20; %filter values such that expoffsets_all(expoffsets_all > 20) = filterval;
files = dir('behavior_data/fitparams_session_averaged.mat');
assert(numel(files) == 1)
load(fullfile(files(1).folder, files(1).name), 'fitparams_all')
animals_ids = fields(fitparams_all);

%% determine last 5 sessions per animal
rootdir = 'behavior_data';
load('behavior_data/fitranges.mat', 'ranges', 'animals');

num_sessions = zeros(1, numel(animals_ids)) - 1;
for animal_id = 1:numel(animals_ids)
    animal = animals_ids{animal_id};
    fname = sprintf('%s_all_sessions.mat', animal);
    load(fullfile(rootdir, fname), 'feedbacks_cell', 'targets_cell');
    
    % Determine corresponding fit range
    idx = contains(animals, animal);
    assert(sum(idx) == 1);
    fitrange = ranges{idx} + 1;

    fprintf('Animal: %s, num sessions = %d\n', animal, numel(fitrange));
    num_sessions(animal_id) = numel(fitrange);
end



%%
exclude_animals = {};
opts.Nsim = 100;
opts.Nwindow = 25;
opts.rng = 123;
metricid = 4;

for i = 1:numel(animals_ids)
    animal = animals_ids{i};

    sessid_lst = (num_sessions(i) - 4) : num_sessions(i);

    if contains(animal, exclude_animals)
        continue
    end
    fprintf('Animal: %s\n', animal)
        
    rng(opts.rng)
    plst = [];
    
    fprintf('## Fitting session %d\n', sessid_lst)
%     try
    out = utils.uniform_strategy_simulation_combined_sessions(animal, sessid_lst, opts, metricid);
    plst(end+1) = out.pval;
%     catch
%         continue;
%     end
    
    save(sprintf('uniform_test_sim_results_eff/%s.mat', animal), ...
        'sessid_lst', 'plst', 'opts', 'animal', 'metricid', 'out');

end


%% For plotting e54 blockwise parameters
load('uniform_test_sim_results_eff/e54.mat');

p = mathfuncs.fit_sigmoid_asym(out.perf(1:15));
results = mathfuncs.sigmoid_asym(1:25, p(1), p(2), p(3), p(4));

% generate the behavior
Nblocks = sum(out.sesslengths);
choices = rand(Nblocks, 25) < results;

pfit_simulations = [];
for i = 1:Nblocks
    blocklength = randsample(15:25, 1);
    pfit_simulations(i, 1:3) = mathfuncs.fit_sigmoid_logistic(choices(i, 1:blocklength));
    pfit_simulations(i, 4) = mean(choices(i, 1:blocklength));
end

mean_pfit_simulations = mean(pfit_simulations);
cov_pfit_simulations = sqrt(cov(pfit_simulations));

figure('Position', [440,124,289,674]);
sesslengths = cumsum(out.sesslengths);
sesslengths = [1 (sesslengths + 1)];
custom_cmap = getPyPlot_cMap('Blues');
for j = 6:10
    i = j - 5;
    subplot(5, 1, i);
    startid = sesslengths(i);
    endid = sesslengths(i + 1) - 1;
    fprintf('start = %d, end = %d\n', startid, endid)
    scatter(out.pfits(startid:endid, 3), out.pfits(startid:endid, 4), ...
        [], 1:out.sesslengths(i), 'filled');
    hold on
    errorbar(mean_pfit_simulations(3), mean_pfit_simulations(4), cov_pfit_simulations(4, 4), cov_pfit_simulations(4,4), ...
        cov_pfit_simulations(3, 3), cov_pfit_simulations(3, 3));
    colormap(gca, custom_cmap)
    xlim([0, 0.5])
    ylim([0.3, 1.1])
    
    if i == 5
        mymakeaxis('yticks', [0.5, 1], 'xticks', [0, 0.25, 0.5], 'x_label', 'Lapse',...
            'y_label', 'Performance', 'xytitle', sprintf('Day %d', i + 25))
    else
        mymakeaxis('yticks', [0.5, 1], 'xticks', [0, 0.25, 0.5], 'y_label', 'Performance', ...
            'xytitle', sprintf('Day %d', i + 25))
    end


end



