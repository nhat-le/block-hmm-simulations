% For official paper figures sent on 2.21.2022

%% classify and visualize state distributions
% paths = pathsetup('matchingsim');
load('single_animal_performance_by_session.mat')
% averaging two seeds
animalModeInfo = struct;
animalModeInfo.animals = {'fh03',...
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
    'e54'}; %K = 6

animalModeInfo.K = [6 5 3 6 4 6 3 6 5 2 5 2 6 6 6 6 4 3 4 3 6];

% Parse the animal mode files
folders = struct;
animalinfo = struct;

alltransfuncs = [];
allregimes = [];

for i = 1:numel(animalModeInfo.K)
    % load file
    K = animalModeInfo.K(i);
    version = sprintf('121821bK%d', K);
    opts.rootdir = fullfile('animal_blockHMM', version);

    if animalModeInfo.K(i) == 6
        classification_info_files = dir(fullfile(opts.rootdir, '*classification_info*_v4.mat'));
    else
        classification_info_files = dir(fullfile(opts.rootdir, '*classification_info*_v1.mat'));
    end
    assert(numel(classification_info_files) == 1);
    load(fullfile(classification_info_files(1).folder, classification_info_files(1).name),...
        'blockhmm_idx', 'statesFlat', 'folders', 'aggparams_native');

    
    % Load the individual animal blockfit file
    selection = contains({folders.name}, animalModeInfo.animals{i});
    
    assert(sum(selection) == 1) %make sure one and only one animal blockfit file found
    % NOTE: Changed this line 1.5.23
    curr_folder = strrep(folders(selection).folder, 'Dropbox (MIT)', 'Documents');
    load(fullfile(curr_folder, folders(selection).name), 'zstates', 'params', 'lengths', 'transmat');
%     load(fullfile(folders(selection).folder, folders(selection).name), 'zstates', 'params', 'lengths', 'transmat');
    parts = strsplit(folders(selection).name, '_');
    animal_name = parts{1};
    
    assert(strcmp(animal_name, animalModeInfo.animals{i}));
    
    
    zclassified = zstates;
    
    n_zstates = max(zstates) + 1;
    assert(n_zstates == K);
    assert(n_zstates == size(params, 2))
    
    % make sure the states are in the right order
    assert(sum(sum(aggparams_native{selection}(1:3,:) ~= params)) == 0);
    
    counter = K * (find(selection) - 1) + 1;
    statesFlat_extracted = statesFlat(counter : counter + n_zstates - 1);
    for istate = 1:n_zstates
        param = params(:, istate);
        transfunc = mathfuncs.sigmoid(1:15, param(1), param(2), param(3));
        alltransfuncs(end+1,:) = transfunc;
        allregimes(end+1) = statesFlat_extracted(istate);
        
        zclassified(zstates == istate - 1) = statesFlat_extracted(istate);
    end
    
    [identityPermuted, stateSortIdx] = sort(statesFlat_extracted);
    transmatPermuted = transmat(stateSortIdx,stateSortIdx);
    
    labels = {};
    for j = 1:numel(identityPermuted)
        if identityPermuted(j) <= 4
            labels{j} = ['Q', num2str(identityPermuted(j))];    
        else
            labels{j} = ['IB', num2str(identityPermuted(j))];
        end
    end
    
%     figure;
%     imagesc(transmatPermuted)
%     colormap gray
% 
%     
% 
%  
%     caxis([0 1])
% %     colorbar;
%     axis xy
%     mymakeaxis('x_label', 'State i', 'y_label', 'State i + 1', 'xticks', 1:K,...
%         'xticklabels', labels, 'yticks', 1:K, 'yticklabels', labels, 'xytitle', animal_name);
% 
%     % Display numerical values in the squares of true_transition_mat
%     for i_mat = 1:size(transmatPermuted, 1)
%         for j_mat = 1:size(transmatPermuted, 1)
%             textHandles = text(j_mat, i_mat, num2str(transmatPermuted(i_mat, j_mat), '%0.2f'), ...
%                 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle', 'Color', 'w');
%             
% %             set(textHandles, 'Color', 'w'); 
%             
%         end
%     end

%     filename = fullfile(sprintf('/Users/minhnhatle/Documents/block-hmm-simulations/figs/animal_transmat/transmat_%s.pdf', animal_name));
%     if ~exist(filename, 'file')
%         saveas(gcf, filename);
%     end
    
    counter = counter + n_zstates;    
    
    % Break down into individual sessions
    assert(sum(lengths) == numel(zclassified));
    zclassified_splits = mat2cell(zclassified, 1, lengths);
    
    
    animalinfo(i).animal = animal_name;
    animalinfo(i).zclassified = zclassified_splits;
    animalinfo(i).classes = sort(statesFlat_extracted);
    
    
    
end

%% Plot the grouped transition functions
for i = 1:6   
    yvals_sub = alltransfuncs(allregimes == i, :);
    
    
    subplot(2,3,i);
    plot(yvals_sub', 'k', 'LineWidth', 0.25)
    hold on
    plot(mean(yvals_sub, 1), 'r', 'Color', 'r', 'LineWidth', 2);
    ylim([0, 1])
    mymakeaxis('x_label', 'Trials in block;', 'y_label', 'P(Correct)', ...
        'yticks', 0:0.2:1, 'xticks', 0:5:15);


end


%% Plot state evolution profile for each animal
opts.plotting = 0;
f = waitbar(0);
compositions = {};
Lwindow = 30;
for id = 1 :numel(animalinfo)
    waitbar(id / numel(animalinfo), f);
    zclassified = animalinfo(id).zclassified;
    nstates_regimes = 6; %numel(animalinfo(id).classes);
    composition = nan(numel(zclassified), nstates_regimes);
    for i = 1:numel(zclassified)
        z_single_session = zclassified{i};
        for j = 1:nstates_regimes
            composition(i,j) = sum(z_single_session == j) / numel(z_single_session);   
        end
    end
    
    animalinfo(id).composition = composition;
    nsess = size(composition, 1);
    xlimmax = min(nsess, Lwindow);

    original_id = find(strcmp(animals_ids, animalinfo(id).animal));
    fprintf('Animal: %s, xlimmax: %d, numsess = %d\n', animalinfo(id).animal, ...
        xlimmax, sum(~isnan(expeff_all(original_id,:))))

    if opts.plotting
        % Plot bar graph to show the composition
        figure('Position', [440,423,736,375], 'Name', animalinfo(id).animal);
        h = bar(composition,'stacked');
        hold on
        plot(expeff_all(original_id, :), 'k', 'LineWidth', 2)
        cols = paperaesthetics;
        colors = cols.colors;
%         colors = brewermap(6, 'Set1');
%         orders = [2, 1, 5, 4, 3];
        for i = 1:nstates_regimes
            h(i).FaceColor = colors(i,:);
            h(i).ShowBaseLine = 'off';
        end
        xlim([0.5 xlimmax + 0.5])
        ylim([0 1])
        mymakeaxis('x_label', 'Session #', 'y_label', 'Fraction', 'xticks', 0:5:xlimmax)
    end
    compositions{id} = composition;
    saveas(gcf, sprintf('figs/single_perf_composition/%s.pdf', animalinfo(id).animal));
end
close(f)



%% Parse and average hmm summary fractions
extracted_all = {};
cols = paperaesthetics;
colors = cols.colors;

Lwindow = 30;

for i = 1:numel(animalinfo)
    animalname = animalinfo(i).animal;
    filename = sprintf('%s_hmmblockfit_*%s.mat', animalname, version);
    blockfitfile = dir(fullfile('animal_blockHMM', version, filename));
    assert(numel(blockfitfile) == 1);
    load(fullfile(blockfitfile(1).folder, blockfitfile(1).name), 'fitrange');
    
    
    trialidx = 1:min(Lwindow, size(animalinfo(i).composition, 1));
    
    extracted = animalinfo(i).composition(trialidx,:);
    
    extracted_all{i} = extracted;
    
    
    
end

% extracted_all([5, 9, 10]) = [];

paperaesthetics;
% colors = brewermap(6, 'Set1');
% colors = colors([1,2,3,4,5,6],:);
% colors(4,:) = [0,0,0];

figure('Position', [440,379,731,419]);
hold on
lines = [];
for i =1:6
    sline = pad_to_same_length(extracted_all, i);
    disp(nanmean(sline(:, 30)))
    N = sum(~isnan(sline));
    if i == 4
        h = errorbar(1:size(sline, 2), nanmean(sline, 1), nanstd(sline, [], 1) ./ sqrt(N), ...
            'o-', 'Color', 'k', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'k');
    else
        h = errorbar(1:size(sline, 2), nanmean(sline, 1), nanstd(sline, [], 1) ./ sqrt(N), ...
            'o-', 'Color', colors(i,:), 'MarkerFaceColor', colors(i,:));
    end
    lines(i) = h;
%     xlim([1, 40])
    ylim([0, 1])
end

mymakeaxis('x_label', 'Session', 'y_label', 'Fraction', 'xticks', 0:5:Lwindow, 'font_size', 22)
l = legend(lines, {'Q1', 'Q2', 'Q3', 'Q4', 'IB5', 'IB6'});
l.Title.String = 'Regime';
l.Title.FontSize = 12;
l.FontSize = 12;

%% plot evolution for male vs female
animal_names_sex = {'e54', 'f11', 'f22', 'f03', 'f20', 'f01', 'e56', 'f12', 'fh03', ...
    'fh02', 'e57', 'f16', 'fh01', 'e46', 'f04', ...
    'f02', 'f23', 'e35', 'f21', ...
    'f17', 'e53'};

sex = {'F', 'F', 'M', 'F', 'M', 'M', 'M', 'F', 'M', ...
    'F', 'F', 'F', 'M', 'M', 'F', ...
    'F', 'M', 'F', 'M', ...
    'F', 'F'};

extracted_all_male = {};
extracted_all_female = {};
for i = 1:numel(animalinfo)
    name = animalinfo(i).animal;
    names_id = find(strcmp(animal_names_sex, name));
    if strcmp(sex{names_id}, 'F')
        extracted_all_female{end+1} = extracted_all{i};
    elseif strcmp(sex{names_id}, 'M')
        extracted_all_male{end+1} = extracted_all{i};
    else
        error('unknown value')
    end
end

% female plot

figure('Position', [440,379,731,419]);
hold on
lines = [];
for i =1:6
    sline = pad_to_same_length(extracted_all_female, i);
    disp(nanmean(sline(:, 30)))
    N = sum(~isnan(sline));
    if i == 4
        h = errorbar(1:size(sline, 2), nanmean(sline, 1), nanstd(sline, [], 1) ./ sqrt(N), ...
            'o-', 'Color', 'k', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'k');
    else
        h = errorbar(1:size(sline, 2), nanmean(sline, 1), nanstd(sline, [], 1) ./ sqrt(N), ...
            'o-', 'Color', colors(i,:), 'MarkerFaceColor', colors(i,:));
    end
    lines(i) = h;
%     xlim([1, 40])
    ylim([0, 1])
end

mymakeaxis('x_label', 'Session', 'y_label', 'Fraction', 'xticks', 0:5:Lwindow, 'font_size', 22)
l = legend(lines, {'Q1', 'Q2', 'Q3', 'Q4', 'IB5', 'IB6'});
l.Title.String = 'Regime';
l.Title.FontSize = 12;
l.FontSize = 12;

%% male plot
% female plot

figure('Position', [440,379,731,419]);
hold on
lines = [];
for i =1:6
    sline = pad_to_same_length(extracted_all_male, i);
    disp(nanmean(sline(:, 30)))
    N = sum(~isnan(sline));
    if i == 4
        h = errorbar(1:size(sline, 2), nanmean(sline, 1), nanstd(sline, [], 1) ./ sqrt(N), ...
            'o-', 'Color', 'k', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'k');
    else
        h = errorbar(1:size(sline, 2), nanmean(sline, 1), nanstd(sline, [], 1) ./ sqrt(N), ...
            'o-', 'Color', colors(i,:), 'MarkerFaceColor', colors(i,:));
    end
    lines(i) = h;
%     xlim([1, 40])
    ylim([0, 1])
end

mymakeaxis('x_label', 'Session', 'y_label', 'Fraction', 'xticks', 0:5:Lwindow, 'font_size', 22)
l = legend(lines, {'Q1', 'Q2', 'Q3', 'Q4', 'IB5', 'IB6'});
l.Title.String = 'Regime';
l.Title.FontSize = 12;
l.FontSize = 12;



%% summarize and do stats
p1lst = [];
p2lst = [];
p3lst = [];
diff1lst = [];
diff2lst = [];
diff3lst = [];

mean_male = zeros(6, 3);
mean_female = zeros(6, 3);
std_male = zeros(6, 3);
std_female = zeros(6, 3);


for mode_id = 1:6
    male_perf_1 = [];
    male_perf_2 = [];
    male_perf_3 = [];

    
    for mouse_id = 1:numel(extracted_all_male)
        raw_perf = extracted_all_male{mouse_id}(:, mode_id);
        padded_perf = nan(1, 30);
        padded_perf(1:numel(raw_perf)) = raw_perf;
        male_perf_1(end+1) = nanmean(padded_perf(1:10));
        male_perf_2(end+1) = nanmean(padded_perf(11:20));
        male_perf_3(end+1) = nanmean(padded_perf(21:30));
    end

    female_perf_1 = [];
    female_perf_2 = [];
    female_perf_3 = [];

    for mouse_id = 1:numel(extracted_all_female)
        raw_perf = extracted_all_female{mouse_id}(:, mode_id);
        padded_perf = nan(1, 30);
        padded_perf(1:numel(raw_perf)) = raw_perf;
        female_perf_1(end+1) = nanmean(padded_perf(1:10));
        female_perf_2(end+1) = nanmean(padded_perf(11:20));
        female_perf_3(end+1) = nanmean(padded_perf(21:30));
    end
    
    mean_male(mode_id, 1) = nanmean(male_perf_1);
    mean_male(mode_id, 2) = nanmean(male_perf_2);
    mean_male(mode_id, 3) = nanmean(male_perf_3);

    mean_female(mode_id, 1) = nanmean(female_perf_1);
    mean_female(mode_id, 2) = nanmean(female_perf_2);
    mean_female(mode_id, 3) = nanmean(female_perf_3);

    std_male(mode_id, 1) = nanstd(male_perf_1);
    std_male(mode_id, 2) = nanstd(male_perf_2);
    std_male(mode_id, 3) = nanstd(male_perf_3);

    std_female(mode_id, 1) = nanstd(female_perf_1);
    std_female(mode_id, 2) = nanstd(female_perf_2);
    std_female(mode_id, 3) = nanstd(female_perf_3);

    p1lst(end+1) = ranksum(male_perf_1(~isnan(male_perf_1)), female_perf_1(~isnan(female_perf_1)));
    p2lst(end+1) = ranksum(male_perf_2(~isnan(male_perf_2)), female_perf_2(~isnan(female_perf_2)));
    p3lst(end+1) = ranksum(male_perf_3(~isnan(male_perf_3)), female_perf_3(~isnan(female_perf_3)));

    diff1lst(end+1) = nanmean(male_perf_1) - nanmean(female_perf_1);
    diff2lst(end+1) = nanmean(male_perf_2) - nanmean(female_perf_2);
    diff3lst(end+1) = nanmean(male_perf_3) - nanmean(female_perf_3);
end



%% Bar plot
figure('Position', [440,80,493,718]);
hold on
gap = 0.1;

for iplot = 1:3
    subplot(3,1,iplot)
    h1 = bar((1:6) + gap, mean_male(:, iplot), 0.2, 'k', 'FaceAlpha', 0.6);
    hold on
    h2 = bar((1:6) - gap, mean_female(:, iplot), 0.2, 'b', 'FaceAlpha', 0.6);
    errorbar((1:6) + gap, mean_male(:, iplot), ...
        std_male(:, iplot), 'o', 'Color', 'k', ...
        'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
    errorbar((1:6) - gap, mean_female(:, iplot), ...
        std_female(:, iplot), 'o', 'Color', 'k', ...
        'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
    ylim([0, 1])

    mymakeaxis('xticks', [1,2,3,4,5,6], ...
        'xticklabels', {'Q1', 'Q2', 'Q3', 'Q4', 'IB5', 'IB6'}, ...
        'font_size', 20, 'yticks', [0, 0.5, 1])
%     legend([h1, h2], {'Male', 'Female'})
end





