% For official paper figures sent on 2.21.2022

%% classify and visualize state distributions
% paths = pathsetup('matchingsim');

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
    
    figure;
    imagesc(transmatPermuted)
    colormap gray
    caxis([0 1])
%     colorbar;
    axis xy
    mymakeaxis('x_label', 'State i', 'y_label', 'State i + 1', 'xticks', 1:K,...
        'xticklabels', labels, 'yticks', 1:K, 'yticklabels', labels, 'xytitle', animal_name);
    
    counter = counter + n_zstates;    
    
    % Break down into individual sessions
    assert(sum(lengths) == numel(zclassified));
    zclassified_splits = mat2cell(zclassified, 1, lengths);
    
    
    animalinfo(i).animal = animal_name;
    animalinfo(i).zclassified = zclassified_splits;
    animalinfo(i).classes = sort(statesFlat_extracted);
    
%     filename = fullfile(paths.figpath, 'hmmblockFigs/transmat_animals/',...
%         sprintf('%s_transmat_tsne_021322b_Nmodes.pdf', animal_name));
%     if ~exist(filename, 'file')
%         saveas(gcf, filename);
%     end
    
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


%% Plot identity of HMM modes per animal
hmmidentities = zeros(numel(animalinfo), 6);
namelst = {};
for i = 1:numel(animalinfo)
    K = animalModeInfo.K(i);
    assert(numel(animalinfo(i).classes) == K);
    hmmidentities(i, 1:K) = animalinfo(i).classes;
    namelst{i} = animalinfo(i).animal;
end

meanmode = mean(hmmidentities, 2);
Nmodes = sum(hmmidentities > 0, 2);
[~,idx] = sort(Nmodes);
namelst = namelst(idx);
hmmidentities = hmmidentities(idx,:);

% idx2 = 22 - [1,2,3,4,5,7,9,11,17,19,6,8,12,13,14,15,18,10,16,20,21];
% idx2 = idx2(end:-1:1);
% hmmidentities = hmmidentities(idx2,:);
% namelst = namelst(idx2);

cmap = brewermap(6, 'Set1');
cmap = cmap([2,1,5,6,4,3],:);
cmap = [0 0 0; cmap];

figure;
imagesc(hmmidentities);
colormap(cmap);
hold on
vline((0:6) + 0.5, 'k');
hline((0:numel(animalinfo)) + 0.5, 'k');

axis xy
mymakeaxis('x_label', 'HMM mode', 'y_label', 'Animal', ...
    'font_size', 22, 'xticks', 1:6, 'yticks', 1:numel(namelst), 'yticklabels', namelst)



%% Plot state evolution profile for each animal
opts.plotting = 1;
f = waitbar(0);
compositions = {};
Lwindow = 30;
for id = 1:numel(animalinfo)
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

    if opts.plotting
        % Plot bar graph to show the composition
        figure('Position', [440,423,736,375], 'Name', animalinfo(id).animal);
        h = bar(composition,'stacked');
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
end
close(f)


%% Save if requested
opts.save = 0;
savefilename = fullfile(opts.rootdir, sprintf('hmmblock_composition_info_%s_v4_021322.mat', version));

if opts.save && ~exist(savefilename, 'file')
    save(savefilename, 'opts', 'animalinfo');
    fprintf('File saved!\n');
else
    fprintf('Skipping save...\n');   
end



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


