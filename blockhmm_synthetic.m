load('blockhmm_simulated.mat');


%% Plot the transition functions
figure;
subplot(231)
imagesc(obs_clustered{1}(:, 1:30))
colormap('redblue')
axis xy
mymakeaxis('x_label', 'Trials in block', 'y_label', 'Block #', 'xytitle', 'Mode 1',...
    'xticks', 0:10:30, 'yticks', 100:100:300)

subplot(232)
imagesc(obs_clustered{2}(:, 1:30))
colormap('redblue')
axis xy
mymakeaxis('x_label', 'Trials in block', 'y_label', 'Block #', 'xytitle', 'Mode 2',...
    'xticks', 0:10:30, 'yticks', 100:100:400)


subplot(233)
imagesc(obs_clustered{3}(:, 1:30))
colormap('redblue')
axis xy
mymakeaxis('x_label', 'Trials in block', 'y_label', 'Block #', 'xytitle', 'Mode 3',...
    'xticks', 0:10:30, 'yticks', 50:50:150)


subplot(234)
plot(transfuncs(1,:), 'k', 'LineWidth', 1)
hold on
plot(mean(obs_clustered{1}), 'r', 'LineWidth', 1)
ylim([0 1])
mymakeaxis('x_label', 'Trials in block', 'y_label', 'P (Correct)', 'yticks', 0:0.5:1,...
    'xticks', 0:10:30)


subplot(235)
plot(transfuncs(2,:), 'k', 'LineWidth', 1)
hold on
plot(mean(obs_clustered{2}), 'r', 'LineWidth', 1)
ylim([0 1])
mymakeaxis('x_label', 'Trials in block', 'y_label', 'P (Correct)', 'yticks', 0:0.5:1,...
    'xticks', 0:10:30)



subplot(236)
l1 = plot(transfuncs(3,:), 'k', 'LineWidth', 1);
hold on
l2 = plot(mean(obs_clustered{3}), 'r', 'LineWidth', 1);
ylim([0 1])
mymakeaxis('x_label', 'Trials in block', 'y_label', 'P (Correct)', 'yticks', 0:0.5:1,...
    'xticks', 0:10:30)
legend([l1, l2], {'Observed mean', 'Model fit'}, 'FontName', 'helvetica',...
    'FontSize', 12, 'FontAngle', 'italic', 'Position', [0.718219194894452,0.478039938556068,0.225,0.079761904761905])

colormap([229, 75, 34; 171, 209, 255]/255);


%% loss function with training
figure;
plot(hmm_lls, 'LineWidth', 2)
hold on
plot([0 3000], [true_ll true_ll], 'k--');

mymakeaxis('x_label', 'Iteration #', 'y_label', 'Log likelihood ( x 1e4)')



%% Parameter comparison
figure('Position', [313,326,654,333]);
colors = brewermap(3, 'Set1');
colors = colors([2, 1, 3], :);
titles = {'Offset s', 'Slope \alpha', 'Lapse \epsilon'};
for i = 1:3
    subplot(1,3,i)
    handles = [];
    count = 1;
    for j = [3, 1, 2]
%         plot([true_params(i,j); sim_params(i,j)], 'Color', colors(j,:))
        hold on
        h = plot([true_params(i,j); sim_params(i,j)], 'o-', 'Color', colors(count,:),...
            'MarkerFaceColor', colors(count,:));
        handles(end+1) = h;
        count = count + 1;
        
    end
    xlim([0.5 2.5])
    mymakeaxis('xytitle', titles{i}, 'xticks', 1:2, 'xticklabels', ...
        {'True', 'Model'})
    
    if i == 3
        legend(handles, {'Mode 1', 'Mode 2', 'Mode 3'}, 'FontSize', 12,...
            'FontName', 'helvetica', 'FontAngle', 'italic', 'Position', [0.80909949196741,0.67301050093294,0.125382262996942,0.145645645645646])
    end
end


%% State segmentation
figure('Position', [113,197,785,473]);
ax1 = subplot(4, 1, [1,2]);
imagesc(obs')
% axis off
title('Observations')
set(gca, 'FontSize', 16)
xticks([])
ylabel('Trials in block')

ax2 = subplot(413);
imagesc(true_states)
axis off
title('True states')
set(gca, 'FontSize', 16)


ax3 = subplot(414);
% do the permutation
zstates_copy = most_likely_states;
zstates_copy(most_likely_states == 0) = 2;
zstates_copy(most_likely_states == 1) = 0;
zstates_copy(most_likely_states == 2) = 1;
imagesc(zstates_copy)
rdbumap = brewermap(5, 'RdBu');
% colormap(ax1, rdbumap([1 3], :))
colormap(ax1, [229, 75, 34; 171, 209, 255]/255);

% colormap(ax1, [239, 201, 175;31, 138, 192] / 255); 
% colormap(ax1, [226, 60, 82; 189, 255, 246] / 255);
cmap = brewermap(6, 'Paired');
cmap = cmap(1:2:end,:);
cmap = cmap([3,2,1],:);
colormap(ax2, cmap)
colormap(ax3, cmap)
title('Inferred states')
xlabel('Block #')
yticks([])
set(gca, 'FontSize', 16)

% axis off

%% Transition matrices
figure('Position', [405,479,743,319]);
subplot(121)
imagesc(true_transition_mat)
axis xy

mymakeaxis('x_label', 'State i', 'y_label', 'State i+1',...
    'xticks', 1:3, 'yticks', 1:3, 'xytitle', 'True transition matrix')

colorbar('Position', [0.5,0.38,0.02,0.5], 'FontSize', 12, 'Ticks', 0:0.2:1, 'Limits', 0:1)


subplot(122)
imagesc(learned_transition_mat)
axis xy
mymakeaxis('x_label', 'State i', 'y_label', 'State i+1',...
    'xticks', 1:3, 'yticks', 1:3, 'xytitle', 'Learned transition matrix')
% colorbar('Position', [0.492897081950026,0.113207542708281,0.022598870056497,0.814016176968269])
colorbar('Position', [0.93,0.38,0.02,0.5], 'FontSize', 12, 'Ticks', 0:0.2:1, 'Limits', 0:1)
colormap gray

%% Validation and K-selection
load('blockhmm_synthetic_K_validation.mat');
figure;
scatter(nstates_lst, ll_lst, 80, 'bo', 'filled');
hold on
plot(nstates_lst, ll_lst, 'b', 'LineWidth', 2);

ylim([0 3])
vline(3, 'k--')
mymakeaxis('x_label', 'K', 'y_label', 'Normalized test log likelihood', 'xticks', ...
    nstates_lst, 'yticks', 0:0.5:3)

