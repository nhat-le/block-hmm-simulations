% load('blockhmm_validation_050823.mat')

%% Averaging the two seeds..
load('blockhmm_validation_021222.mat')
animal_lst2 = animal_lst;
ll_lst_all2 = ll_lst_all;
aic_lst2 = aic_lst;

load('blockhmm_validation_021322.mat')

assert(all(all(animal_lst == animal_lst2)))

ll_average = (ll_lst_all + ll_lst_all2) / 2;
aic_average = (aic_lst + aic_lst2) / 2;
K_lst = [6 5 3 6 4 6 3 6 5 2 5 2 6 6 6 6 4 3 4 3 6];
% Plot the average for all animals
figure('Position', [288,186,932,448]);

% sort animal names
sorted_names = sort(cellstr(animal_lst));

for id_sort = 1:21
    % find where the original id lives
    i = find(strcmp(cellstr(animal_lst), sorted_names{id_sort}));
    disp(i)
    subplot(3, 7, id_sort)
    plot(ll_average(i, 1:6))
    
%     title(upper(animal_lst(i,:)))
    ylim([-0.5, 4])
    vline(K_lst(i))
    mymakeaxis('y_label', 'c.v. LL', 'xytitle', upper(animal_lst(i,:)), ...
        'xticks', 1:6, 'yticks', [0, 2, 4])
%     xticks(1:6)
end






