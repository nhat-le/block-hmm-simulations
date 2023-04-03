
load('classification_out_result.mat')

%% Parse the features


opts.perf_thres1 = 0.65;
opts.perf_thres2 = 0.84;
opts.lapse_thres = 0.05;

class_by_perf = [];
for i = 1:size(out.features, 1)
    if out.features(i, 1) < opts.perf_thres1
        class_by_perf(i) = 1;
    elseif out.features(i, 1) < opts.perf_thres2
        if out.features(i, 2) < 0.1
            class_by_perf(i) = 2;
        else
            class_by_perf(i) = 3;
        end
    else
        class_by_perf(i) = 4;
    end
end

%% Plot the correspondence
hmmidentities_flat = out.idx;
hmmidentities_bflat = class_by_perf;

breakdowns = {};
breakdowns{1} = hmmidentities_flat(hmmidentities_bflat == 1);
breakdowns{2} = hmmidentities_flat(hmmidentities_bflat == 2);
breakdowns{3} = hmmidentities_flat(hmmidentities_bflat == 3);
breakdowns{4} = hmmidentities_flat(hmmidentities_bflat == 4);

compositions = [];
for i = 1:4
    compositions(i,:) = parse_composition(breakdowns{i});

end

% stacked bar charts
figure;
h = barh((compositions ./ sum(compositions, 1))', 'stacked');
cmap = brewermap(6, 'Set2');
cmap = cmap([3, 4, 6, 1],:);

for i = 1:4
    h(i).FaceColor = cmap(i,:);
    h(i).ShowBaseLine = 'off';
end

mymakeaxis('x_label', '% modes', 'font_size', 25, ...
    'yticks', 1:6, 'yticklabels', {'Q1', 'Q2', 'Q3', 'Q4', 'IB5', 'IB6'});




function composition = parse_composition(arr)

composition = [];
for i = 1:6
    composition(i) = sum(arr == i) / numel(arr);
end


end
