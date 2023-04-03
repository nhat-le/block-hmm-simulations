function p_gen = generate_behavior_and_fit2(perf, Nblocks, raw)
% Generate a uniform behavior according to perf, for Nblocks
% Then fit the behavior in each block with a sigmoidal function
% raw: nblocks x ntrials array of performance 0/1

y_gen = double(rand(Nblocks, numel(perf)) < perf);
y_gen(isnan(raw)) = nan;

p_gen = [];
for i = 1:size(y_gen, 1)
    feedbacks = y_gen(i, :);
    feedbacks = feedbacks(~isnan(feedbacks));
    p = mathfuncs.fit_sigmoid_logistic(feedbacks);
    eff = sum(feedbacks) / numel(feedbacks);
    p_gen(i, :) = [p eff];
end


end