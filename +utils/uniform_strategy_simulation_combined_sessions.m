function out = uniform_strategy_simulation_combined_sessions(animal, session_ids, opts, metricid)
% Runs simulations of a uniform strategy
% and finds the p-value of the lapse parameter
% metricid: 1-4, 1=offset, 2=slope, 3=lapse, 4=perf
% session_ids: list of session id's of interest

Nwindow = opts.Nwindow;
Nsim = opts.Nsim;
sesslengths = [];

% find the pfits for the observed behavior
raw_all = [];
pfits = [];

for i = 1:numel(session_ids)
    sessid = session_ids(i);
    [~, raw] = utils.get_perf_animal_from_session(animal, sessid, Nwindow);
    if isnan(raw)
        break
    end
    raw_all = [raw_all; raw];
    sesslengths(i) = size(raw, 1);
end

perf = nanmean(raw_all, 1);
perf = perf(~isnan(perf));

    
for j = 1:size(raw_all, 1)
    feedbacks = raw_all(j, :);
    feedbacks = feedbacks(~isnan(feedbacks));
    p = mathfuncs.fit_sigmoid_logistic(feedbacks);
    eff = sum(feedbacks) / numel(feedbacks);
    pfits(j, :) = [p eff];
end

% Fit the average behavior to a sigmoid
p = mathfuncs.fit_sigmoid(perf);
% xvals = 1:sum(~isnan(perf));
xvals = 1:size(raw_all, 2);
y = mathfuncs.sigmoid(xvals, p(1), p(2), p(3));

% Generate behavior according to latent rate y
Nblocks = size(raw_all, 1);
p_lapse_all = nan(Nblocks, Nsim);

for i = 1:Nsim
    disp(i)
    p_gen = utils.generate_behavior_and_fit2(y, Nblocks, raw_all);
    p_lapse_all(:,i) = p_gen(:,metricid);
end

% Get the std and the distribution to derive the p-value
stdarr = [];
for i = 1:size(p_lapse_all, 2)
    pvals = p_lapse_all(:, i);
    stdarr(i) = nanstd(pvals);
end

if sum(~isnan(stdarr)) == 0
    out.pval = -1;
else
    out.pval = sum(stdarr > nanstd(pfits(:,metricid))) / Nsim;
end

out.stdarr = stdarr;
out.p_lapse_all = p_lapse_all;

out.pfits = pfits;
out.perf = perf;
out.y = y;
out.sesslengths = sesslengths;


end