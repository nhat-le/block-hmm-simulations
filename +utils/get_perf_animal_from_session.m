function [perf, raw] = get_perf_animal_from_session(animal, sessid, plot_window)
% animal: str, animal to plot
% sessid: int, id of the session
info = utils.get_animal_raw_data(animal, sessid);

if ~isstruct(info)
    perf = nan;
    raw = nan;
    warning('Session does not exist')
else
    [perf, raw] = utils.get_session_transition_curve(info.feedbacks, ...
        info.targets, plot_window);
    
    % all the nan's of perf should be at the end
    assert(sum(isnan(perf)) == 0 || ...
        find(isnan(perf), 1, 'first') > find(~isnan(perf), 1, 'last'))

end
