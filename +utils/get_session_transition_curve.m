function [perf, sess_average] = get_session_transition_curve(feedbacks, targets, plot_window)
% Inputs: fb: array of fb, bool of session
% targets: array of targets
% plot_window: how many trials per block to extract?
% Returns the transition curve of the session
% sess_average: performance of individual blocks
blockstarts = find(diff(targets)) + 1;
blockstarts = [1 blockstarts numel(targets) + 1];

sess_average = nan(numel(blockstarts) - 1, plot_window);

for i = 1:numel(blockstarts) - 1
    startidx = blockstarts(i);
    endidx = min(blockstarts(i + 1) - 1, blockstarts(i) + plot_window - 1);    
    N = endidx - startidx + 1;
    sess_average(i, 1:N) = feedbacks(startidx:endidx);
end

perf = nanmean(sess_average);

end
