function offsets = random_agent_metrics(blocklen_lims, Nblocks, Nreps)
% blocklen_lims: block lengths to sample from
% Nblocks: number of blocks in the session
% Nreps: number of reps to simulate
% Simulates the performance and metrics of a "random" agent
% that selects actions at random
rng(123)
offsets = [];
for i = 1:Nreps
    offsets(i) = mean(offset_simulation_random(Nblocks, blocklen_lims));
end

end

function offsets = offset_simulation_random(Nblocks, blocklen_lims)
% Simulates a random agent and return the offset of the agent
% Nblocks: number of blocks in the session
% blocklen_lims: ranges of block lengths to sample from

offsets = [];
for i = 1:Nblocks
    blen = randsample(blocklen_lims, 1);
    perf = rand(1, blen) > 0.5;
    offset = find(perf, 1, 'first');
    if isempty(offset)
        offset = blen;
    end
    offsets(i) = offset;

end



end
