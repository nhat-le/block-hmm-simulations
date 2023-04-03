function effs = perfect_agent_metrics(blocklen_lims, Nblocks, Nreps)
% blocklen_lims: block lengths to sample from
% Nblocks: number of blocks in the session
% Nreps: number of reps to simulate
% Simulates the performance and metrics of a "perfect" agent
% that performs a win-stay lose-switch strategy
rng(123)
effs = nan(Nreps, 1);
for i = 1:Nreps
    effs(i) = eff_simulation(Nblocks, blocklen_lims);
end

end



function eff = eff_simulation(Nblocks, blocklen_lims)
% Simulates a perfect agent and return the efficiency of the agent
Ntotal = 0;
Ncorr = 0;

for i = 1:Nblocks
    blen = randsample(blocklen_lims, 1);
    Ntotal = Ntotal + blen;
    Ncorr = Ncorr + blen - 1;

end

eff = Ncorr / Ntotal;


end