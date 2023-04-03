function [params, aggmeans, aggparams] = load_params(folders, opts)
    effmethod = opts.effmethod;
    filter_blocks_by_lengths = opts.filter_blocks_by_lengths;
    weighted = opts.weighted;


    aggmeans = {};
    aggparams = {};
    block_corr_all = {};
    block_lens_all = {};
    for i = 1:numel(folders)
        load(fullfile(folders(i).folder, folders(i).name));

        if i == numel(folders)
            obs(isnan(obs)) = 1;
        end

        % Mean transition function for all trials in a particular z-state
        allmeans = getmeans(obs, zstates);

        % efficiency second try
        effs = [];
        nstates = size(params, 2);
        switch effmethod
            case 'rawdata'
                for zid = 1:nstates
                    block_corr_filt = double(block_corrs(zstates == zid - 1));
                    block_lens_filt = double(block_lens(zstates == zid - 1));

                    if filter_blocks_by_lengths
                        block_corr_filt = block_corr_filt(block_lens_filt > 15 & block_lens_filt < 25);
                        block_lens_filt = block_lens_filt(block_lens_filt > 15 & block_lens_filt < 25);
                    end

                    if weighted
                        effs(zid) = sum(block_corr_filt) / sum(block_lens_filt);
                    else
                        effs(zid) = mean(block_corr_filt ./ block_lens_filt);
                    end

                    block_corr_all{end+1} = block_corr_filt;
                    block_lens_all{end+1} = block_lens_filt;
                end

            case 'boost'
                for zid = 1:nstates
                    obsfiltered = obs(zstates == zid-1,:);

                % TODO: determine if this 'boosting' can be improved...
                    effs(zid) = sum(obsfiltered(:) == 1) / numel(obsfiltered) / 15*20;
                end

            case 'sim'
                for zid = 1:nstates
                    paramset = params(:, zid);
                    delta = 0.1;
                    ntrials = 25;
                    transfunc = mathfuncs.sigmoid(0:delta:ntrials, paramset(1), paramset(2), paramset(3));
                    effs(zid) = sum(transfunc) * delta / ntrials;

                end

        end 

        params(end + 1, :) = effs;

        aggmeans{i} = allmeans;
        aggparams{i} = params;   

    end
end