function allmeans = getmeans(obs, zstates)
allmeans = [];
for i = 1:max(zstates) + 1
    obsfilt = obs(zstates == i-1, :);
    allmeans(i,:) = nanmean(obsfilt, 1);
end

end
