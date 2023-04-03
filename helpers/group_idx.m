function new_idx = group_idx(idx, groups)
for i = 1:numel(groups)
    idx = group_helper(idx, groups{i});
end

% Re-number
group_labels = unique(idx);
new_idx = idx;
for i = 1:numel(group_labels)
    new_idx(idx == group_labels(i)) = i;
end

end


function new_idx = group_helper(idx, group)
new_idx = idx;
for i = 1:numel(group)
    new_idx(idx == group(i)) = group(1);
end

end