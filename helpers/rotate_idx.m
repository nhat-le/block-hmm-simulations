function new_idx = rotate_idx(idx, rotations)
new_idx = idx;
for i = 1:numel(rotations)
    new_idx = rotate_helper(new_idx, rotations{i});
end

end


function new_idx = rotate_helper(idx, rotation)

if numel(rotation) == 1
    new_idx = idx;
    return;
end

new_idx = idx;
for i = 1:numel(rotation) - 1
    new_idx(idx == rotation(i)) = rotation(i+1);
end

new_idx(idx == rotation(end)) = rotation(1);

end