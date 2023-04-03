function L = remove_borders(L)
%L: w x h image where borders are denoted zero
%use a simple algorithm to assign a class number to each border pixel
% by choosing probabilistically among the neighboring pixels

[x, y] = find(L == 0);

for i = 1:numel(x)
    cands = find_cands(x(i), y(i), L); %candidate labels    
    L(x(i), y(i)) = randsample(cands, 1); 
end


end


function cands = find_cands(i, j, L)
[lx, ly] = size(L);
% Find neighboring pixels around L(i,j) that are not zeros
left = max(i-1, 1);
right = min(i+1, lx);
up = max(j-1, 1);
down = min(j+1, ly);

square = L(left:right, up:down);
cands = square(:);
cands = cands(cands ~= 0);

end