function labels = assign_labels(Y, L, binX, binY, varargin)
if numel(varargin) == 0
    method = 'nearest';
elseif numel(varargin) == 2
    method = varargin{1};
    Lfill = varargin{2};
elseif numel(varargin) == 1
    method = varargin{1};
    if strcmp(method, 'Lfill')
        Lfill = remove_borders(L);
        warning('No Lfill matrix provided, re-computing Lfill...')
    end
end


if strcmp(method, 'nearest')
    labels = [];
    for i = 1:size(Y,1)
        ordersX(i) = find_order_in_arr(binX, Y(i,1)) - 1;
        ordersY(i) = find_order_in_arr(binY, Y(i,2)) - 1;
        
        if L(ordersX(i), ordersY(i)) ~= 0
            labels(i) = L(ordersX(i), ordersY(i));
        else
            % 0 <= xedge, yedge < 1 is the normalized edge of each point
            xedge = (Y(i,1) - binX(ordersX(i))) / (binX(2) - binX(1));
            yedge = (Y(i,2) - binY(ordersY(i))) / (binY(2) - binY(1));
            assert(xedge >= 0 & xedge < 1 & yedge >= 0 & yedge < 1);
        
            % find neighbors
            [lx, ly] = size(L);
            candlabels = [];
            canddists = [];
            
            % sweep through the neighbors, keeping track of the distance
            % and the label of those neighbors, skipping if needed
            for dx = [-1, 0, 1]
                for dy = [-1, 0, 1]
                    xnew = ordersX(i) + dx;
                    ynew = ordersY(i) + dy;
                    if xnew <= 0 || xnew > lx || ynew <= 0 || ynew > ly
                        continue
                    end
                    
                    candlabel = L(xnew, ynew);
                    if candlabel == 0
                        continue
                    else
                        candlabels = [candlabels candlabel];
                        canddists = [canddists (xedge - dx)^2 + (yedge - dy)^2];
                        
                    end
                    
                    
                    
                end
            end
            
            % Minimize the distance
            [~,id] = min(canddists);
            if isempty(id)
                disp('here')
            end
            labels(i) = candlabels(id);
        end
        
    end
    
elseif strcmp(method, 'Lfill')
    ordersX = [];
    ordersY = [];
    labels = [];

    for i = 1:size(Y,1)
        ordersX(i) = find_order_in_arr(binX, Y(i,1)) - 1;
        ordersY(i) = find_order_in_arr(binY, Y(i,2)) - 1;
        labels(i) = Lfill(ordersX(i), ordersY(i));
    end
            
end


end