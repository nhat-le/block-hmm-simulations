function cols = paperaesthetics

% redcol = [178,24,43]/255;%[178,24,43]/255; %[5,113,176]/255
cols.redcol = [0.7 0 0];
cols.bluecol = [33,102,172]/255; %[202,0,32]/255
cols.redbluecolormap = [229, 75, 34; 171, 209, 255]/255;
cols.redbluecolormap2 = [178,24,43; 209,229,240]/ 255;
cols.greencol = [0, 158, 115]/255;


% bluecol = [0 0 0.7];

cols.colors = brewermap(9, 'Set1');
cols.colors = cols.colors([2,1,5,6,4,3,7,8,9],:);


cols.colorsLight = brewermap(6, 'Pastel1');
cols.colorsLight = cols.colorsLight([2,1,5,6,4,3],:);
% colors(4,:) = [0,0,0];

end