function goshortcut(pathname)
% Utility to quickly change to desired working directory

switch pathname
    case 'glm'
        cd('/Users/minhnhatle/Documents/ExternalCode/GLM_clustering');
    case 'tca'
        cd('/Users/minhnhatle/Documents/ExternalCode/tca');
    case 'matchingsim'
        cd('/Users/minhnhatle/Documents/Sur/MatchingSimulations')
    case 'opto'
        cd('/Users/minhnhatle/Documents/Sur/MatchingSimulations/PaperFigures/code/blockhmm/opto_analysis')
    case 'wftoolbox'
        cd('/Users/minhnhatle/Documents/ExternalCode/wftoolbox/scripts/glm')
    otherwise
        fprintf('Path not recognized\n');




end



end