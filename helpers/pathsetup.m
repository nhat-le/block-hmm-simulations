function out = pathsetup(project)
% Set up common paths for several projects

switch project 
%     case 'matchingsim' % old Dropbox folder
%         out.rigboxpath = '/Users/minhnhatle/Dropbox (MIT)/Nhat/Rigbox';
%         out.rootpath = '/Users/minhnhatle/Dropbox (MIT)/Sur/conferences_and_manuscripts/DynamicForagingPaper/NatureCommunications/Submission 1/code';
% %         out.rootpath = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations';
%         out.datapath = fullfile(out.rootpath, 'processed_data');
%         out.codepath = fullfile(out.rootpath, 'PaperFigures/code');
%         out.figpath = fullfile(out.rootpath, '/PaperFigures/figs');
%         
%         out.expdatapath = fullfile(out.datapath, 'expdata');
%         out.blockhmmfitpath = fullfile(out.datapath, 'blockhmmfit');
%         out.simdatapath = fullfile(out.datapath, 'simdata');
%         out.decodingdatapath = fullfile(out.datapath, 'decoding');
%         out.decodingconfigpath = fullfile(out.datapath, 'decoding/configs');
%         out.decodingmodelpath = fullfile(out.datapath, 'decoding/models');
% 
%         out.blockhmm_codepath = fullfile(out.codepath, 'blockhmm');
%         out.characterize_codepath = fullfile(out.codepath, 'characterization');
%         out.decoding_codepath = fullfile(out.codepath, 'decoding');
%         out.expfit_codepath = fullfile(out.codepath, 'expfit');
%         out.schematic_codepath = fullfile(out.codepath, 'schematic');

    case 'matchingsim'
        out.rigboxpath = '/Users/minhnhatle/Dropbox (MIT)/Nhat/Rigbox';
        out.rootpath = '/Users/minhnhatle/Documents/Sur/MatchingSimulations';
%         out.rootpath = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations';
        out.datapath = fullfile(out.rootpath, 'processed_data');
        out.codepath = fullfile(out.rootpath, 'PaperFigures/code');
        out.figpath = fullfile(out.rootpath, '/PaperFigures/figs');
        
        out.expdatapath = fullfile(out.datapath, 'expdata');
        out.blockhmmfitpath = fullfile(out.datapath, 'blockhmmfit');
        out.simdatapath = fullfile(out.datapath, 'simdata');
        out.decodingdatapath = fullfile(out.datapath, 'decoding');
        out.decodingconfigpath = fullfile(out.datapath, 'decoding/configs');
        out.decodingmodelpath = fullfile(out.datapath, 'decoding/models');

        out.blockhmm_codepath = fullfile(out.codepath, 'blockhmm');
        out.characterize_codepath = fullfile(out.codepath, 'characterization');
        out.decoding_codepath = fullfile(out.codepath, 'decoding');
        out.expfit_codepath = fullfile(out.codepath, 'expfit');
        out.schematic_codepath = fullfile(out.codepath, 'schematic');
        
    case 'tca'
        out.datapath = '/Users/minhnhatle/Documents/ExternalCode/tca/data';
        out.codepath = '/Users/minhnhatle/Documents/ExternalCode/tca/src/matlab';
        out.rawdatapath = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/raw/extracted';
        out.tcamatpath = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/tca-factors';
        out.surfaceimgpath = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/session_surface_imgs';
        out.rigboxpath = '/Users/minhnhatle/Dropbox (MIT)/Nhat/Rigbox';
        
    case 'wftoolbox'
        out.rawdatapath = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/raw/extracted';
        out.locanmfpath = '/Users/minhnhatle/Documents/ExternalCode/locaNMF-preprocess';
        out.codepath = '/Users/minhnhatle/Documents/ExternalCode/wftoolbox';
        out.optoImg_path = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/Dec21/Opto Image Extraction';
        out.templates_path = '/Users/minhnhatle/Documents/ExternalCode/wftoolbox/templates';
        out.keji_path = '/Volumes/KEJI_DATA_1/nhat/processed-WF';
        out.processed_path = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/';
        out.surface_path = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/processed/surface_imgs';
        out.opto_data_path = '/Volumes/GoogleDrive/Other computers/Opto galvo/Data (1)/';
        out.rigboxpath = '/Users/minhnhatle/Dropbox (MIT)/Nhat/Rigbox';
        
    case '3p1'
        out.codepath = '/Users/minhnhatle/Dropbox (MIT)/Sur/3p1/src/';
        out.recordpath = '/Users/minhnhatle/Dropbox (MIT)/Sur/3p1/processed/records/';
        out.shared_data_path = '/Volumes/GoogleDrive/My Drive/3p1data';
        
        
    case 'opto'
        out.opto_expdatapath = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/PaperFigures/code/blockhmm/opto_analysis/optodata';
        out.opto_rigboxpath = '/Volumes/GoogleDrive/Other computers/ImagingDESKTOP-AR620FK/LocalExpData/';
        out.default_fitrange = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/processed_data/expdata/102121/fitranges_122221.mat';
        out.decodingmodelpath = '/Users/minhnhatle/Dropbox (MIT)/Sur/conferences_and_manuscripts/DynamicForagingPaper/NatureCommunications/Submission 1/code/processed_data/decoding/models';
       

    case 'hmm3p'
        out.expdatapath = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/PaperFigures/code/blockhmm/hmm_analysis_3p/data_3p';
        out.rigboxpath = '/Users/minhnhatle/Dropbox (MIT)/Nhat/Rigbox';
        out.default_fitrange = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/processed_data/expdata/102121/fitranges_122221.mat';
        out.decodingmodelpath = '/Users/minhnhatle/Dropbox (MIT)/Sur/conferences_and_manuscripts/DynamicForagingPaper/NatureCommunications/Submission 1/code/processed_data/decoding/models';
        out.glmpath = '/Users/minhnhatle/Documents/ExternalCode/GLM_clustering';

    case 'hmm1p'
        out.expdatapath = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/PaperFigures/code/blockhmm/hmm_analysis_1p/data_1p';
        out.rigboxpath = '/Users/minhnhatle/Dropbox (MIT)/Nhat/Rigbox';
        out.default_fitrange = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/processed_data/expdata/102121/fitranges_122221.mat';
        out.decodingmodelpath = '/Users/minhnhatle/Dropbox (MIT)/Sur/conferences_and_manuscripts/DynamicForagingPaper/NatureCommunications/Submission 1/code/processed_data/decoding/models';
           

    otherwise
        error('Unrecognized project.')
end


end