function [opt] = getOpts(expt_name)
	
	switch expt_name
        
		case 'DRPAN_without_expt'
			opt = getDefaultOpts();
		
			opt.which_algs_paths = {'DRPAN_without'};
			opt.Nimgs = 1050;
			opt.ut_id = '5becadc07d214f370935453b277604d1'; % set this using http://uniqueturker.myleott.com/
			opt.base_url = 'http://wuziding.org/resource/AMT/Maps/';
			opt.instructions_file = './instructions_basic.html';
			opt.short_instructions_file = './short_instructions_basic.html';
			opt.consent_file = './consent_basic.html';
			opt.use_vigilance = false;
			opt.paired = true;
            
        case 'DRPAN_iter3'
            opt = getDefaultOpts();
		
			opt.which_algs_paths = {'DRPAN_iteration_3'};
            opt.gt_path = 'DRPAN'; 
			opt.Nimgs = 500;
			opt.ut_id = 'fd58991ae1aa289245699d400d0daa7b'; % set this using http://uniqueturker.myleott.com/
			opt.base_url = 'http://checkimage.blockfundchain.net/';
			opt.instructions_file = './instructions_DRPAN_iter3.html';
			opt.short_instructions_file = './short_instructions_DRPAN_iter3.html';
			opt.consent_file = './consent_DRPAN_iter3.html';
			opt.use_vigilance = false;
			opt.paired = true;
            opt.im_height = 256;                    % dimensions at which to display the stimuli
            opt.im_width = 512;
            
        case 'DRPAN_R'
            opt = getDefaultOpts();
		
			opt.which_algs_paths = {'DRPAN'};
            opt.gt_path = 'DRPAN_R'; 
			opt.Nimgs = 500;
			opt.ut_id = '977a8c2c04b26b5a1b097e5f4558249c'; % set this using http://uniqueturker.myleott.com/
			opt.base_url = 'http://checkimage.blockfundchain.net/';
            opt.instructions_file = './instructions_DRPAN_iter3.html';
			opt.short_instructions_file = './short_instructions_DRPAN_iter3.html';
			opt.consent_file = './consent_DRPAN_iter3.html';
% 			opt.instructions_file = './instructions_DRPAN_R.html';
% 			opt.short_instructions_file = './short_instructions_DRPAN_R.html';
% 			opt.consent_file = './consent_DRPAN_R.html';
			opt.use_vigilance = false;
			opt.paired = true;
            opt.im_height = 256;                    % dimensions at which to display the stimuli
            opt.im_width = 512;
            
        case 'pix2pixHD'
            opt = getDefaultOpts();
		
			opt.which_algs_paths = {'DRPAN_pix2pixHD'};
            opt.gt_path = 'pix2pixHD'; 
			opt.Nimgs = 106;
			opt.ut_id = '86a8abdd7bf5558e40475a124c54a82f'; % set this using http://uniqueturker.myleott.com/
			opt.base_url = 'http://checkimage.blockfundchain.net/';
            opt.instructions_file = './instructions_DRPAN_iter3.html';
			opt.short_instructions_file = './short_instructions_DRPAN_iter3.html';
			opt.consent_file = './consent_DRPAN_iter3.html';
% 			opt.instructions_file = './instructions_pix2pixHD.html';
% 			opt.short_instructions_file = './short_instructions_pix2pixHD.html';
% 			opt.consent_file = './consent_pix2pixHD.html';
			opt.use_vigilance = false;
			opt.paired = true;
            opt.im_height = 512;                    % dimensions at which to display the stimuli
            opt.im_width = 1024;
		
		otherwise
			error(sprintf('no opts defined for experiment %s',expt_name));
	end
	
	opt.expt_name = expt_name;
end