function [opt] = getOpts(expt_name)
	
	switch expt_name
        
		case 'DRPAN_without_expt'
			opt = getDefaultOpts();
		
			opt.which_algs_paths = {'DRPAN_without'};
			opt.Nimgs = 500;
			opt.ut_id = '0d827e73e4f4d77d2b40fb9d04726d7f'; % set this using http://uniqueturker.myleott.com/
			opt.base_url = 'http://wuziding.org/resource/AMT/Maps/';
			opt.instructions_file = './instructions_basic.html';
			opt.short_instructions_file = './short_instructions_basic.html';
			opt.consent_file = './consent_basic.html';
			opt.use_vigilance = false;
			opt.paired = true;
		
		otherwise
			error(sprintf('no opts defined for experiment %s',expt_name));
	end
	
	opt.expt_name = expt_name;
end