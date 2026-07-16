
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = imageTask_paper_parameter_recovery_combinedModel; 

%this code implements paramter recovery for the biased prior + cost
%to sample combo model and makesthe supplementary figures. Use v5 for other
%models, its optimised for them,

%took v4 and modified it to run combo model with cs and bp parameters both

tic

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\FMINSEARCHBND'))
addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\plotSpread'));
addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\klabhub-bayesFactor-3d1e8a5'));

% Hybrid CtS + Biased Prior parameter recovery diagnostic
% -------------------------------------------------------------------------
% This version runs ONLY the hybrid model. It is intended as a diagnostic,
% not as a polished/full PR pipeline.
%
% IMPORTANT: analyzeSecretaryPR.m must implement identifier == 9 as a
% Bayesian model in which BOTH:
%   - cost-to-sample/reward-to-sample Cs is active, and
%   - biased-prior mean offset BP is active.
% See notes in the ChatGPT response for the small patch to make there.

% Run switches. For a fresh run, leave these as 1/1/1/0/0/1.
simulate_stimuli      = 1;
make_config_model_data = 1;
check_params          = 1;
make_est_model_data   = 0;  % not needed for parameter-recovery scatterplots
use_file_for_plots    = 0;  % set to 1 only to replot an existing file
make_plots            = 1;

all_draws_set = 1;
log_or_not = 0;

% Hybrid model identifier. Do not reuse 6 unless analyzeSecretaryPR applies
% Cs in the BP branch. Safer is to add identifier==9 handling there.
hybrid_identifier = 9;
do_models = hybrid_identifier;  % switch off all other models
model_names = {'CO' 'Cs' 'IO' 'BV' 'BR' 'BPM' 'Opt' 'BPV' 'CsBPM'};
num_model_identifiers = numel(model_names);

% Keep this TRUE for a quick smoke test; set FALSE for the real diagnostic.
quick_test = false;
if quick_test
    pr_num_subs = 5;
    Cs_levels = linspace(-5.5, 3.0, 3);   % 3 x 3 = 9 model configurations
    BP_levels = linspace(-50, 70, 3);
else
    pr_num_subs = 20;
    Cs_levels = linspace(-5.5, 3.0, 5);   % 5 x 5 = 25 model configurations
    BP_levels = linspace(-50, 70, 5);
end

fit_betas = 1;      % beta is still fitted; this is its configured/initial value
num_fit_betas = numel(fit_betas);
num_param_levels = numel(Cs_levels) * numel(BP_levels);  % for file naming only
total_params_per_model = num_param_levels * num_fit_betas;

diagnostic_stamp = char(datetime('now','Format','yyyyMMdd_HHmmss'));
comment = sprintf('COMBO_CsBPM_PR_%s', diagnostic_stamp);
outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\outputs';
if ~exist(outpath, 'dir'); mkdir(outpath); end
filename_for_plots = '';  % fill manually only if use_file_for_plots==1

rng(1);  % reproducible simulated stimuli and exceedance-ish plots

if simulate_stimuli == 1;
    
    Generate_params.log_or_not = log_or_not;
    Generate_params.all_draws_set = all_draws_set;
    Generate_params.do_models_identifiers = do_models;
    Generate_params.num_param_levels = num_param_levels;
    Generate_params.fit_betas = fit_betas;
    Generate_params.num_fit_betas = num_fit_betas;
    Generate_params.total_params_per_model = total_params_per_model;
    
    %Configure sequences
    %Careful, the number of items in the sequences (seq_length*num_seqs)
    %should not exceed the number of items rated in phase 1. Ideally the
    %former should be at most 60% of latter.
    Generate_params.num_subs = pr_num_subs;   %So this will be per parameter value
    Generate_params.num_seqs = 5;
    Generate_params.seq_length = 8; %hybrid might've usually used 12 but imageTask always uses 8!
    Generate_params.num_vals = 426;  %How many items in phase 1 and available as options in sequences? I've used before 426 or 90
    Generate_params.rating_bounds = [1 100];    %What is min and max of rating scale?
    Generate_params.rating_grand_mean = 40;     %Individual subjects' rating means will jitter around this (50 or 39.5. The latter comes from the midpoint between NEW hybrid SV ratings mean (30) and normalised price mean (49.2)
    Generate_params.rating_mean_jitter = 5;     %How much to jitter participant ratings means on average?
    Generate_params.rating_grand_std = 20;       %Individual subjects' rating std devs will jitter around this (5 or 18, the latter is the midpoint b/n NEW hybrid SV and OV)
    Generate_params.rating_var_jitter = 2;     %How much to jitter participant ratings vars on average?
    
    for sub = 1:Generate_params.num_subs;
        
        this_sub_rating_mean = Generate_params.rating_grand_mean + normrnd( 0, Generate_params.rating_mean_jitter );
        this_sub_rating_std = Generate_params.rating_grand_std + normrnd( 0, Generate_params.rating_var_jitter );
        
        %Generate a truncated normal distribution of option values
        pd = truncate(makedist('Normal','mu',this_sub_rating_mean,'sigma',this_sub_rating_std),Generate_params.rating_bounds(1),Generate_params.rating_bounds(2));
        phase1 = random(pd,Generate_params.num_vals,1);
        
        if log_or_not == 1;
            phase1 = log(phase1);
            Generate_params.BVrange = log( Generate_params.rating_bounds )
        else
            Generate_params.BVrange = Generate_params.rating_bounds;
        end;    %transform ratings if log_or_not
        
        %Save this sub's ratings data
        Generate_params.ratings(:,sub) = phase1;
        
        %Grab the requisit number of random ratings
        temp_ratings = phase1(randperm(numel(phase1)),1);
        Generate_params.seq_vals(:,:,sub) = reshape(...
            temp_ratings(1:Generate_params.num_seqs*Generate_params.seq_length,1) ...
            ,Generate_params.num_seqs ...
            ,Generate_params.seq_length ...
            );
        
        Generate_params.PriorMean = mean(Generate_params.ratings(:,sub));
        Generate_params.PriorVar = var(Generate_params.ratings(:,sub));
        
    end;    %Each subject to create stimuli
end;    %Should I create stimuli for simulation?

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%CONFIGURE MODELS AND GET PERFORMANCE!!!!!!%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if make_config_model_data == 1

    % ---- Template model -------------------------------------------------
    opt_rule = ceil(exp(-1)*Generate_params.seq_length);

    model_template.identifier = hybrid_identifier;
    model_template.kappa = 2;
    model_template.nu = 1;
    model_template.cutoff = opt_rule;
    model_template.Cs = 0;
    model_template.BVslope = 0.2;
    model_template.BVmid = 50;
    model_template.BRslope = 1;
    model_template.BRmid = 50;
    model_template.BP = 0;
    model_template.optimism = 0;
    model_template.BPV = 0;
    model_template.all_draws = all_draws_set;
    model_template.beta = fit_betas(1);
    model_template.name = 'CsBPM';

    % Row lookup for param_config. This avoids the legacy beta-row confusion.
    row.identifier = 1;
    row.kappa      = 2;
    row.nu         = 3;
    row.cutoff     = 4;
    row.Cs         = 5;
    row.BVslope    = 6;
    row.BVmid      = 7;
    row.BRslope    = 8;
    row.BRmid      = 9;
    row.BP         = 10;
    row.optimism   = 11;
    row.BPV        = 12;
    row.all_draws  = 13;
    row.beta       = 14;
    Generate_params.row = row;

    base_config = [ ...
        model_template.identifier;
        model_template.kappa;
        model_template.nu;
        model_template.cutoff;
        model_template.Cs;
        model_template.BVslope;
        model_template.BVmid;
        model_template.BRslope;
        model_template.BRmid;
        model_template.BP;
        model_template.optimism;
        model_template.BPV;
        model_template.all_draws;
        model_template.beta ];

    [Cs_grid, BP_grid] = ndgrid(Cs_levels, BP_levels);
    n_configs = numel(Cs_grid);

    param_config = repmat(base_config, 1, n_configs);
    param_config(row.identifier,:) = hybrid_identifier;
    param_config(row.Cs,:) = Cs_grid(:)';
    param_config(row.BP,:) = BP_grid(:)';
    param_config(row.beta,:) = fit_betas(1);

    % Defaults/starting values for fitting. These remain 0/0 for Cs/BP.
    param_config_default = repmat(base_config, 1, n_configs);
    param_config_default(row.identifier,:) = hybrid_identifier;
    param_config_default(row.Cs,:) = 0;
    param_config_default(row.BP,:) = 0;
    param_config_default(row.beta,:) = fit_betas(1);

    free_parameters = zeros(size(param_config));
    free_parameters(row.Cs,:) = 1;
    free_parameters(row.BP,:) = 1;

    Generate_params.do_models_identifiers = hybrid_identifier;
    Generate_params.hybrid_identifier = hybrid_identifier;
    Generate_params.Cs_levels = Cs_levels;
    Generate_params.BP_levels = BP_levels;
    Generate_params.num_param_levels = n_configs;
    Generate_params.fit_betas = fit_betas;
    Generate_params.num_fit_betas = num_fit_betas;
    Generate_params.total_params_per_model = n_configs;

    Generate_params.num_models = size(param_config,2);
    Generate_params.param_config_matrix = param_config;
    Generate_params.free_parameters_matrix = free_parameters;
    Generate_params.comment = comment;
    Generate_params.outpath = outpath;

    analysis_name = sprintf( ...
        'out_PR_%dHybridConfigs_%dsubs_%dseqs_%dopts_%s_', ...
        Generate_params.num_models, ...
        Generate_params.num_subs, ...
        Generate_params.num_seqs, ...
        Generate_params.seq_length, ...
        Generate_params.comment);

    Generate_params.analysis_name = analysis_name;
    Generate_params.outname = [analysis_name '.mat'];

    % Refuse to overwrite an existing diagnostic file.
    out_file = fullfile(Generate_params.outpath, Generate_params.outname);
    if exist(out_file, 'file')
        error('Output file already exists: %s', out_file);
    end

    disp(sprintf('Running %s', Generate_params.outname));

    % ---- Create each configured hybrid model and simulate data -----------
    fields = fieldnames(model_template);
    for model = 1:Generate_params.num_models

        Generate_params.current_model = model;

        for field = 1:(numel(fields)-1) % exclude name
            Generate_params.model(model).(fields{field}) = param_config(field,model);
        end
        Generate_params.model(model).name = 'CsBPM';

        Generate_params.model(model).this_models_free_parameters = find(free_parameters(:,model)==1);
        Generate_params.model(model).this_models_free_parameter_default_vals = ...
            param_config_default(Generate_params.model(model).this_models_free_parameters,model)';
        Generate_params.model(model).this_models_free_parameter_configured_vals = ...
            param_config(Generate_params.model(model).this_models_free_parameters,model)';

        Generate_params.num_subs_to_run = 1:Generate_params.num_subs;
        [Generate_params.model(model).num_samples, Generate_params.model(model).ranks] = ...
            generate_a_models_data(Generate_params);
    end

    save(out_file, 'Generate_params');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%DO THE MODEL FITTING!!!!!!%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if check_params == 1

    % Parameter bounds by row index in param_config/model_template.
    row = Generate_params.row;
    param_bounds = nan(14,2);
    param_bounds(row.cutoff,:) = [2 Generate_params.seq_length-1];
    param_bounds(row.Cs,:) = [-100 100];
    param_bounds(row.BVmid,:) = [1 100];
    param_bounds(row.BRmid,:) = [1 100];
    param_bounds(row.BP,:) = [-100 100];
    param_bounds(row.optimism,:) = [-100 100];
    param_bounds(row.BPV,:) = [-100 100];
    param_bounds(row.beta,:) = [0 100];

    do_models = 1:Generate_params.num_models;

    for model = do_models
        for sub = 1:Generate_params.num_subs

            Generate_params.current_model = model;
            Generate_params.num_subs_to_run = sub;

            free_idx = Generate_params.model(model).this_models_free_parameters;

            % Start at default Cs=0, BP=0, beta=1.
            params = [ ...
                Generate_params.model(model).this_models_free_parameter_default_vals ...
                Generate_params.model(model).beta ];

            lower_bounds = [param_bounds(free_idx,1)' param_bounds(row.beta,1)];
            upper_bounds = [param_bounds(free_idx,2)' param_bounds(row.beta,2)];

            warning('off');

            param_str = sprintf(' %.3f', Generate_params.model(model).this_models_free_parameter_configured_vals);
            disp(sprintf('fitting model %d %s configured params [%s ] subject %d', ...
                model, ...
                Generate_params.model(Generate_params.current_model).name, ...
                param_str, ...
                sub));

            [Generate_params.model(model).estimated_params(sub,:), ...
             Generate_params.model(model).ll(sub,:), ...
             exitflag, search_out] = ...
                fminsearchbnd(@(params) f_fitparams(params, Generate_params), ...
                    params, lower_bounds, upper_bounds);

            Generate_params.model(model).exitflag(sub) = exitflag;
        end

        save(fullfile(Generate_params.outpath, Generate_params.outname), 'Generate_params');
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Generate performance from estimated parameters!%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if make_est_model_data == 1;
    
    for model = 1:Generate_params.num_models;
        
        Generate_params.current_model = model;
        
        %%%%%Here's the main function call in this
        %%%%%section!!!%%%%%%%%%%%%%%%%%%%
        %returns subject*sequences matrices of numbers of draws and ranks
        %I'm using a slightly modified function so I can manipulate params
        %in Generate_params that aren't permanent
        [temp1 temp2] = ...
            generate_a_models_data_est(Generate_params);
        
        %For
        Generate_params.model(model).num_samples_est = temp1';
        Generate_params.model(model).ranks_est = temp2';
        
    end;    %Loop through models
    
    save([Generate_params.outpath filesep Generate_params.outname], 'Generate_params');
    
end;        %if make_est_model_data?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%PLOT!%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if make_plots == 1;
    
    %Use data already in workspace because computed above by or ?
    %Or open a file where these things were computed on previous run?
    if use_file_for_plots == 1;
        
        load(filename_for_plots,'Generate_params');
        
    end;    %Plot data structure from a file?
    
    plot_data(Generate_params);
    
end;    %Do plots or not?


%Just to be safe
% save([Generate_params.outpath filesep Generate_params.outname],'Generate_params');

disp('audi5000')

toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [num_samples ranks] = generate_a_models_data_est(Generate_params);

%I use this version to get performance data when I need to change from
%configured to estimated parameters first

for sub = 1:Generate_params.num_subs;
    
    %So, change from configured to estimated parameters first then ...
    %We need to temporarilty change the parameter fields to the current
    %parameter settings if are to use generate_a_models_data to get performance
    it = 1;
    fields = fieldnames(Generate_params.model(Generate_params.current_model));
    for field = Generate_params.model(Generate_params.current_model).this_models_free_parameters';   %loop through all free parameter indices (except beta)
        
        Generate_params.model(Generate_params.current_model).(fields{field}) = ...
            Generate_params.model(Generate_params.current_model).estimated_params(sub,it);
        it=it+1;
        
    end;
    
    disp(...
        sprintf('computing performance, fitted modeli %d name %s subject %d' ...
        , Generate_params.current_model ...
        , Generate_params.model( Generate_params.current_model ).name ...
        , sub ...
        ) );
    
    Generate_params.num_subs_to_run = sub;
    [num_samples(sub,:) ranks(sub,:)] = generate_a_models_data(Generate_params);
    
end;    %Loop through subs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  ll = f_fitparams( params, Generate_params );

%We need to temporarilty change the parameter fields to the current
%parameter settings if are to use generate_a_models_data to get performance
it = 1;
fields = fieldnames(Generate_params.model(Generate_params.current_model));
for field = Generate_params.model(Generate_params.current_model).this_models_free_parameters';   %loop through all free parameter indices (except beta)
    
    Generate_params.model(Generate_params.current_model).(fields{field}) = params(it);
    it=it+1;
    
end;
%and now assign beta too
b = params(end);

%(generate_a_models_data can do multiple subjects but here we want to fit
%one subject at a time and the number of subjects to be run is set before f_fitparams function call in a
%field of Generate_params
[num_samples ranks choiceStop_all choiceCont_all] = generate_a_models_data(Generate_params);
%num_samples and ranks are seqs and choice* are both seq*sub

ll = 0;

for seq = 1:Generate_params.num_seqs;
    
    %Log likelihood for this subject
    
    %Get action values for this sequence
    %seq*seqpos
    choiceValues = [choiceCont_all(seq,:); choiceStop_all(seq,:)]';
    
    %Need to limit the sequence by the "subject's" (configured simulation's)
    %number of draws ...
    
    %How many samples for this model for this sequence and subject
    listDraws = ...
        Generate_params.model(Generate_params.current_model).num_samples(seq,Generate_params.num_subs_to_run);
    
    %Loop through trials to be modelled to get choice probabilities for
    %each action value
    for drawi = 1 : listDraws
        %cprob seqpos*choice(draw/stay)
        cprob(drawi, :) = exp(b*choiceValues(drawi, :))./sum(exp(b*choiceValues(drawi, :)));
    end;
    
    %Compute ll
    if listDraws == 1;  %If only one draw
        ll = ll - 0 - log(cprob(listDraws, 2));
    else
        if  Generate_params.model(Generate_params.current_model).all_draws == 1;
            ll = ll - sum(log(cprob((1:listDraws-1), 1))) - log(cprob(listDraws, 2));
        else;
            ll = ll - sum(log(cprob((listDraws-1), 1))) - log(cprob(listDraws, 2));
        end;
    end;
    
end;    %seq loop



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [num_samples ranks choiceStop_all choiceCont_all] = generate_a_models_data(Generate_params);


%returns subject*sequences matrices of numbers of draws and ranks

%note: What is called which_model here is param_to_fit(model) outside this programme,
%at the base level

this_sub = 1;   %Need to assign each sub to output array by how many have been run, rathere than by sub num
for num_subs_found = Generate_params.num_subs_to_run;
    
    if numel(Generate_params.num_subs_to_run) > 1; %i.e., if model fitting to a single subject is not going on here
        disp(...
            sprintf('generating performance for preconfigured modeli %d name %s subject %d' ...
            , Generate_params.current_model ...
            ,Generate_params.model( Generate_params.current_model ).name ...
            , num_subs_found ...
            ) );
    end;
    
    for sequence = 1:Generate_params.num_seqs;
        
        list.vals = squeeze(Generate_params.seq_vals(sequence,:,num_subs_found));
        
        %ranks for this sequence
        dataList = tiedrank(squeeze(Generate_params.seq_vals(sequence,:,num_subs_found))');

        %Do cutoff model, if needed
        if Generate_params.model(Generate_params.current_model).identifier == 1;

            %initialise all sequence positions to zero/continue (value of stopping zero)
            choiceStop = zeros(1,Generate_params.seq_length);
            %What's the cutoff?
            estimated_cutoff = round(Generate_params.model(Generate_params.current_model).cutoff);
            if estimated_cutoff < 1; estimated_cutoff = 1; end;
            if estimated_cutoff > Generate_params.seq_length; estimated_cutoff = Generate_params.seq_length; end;
            %find seq vals greater than the max in the period before cutoff and give these candidates a maximal stopping value of 1
            choiceStop(1,find( list.vals > max(list.vals(1:estimated_cutoff)) ) ) = 1;
            %set the last position to 1, whether it's greater than the best in the learning period or not
            choiceStop(1,Generate_params.seq_length) = 1;
            %find first index that is a candidate ....
            num_samples(sequence,this_sub) = find(choiceStop == 1,1,'first');   %assign output num samples for cut off model
            %Reverse 0s and 1's
            choiceCont = double(~choiceStop);

        else;   %Any Bayesian models

            [choiceStop, choiceCont, difVal]  = ...
                analyzeSecretaryPR(Generate_params,list.vals);
            %                 analyzeSecretaryNick_2021(Generate_params,list);


            num_samples(sequence,this_sub) = find(difVal<0,1,'first');  %assign output num samples for Bruno model

        end;    %Cutoff or other model?

        %...and its rank
        ranks(sequence,this_sub) = dataList( num_samples(sequence,this_sub) );
        %Accumulate action values too so you can compute ll outside this function if needed
        choiceStop_all(sequence, :, this_sub) = choiceStop;
        choiceCont_all(sequence, :, this_sub) = choiceCont;

    end;    %loop through sequences

    this_sub = this_sub + 1;
    
end;    %loop through subs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%












%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = plot_data(Generate_params)
% Minimal hybrid PR plot: configured vs recovered Cs and BP, plus
% fitted-parameter trade-off. Saves PNG and CSV next to the .mat output.

row = Generate_params.row;

cfg = [];
est = [];
model_idx = [];
sub_idx = [];
mean_samples = [];

for model = 1:Generate_params.num_models
    if ~isfield(Generate_params.model(model), 'estimated_params')
        continue
    end

    true_vals = Generate_params.model(model).this_models_free_parameter_configured_vals; % [Cs BP]
    E = Generate_params.model(model).estimated_params; % columns: Cs, BP, beta

    nsub = size(E,1);
    cfg = [cfg; repmat(true_vals, nsub, 1)];
    est = [est; E(:,1:2)];
    model_idx = [model_idx; repmat(model, nsub, 1)];
    sub_idx = [sub_idx; (1:nsub)'];
    mean_samples = [mean_samples; repmat(nanmean(nanmean(Generate_params.model(model).num_samples)), nsub, 1)];
end

configured_Cs = cfg(:,1);
configured_BP = cfg(:,2);
estimated_Cs = est(:,1);
estimated_BP = est(:,2);

[r_Cs, p_Cs] = corr(configured_Cs, estimated_Cs, 'rows', 'complete');
[r_BP, p_BP] = corr(configured_BP, estimated_BP, 'rows', 'complete');
[r_trade, p_trade] = corr(estimated_Cs, estimated_BP, 'rows', 'complete');

fprintf('\nHybrid parameter recovery summary\n');
fprintf('Configured vs estimated Cs: r = %.3f, p = %.3g\n', r_Cs, p_Cs);
fprintf('Configured vs estimated BP: r = %.3f, p = %.3g\n', r_BP, p_BP);
fprintf('Estimated Cs vs estimated BP: r = %.3f, p = %.3g\n', r_trade, p_trade);

outstem = fullfile(Generate_params.outpath, strrep(Generate_params.outname, '.mat', ''));

T = table(model_idx, sub_idx, configured_Cs, configured_BP, estimated_Cs, estimated_BP, mean_samples, ...
    'VariableNames', {'model_idx','sub_idx','configured_Cs','configured_BP','estimated_Cs','estimated_BP','configured_mean_samples'});
writetable(T, [outstem '_hybrid_PR_values.csv']);

fig = figure('Color','w','Name','Hybrid Cs+BPM parameter recovery', 'Position', [100 100 1100 350]);

subplot(1,3,1); hold on;
scatter(configured_Cs, estimated_Cs, 25, 'filled', 'MarkerFaceAlpha', .45);
add_identity_line(configured_Cs, estimated_Cs);
xlabel('Configured Cs'); ylabel('Recovered Cs');
title(sprintf('Cs recovery: r = %.2f', r_Cs));
box off;

subplot(1,3,2); hold on;
scatter(configured_BP, estimated_BP, 25, 'filled', 'MarkerFaceAlpha', .45);
add_identity_line(configured_BP, estimated_BP);
xlabel('Configured BP'); ylabel('Recovered BP');
title(sprintf('BP recovery: r = %.2f', r_BP));
box off;

subplot(1,3,3); hold on;
scatter(estimated_Cs, estimated_BP, 25, 'filled', 'MarkerFaceAlpha', .45);
xlabel('Recovered Cs'); ylabel('Recovered BP');
title(sprintf('Recovered trade-off: r = %.2f', r_trade));
box off;

saveas(fig, [outstem '_hybrid_PR_scatter.png']);
savefig(fig, [outstem '_hybrid_PR_scatter.fig']);

function add_identity_line(x, y)
    vals = [x(:); y(:)];
    vals = vals(isfinite(vals));
    if isempty(vals); return; end
    lo = min(vals); hi = max(vals);
    pad = .05 * (hi - lo + eps);
    xlim([lo-pad hi+pad]); ylim([lo-pad hi+pad]);
    plot([lo-pad hi+pad], [lo-pad hi+pad], 'k--', 'LineWidth', 1);
