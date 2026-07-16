
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = fit_models_matlabStudies;

tic

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\FMINSEARCHBND'))

%The idea was to save data in files along the way as needed. But as
%implemented now, later sections are dependent on previous ones.
which_study = 2;    %1: av/face_1 %2: matchmaker %3: (small) trust 1 
check_params = 1;       %fit the same model that created the data and output estimated parameters
make_est_model_data = 1;
use_file_for_plots = 0; %Set the above to zero and this to 1 and it'll read in a file you specify (See filename_for_plots variable below) and make plots of whatever analyses are in the Generate_params structure in that file;
make_plots = 0;         %if 1, plots the results
all_draws_set = 1;          %You can toggle how the ll is computed here for all models at once if you want or go below and set different values for different models manually in structure
log_or_not = 0; %I'm changing things so all simulated data is logged at point of simulation (==1) or not
%1: cutoff 2: Cs 3: dummy (formerly IO in v2) 4: BV 5: BR 6: BPM 7: Opt 8: BPV
%(I keep model 3 as a legacy for IO because analyseSecertaryNick_2021
%looks for identifiers 4 and 5 for BV and BR and needs that for v2. Also it keeps the same color scheme as v2)
do_models = [1 2 4 5 7];    %These are now what v2 called model identifiers - Applies at the moment to both make_model_data and check_params;
comment = sprintf('imageTask_matlab_study%02d',which_study);    %The filename will already fill in basic parameters so only use special info for this.
outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\outputs';
%Unfortunately still needs to be typed in manually
filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\outputs\out_new_ll1_bounds_Log0vals_20230910.mat';
%These correspond to identifiers (not configured implementations like in v2) in the v3_sweep version
model_names = {'CO' 'Cs' 'IO' 'BV' 'BR' 'BPM' 'Opt' 'BPV' }; %IO is a placeholder, don't implement
num_model_identifiers = size(model_names,2);
subjects = 1:64; 
IC = 2; %1 if AIC, 2 if BIC
do_io = 1;  %If a 1, will add io performance as a final model field when make_est_model_data is switched to 1.



if check_params == 1;
    
    disp('Getting subject data ...');
    num_subs_found = 0;
    for subject = subjects;
        
        
        
        %sub_fail 1*1, female 1*1, mean_ratings num_stim*1, seq_vals seqs*opts, output seqs*1
        %Note that ratings data are scaled 1 to 100 by this function.
        [sub_fail female mean_ratings seq_vals output] = get_sub_data_matlab(subject,which_study);
        
        if sub_fail == 0;
            continue;
        end;
        num_subs_found = num_subs_found + 1;
        
        %Get ranks
        clear seq_ranks ranks;
        seq_ranks = tiedrank(seq_vals')';
        for i=1:size(seq_ranks,1);
            ranks(i,1) = seq_ranks(i,output(i,1));
        end;    %loop through sequences to get rank for each
        
        if log_or_not == 1;
            Generate_params.ratings(:,num_subs_found) = log(mean_ratings');
            Generate_params.seq_vals(:,:,num_subs_found) = log(seq_vals);
        else;
            Generate_params.ratings(:,num_subs_found) = mean_ratings';
            Generate_params.seq_vals(:,:,num_subs_found) = seq_vals;
        end;
        Generate_params.num_samples(:,num_subs_found) = output;
        Generate_params.ranks(:,num_subs_found) = ranks;
        
    end;
    
    %Now that you have info on the subs, load up the main struct with all
    %the basic info you might need
    %     Generate_params.IC = IC;    %AIC (0) or BIC (1) correction?
    Generate_params.log_or_not = log_or_not;
    Generate_params.all_draws_set = all_draws_set;
    Generate_params.do_models_identifiers = do_models;
    Generate_params.num_subs =  size(Generate_params.seq_vals,3);
    Generate_params.num_seqs =  size(Generate_params.seq_vals,1);
    Generate_params.seq_length =  size(Generate_params.seq_vals,2);
    Generate_params.num_vals = size(Generate_params.ratings,1);
    Generate_params.rating_bounds = [1 100];    %Either Gorilla so already 1 to 100 or the function that gets the matlab behavioural data rescaled it to be 1 to 100
    if log_or_not == 1;
        Generate_params.rating_bounds = log(Generate_params.rating_bounds);
    end;
    Generate_params.BVrange = Generate_params.rating_bounds;
    Generate_params.nbins_reward = numel(Generate_params.rating_bounds(1):Generate_params.rating_bounds(2));  %This should effectuvely remove the binning
    Generate_params.binEdges_reward = ...
        linspace(...
        Generate_params.BVrange(1) ...
        ,Generate_params.BVrange(2)...
        ,Generate_params.nbins_reward+1 ...
        );   %organise bins by min and max
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%SET UP MODELS !!!!!!%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %We build off of (Or rather use part of) the infrastructure I created for configuring models in
    %Param_recover*.m. It involves a template list of default parameters
    %values that is then repmatted into a parameter*model type matrix of
    %default parameters. Then a matching parameter*model type free_parameters
    %matrix marks which parameters to fit down below.These matrices are
    %then used to populate separate model fields in Generate_params, which
    %is then dropped into the estimation function.
    
    %Make the template parameter list
    opt_rule = ceil(exp(-1)*Generate_params.seq_length);  %37% rule cutoff
    
    model_template.identifier = 2;    %row 1 in param_config, 1:CO 2:IO 3:Cs 4:BV 5:BR 6:BP 7:optimism 8:BPV
    model_template.kappa = 2;        %row 2 in param_config
    model_template.nu = 1;           %row 3
    model_template.cutoff = opt_rule;     %row 4, initialised to optimal 37%
    model_template.Cs = 0;            %row 5, intiialised to optimal no cost to sample
    model_template.BVslope = 0.2;   %row 6, intialised to 1 (like a threshold)
    model_template.BVmid = 55;      %row 7, initialised to halfway through the rating scale (can't be used with log)
    model_template.BRslope = 1;    %row 8
    model_template.BRmid = 55;      %row 9
    model_template.BP = 0;           %row 10
    model_template.optimism = 0;    %row 11
    model_template.BPV = 0;          %row 12
    %     model_template.log_or_not = log_or_not; %1 = log transform (normalise) ratings  %row 13 (This is already legacy - log or not is now controlled by switch at start of programme and simulated data was logged before reaching this point).
    model_template.all_draws = all_draws_set;  %1 = use all trials when computing ll instead of last two per sequence.   %row 14
    model_template.beta = 1;        %Just for parameter estimation.
    model_template.name = 'template';
    
    %Correct the starting parameters that are in units of ratings
    if log_or_not == 1;
        model_template.BVmid = log(model_template.BVmid);
        model_template.BRmid = log(model_template.BRmid);
        %         model_template.BP = log(model_template.BP);
        %         model_template.optimism = log(model_template.optimism);
        %         model_template.BPV = log(model_template.BPV);
    end;
    
    
    %Repmat the template to create a column for each model. For now, we are
    %doing all possible models, not the ones specified in do_models. We'll
    %reduce this matrix to just those below.
    identifiers = 1:num_model_identifiers;
    num_cols = num_model_identifiers;
    param_config_default = [ ...
        identifiers;    %row 1: identifiers,  1:CO 2:IO 3:Cs 4:BV 5:BR 6:BP 7:optimism 8:BPV
        repmat(model_template.kappa,1,num_cols);   %row 2: kappa
        repmat(model_template.nu,1,num_cols);   %row 3: nu
        repmat(model_template.cutoff,1,num_cols)   %row 4: cutoff
        repmat(model_template.Cs,1,num_cols);   %row 5: Cs
        repmat(model_template.BVslope,1,num_cols);        %row 6: BV slope
        repmat(model_template.BVmid,1,num_cols);       %row 7: BV mid
        repmat(model_template.BRslope,1,num_cols);        %row 8: BR slope
        repmat(model_template.BRmid,1,num_cols);       %row 9: BR mid
        repmat(model_template.BP,1,num_cols);        %row 10: prior mean offset (BP)
        repmat(model_template.optimism,1,num_cols);       %row 11: optimism
        repmat(model_template.BPV,1,num_cols);       %row 12: prior variance offset (BPV)
        %         repmat(model_template.log_or_not,1,num_cols);   %row 13: log or not (at the moment not to be trusted)
        repmat(model_template.all_draws,1,num_cols);   %row 14: all draws
        repmat(model_template.beta,1,num_cols);   %row 15: beta
        ];
    
    %Mark which are free/to be estimated
    free_parameters = zeros(size(param_config_default));
    free_parameters(4,1) = 1; %Model indicator 1, parameter 4: Cut off
    free_parameters(5,2) = 1;  %Model indicator 2, parameter 5: Cs
    free_parameters(7,4) = 1;   %Model indicator 4, parameter 7: BV
    free_parameters(9,5) = 1;     %Model indicator 5, parameter 9: BR
    free_parameters(10,6) = 1;  %Model indicator 6, parameter 10: BPM
    free_parameters(11,7) = 1;  %Model indicator 7, parameter 11: Opt
    free_parameters(12,8) = 1;  %Model indicator 8, parameter 12: BPV
    
    %Now reduce matrices to just those in do_models
    %In Param_recover*.m we had distrinctions between model instantiations
    %and
    param_config_default = param_config_default(:,do_models);
    free_parameters = free_parameters(:,do_models);
    
    %Save your work into struct
    Generate_params.num_models = numel(do_models);
    Generate_params.param_config_default = param_config_default;
    Generate_params.free_parameters_matrix = free_parameters;
    Generate_params.comment = comment;
    Generate_params.outpath = outpath;
    analysis_name = sprintf(...
        'out_new_ll%d_%s_'...
        , Generate_params.all_draws_set ...
        , Generate_params.comment ...
        );
    Generate_params.analysis_name = analysis_name;
    outname = [analysis_name char(datetime('now','format','yyyyddMM')) '.mat'];
    Generate_params.outname = outname;
    
    disp( sprintf('Running %s', outname) );
    
    %Now fill in default parameters to model fields
    for model = 1:Generate_params.num_models;   %How many models are we implementing (do_models)?
        
        %         Generate_params.current_model = Generate_params.do_models_identifiers(model);  %So now model 1 will be the first model implementation in the param_config array after it has been reduced by do_models
        
        it = 1;
        fields = fieldnames(model_template);
        for field = 1:size(fields,1)-1 %exclude name, the last one
            Generate_params.model(model).(fields{field}) = param_config_default(field,model);
            it=it+1;
        end;
        Generate_params.model(model).name = ...
            model_names{...
            Generate_params.model(model).identifier...
            };  %I think this is the only matrix here that hasn't already been reduced to do_models in the preceding step
        
        %Fill in this model's free parameters to be estimated later, if you
        %get to the parameter estimatioin this run
        Generate_params.model(model).this_models_free_parameters = find(free_parameters(:,model)==1);
        Generate_params.model(model).this_models_free_parameter_default_vals = param_config_default(find(free_parameters(:,model)==1),model)';
        
    end;    %loop through models
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%DO THE MODEL FITTING!!!!!!%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %set upper and lower bounds for the different models (A bit
    %disorganised to put it here, I admit, but this is a new section of
    %code grafted onto a much older structure so this bit will look a little more ad hoc
    
    %I'll just specify all the bounds together here and then assign them to
    %Generate_params model structures and pass them to fminsearch in the
    %model loop immediately below
    fitting_bounds.CO = [1 Generate_params.seq_length];   %cut off, it's a threshold that must be inside sequence
    fitting_bounds.Cs = [-Inf Inf];   %cost to sample. In practice will be between -1 and 1 but not good theoretical reason to constrain
    fitting_bounds.BV = [1 Generate_params.seq_length]; %biased values (O), it's a threshold that must be inside sequence
    fitting_bounds.BR = [1 Generate_params.seq_length]; %biased reward (O), it's a threshold that must be inside sequence
    fitting_bounds.BP = [0 100];  %biased prior, value can't exist outside the rating scale
    fitting_bounds.Opt = [0 100];  %optimism, value can't exist outside the rating scale
    fitting_bounds.BV = [0 100];  %biased variances, can't be wider than the whole rating scale
    fitting_bounds.beta = [0 100];   %if not sometimes produces crazy values in the thousands, though rarely
    

    %In the previous section, we only assigned to the Generate_param
    %struct the models that were in the do_models in that section. So
    %this can only operate on those models (until I start implementing these sections from datafiles later).
    for model = 1:numel(Generate_params.do_models_identifiers);
        
        for sub = 1:Generate_params.num_subs;
            
            
            %You want to fit one model for one subject at a time
            %             Generate_params.current_model = Generate_params.do_models_identifiers(model);
            Generate_params.current_model = model;
            Generate_params.num_subs_to_run = sub;
            
            %Use default params as initial values
            params = [ ...
                Generate_params.model(model).this_models_free_parameter_default_vals ...
                Generate_params.model(model).beta ...
                ];
            
            %Assign upper and lower bounds
            test_name = Generate_params.model( Generate_params.current_model ).name;
            Generate_params.model(model).lower_bound = eval(sprintf('fitting_bounds.%s(1)',test_name));
            Generate_params.model(model).upper_bound = eval(sprintf('fitting_bounds.%s(2)',test_name)); 
            Generate_params.model(model).lower_bound_beta = fitting_bounds.beta(1);
            Generate_params.model(model).upper_bound_beta = fitting_bounds.beta(2);
            
            warning('off');
            
            disp(...
                sprintf('fitting modeli %d name %s subject %d' ...
                , model ...
                , Generate_params.model( model ).name ...
                , sub ...
                ) );
            
            %%%%%%%%%%%%%%%%%%%%%%%%
            %%%%Main function call in this section
            %%%%%%%%%%%%%%%%%%%%%%%%%%
            [Generate_params.model(model).estimated_params(sub,:) ...
                ,  Generate_params.model(model).ll(sub,:) ...
                , exitflag, search_out] = ...
                fminsearchbnd(  @(params) f_fitparams( params, Generate_params ), ...
                    params,...
                    [Generate_params.model(model).lower_bound Generate_params.model(model).lower_bound_beta], ...
                    [Generate_params.model(model).upper_bound Generate_params.model(model).upper_bound_beta] ...
                    );
    
        end;    %Loop through subs
        
        %Should save after each model completed
        save([Generate_params.outpath filesep Generate_params.outname], 'Generate_params');
        
    end;   %loop through models
    
end;    %estimate parameters of simulated data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Generate performance from estimated parameters!%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if make_est_model_data == 1;
    
    if check_params == 0 & use_file_for_plots == 1;   %use the saved file, if analysis is not succeeding from model fitting
        
        %should create Generate_params in workspace
        load(filename_for_plots,'Generate_params');
        Generate_params.do_io = do_io;
        
    end;
    
        %Run ideal observer if configured to do so
    if Generate_params.do_io == 1;
        
        Generate_params = run_io(Generate_params);
        
    end;
    
    
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
        
        %theoretical parameters
        Generate_params.model(Generate_params.current_model).(fields{field}) = ...
            Generate_params.model(Generate_params.current_model).estimated_params(sub,it);
        
        it=it+1;
        
    end;
    
    %...and beta (assume beta is last, after all of the (it = one)
    %theoretical parameters
    Generate_params.model(Generate_params.current_model).beta = ...
        Generate_params.model(Generate_params.current_model).estimated_params(sub,end);
    
    disp(...
        sprintf('computing performance, fitted modeli %d name %s subject %d' ...
        , Generate_params.current_model ...
        , Generate_params.model( Generate_params.current_model ).name ...
        , sub ...
        ) );
    
    Generate_params.num_subs_to_run = sub;
    [num_samples(sub,:) ranks(sub,:)] = generate_a_models_data(Generate_params);
    
end;    %Loop through subs

fprintf(' ');








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
    
    %How many samples for this sequence and subject
    listDraws = ...
        Generate_params.num_samples(seq,Generate_params.num_subs_to_run);
    
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
        
        list.allVals = squeeze(Generate_params.seq_vals(sequence,:,num_subs_found));
        Generate_params.PriorMean = mean(Generate_params.ratings(:,num_subs_found));
        Generate_params.PriorVar = var(Generate_params.ratings(:,num_subs_found));
        
        %ranks for this sequence
        dataList = tiedrank(squeeze(Generate_params.seq_vals(sequence,:,num_subs_found))');
    
        list.vals =  list.allVals;
        
        %Do cutoff model, if needed
        if Generate_params.model(Generate_params.current_model).identifier == 1;
            
            %get seq vals to process
            this_seq_vals = list.allVals;
            %initialise all sequence positions to zero/continue (value of stopping zero)
            choiceStop = zeros(1,Generate_params.seq_length);
            %What's the cutoff?
            estimated_cutoff = round(Generate_params.model(Generate_params.current_model).cutoff);
            if estimated_cutoff < 1; estimated_cutoff = 1; end;
            if estimated_cutoff > Generate_params.seq_length; estimated_cutoff = Generate_params.seq_length; end;
            %find seq vals greater than the max in the period
            %before cutoff and give these candidates a maximal stopping value of 1
            choiceStop(1,find( this_seq_vals > max(this_seq_vals(1:estimated_cutoff)) ) ) = 1;
            %set the last position to 1, whether it's greater than
            %the best in the learning period or not
            choiceStop(1,Generate_params.seq_length) = 1;
            
            %Reverse 0s and 1's for ChoiceCont
            choiceCont = double(~choiceStop);
            
%             ranks(sequence,this_sub) = dataList( num_samples(sequence,this_sub) );
            
            
        else;   %Any Bayesian models
            
            [choiceStop, choiceCont, difVal]  = ...
                analyzeSecretaryNick_2021(Generate_params,list);

        end;    %Cutoff or other model?
            
%             num_samples(sequence,this_sub) = find(difVal<0,1,'first');  %assign output num samples for Bruno model
            
            choiceValues = [choiceCont; choiceStop]';
            
            b = Generate_params.model( Generate_params.current_model ).beta;
            
            %softmax the action values, using this sub's estimated beta
            for drawi = 1 : Generate_params.seq_length
                %cprob seqpos*choice(draw/stay)
                cprob(drawi, :) = exp(b*choiceValues(drawi, :))./sum(exp(b*choiceValues(drawi, :)));
            end;

            cprob(end,2) = Inf; %ensure stop choice on final sample.
            
            %Now get samples from uniform distribution
            test = rand(1000,Generate_params.seq_length);
            for iteration = 1:size(test,1);
                
                samples_this_test(iteration) = find(cprob(:,2)'>test(iteration,:),1,'first');
                ranks_this_test(iteration) = dataList( samples_this_test(iteration) );
                
            end;    %iterations
            
            num_samples(sequence,this_sub) = round(mean(samples_this_test));
            ranks(sequence,this_sub) = round(mean(ranks_this_test));
            
        %...and its rank
        %         ranks(sequence,this_sub) = dataList( num_samples(sequence,this_sub) );
        %Accumulate action values too so you can compute ll outside this function if needed
        choiceStop_all(sequence, :, this_sub) = choiceStop;
        choiceCont_all(sequence, :, this_sub) = choiceCont;
        
    end;    %loop through sequences
    
    this_sub = this_sub + 1;
    
end;    %loop through subs



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sub_fail rated_female mean_ratings seq_vals draws] = get_sub_data_matlab(subject, which_study);

if which_study == 1;    %av/face 1
    
    data_folder = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\face_study_1_av';
    rating_files = dir( [data_folder filesep sprintf('av_ratings_sub%02d*.mat',subject)] );
    seq_files = dir( [data_folder filesep sprintf('aav_sequence_sub%02d*.mat',subject)] );
    
elseif which_study == 2; %matchmaker
    
    data_folder = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\match_maker_data';
    rating_files = dir( [data_folder filesep sprintf('av_ratings_sub%02d*.mat',subject)] );
    seq_files = dir( [data_folder filesep sprintf('av_sequence_sub%02d*.mat',subject)] );
    
elseif which_study == 3;    %small trust / trust 1
    
    data_folder = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\trust_study_1_small';
    rating_files = dir( [data_folder filesep sprintf('av_ratings_sub%03d*.mat',subject)] );
    seq_files = dir( [data_folder filesep sprintf('av_sequence_sub%03d*.mat',subject)] );
    
end;    %which study?

%In case I need to break
sub_fail = 0;
rated_female = [];
mean_ratings = [];
seq_vals = [];
output = [];
draws = [];


num_rating_files = [];
if ~isempty(rating_files);  %is there a subject?
    num_rating_files = size( rating_files, 1); %how many ratings (there's usually three)?
else
    disp(sprintf('there is no subject %d',subject));
    return;   %skip to next sub if this sub has no files
end;

%rated_female is 1 if female face, 0 if male
if ~isempty( strfind(rating_files(1).name,'sex01') );
    rated_female = 1;
elseif ~isempty( strfind(rating_files(1).name,'sex00') );
    rated_female = 0;
end;

%get and store ratings data
for file = 1:num_rating_files;
    
    clear output;   %just in case
    load( [data_folder filesep rating_files(file).name],'output'); %columns: trial_num, filename, face_num, rating
    output = sortrows(output,3); %sort output by face_num
    
    %correct cogent codes
    ratings( find(cell2mat(output(:,4)) > 70),file) = cell2mat( output(find(cell2mat(output(:,4)) > 70),4) ) - 75;  %if number pad
    ratings( find(cell2mat(output(:,4)) < 70),file) = cell2mat( output(find(cell2mat(output(:,4)) < 70),4) ) - 27;  %if top of keyboard numbers
    
    %NaN other weird values
    if numel(find( ratings(:,file) < 1 ))>0 | numel(find( ratings(:,file) > 9 ))>0
        ratings( find( ratings(:,file) < 1 ), file ) = NaN;    %in case subject presses wrong key, log will choke later
        ratings( find( ratings(:,file) > 9 ), file ) = NaN;    %in case subject presses wrong key
    end;    %if any residual weird keypresses
    
end;    %loop through ratings files

mean_ratings = nanmean( ratings' )';    %raw ratings
old_min = 1;
old_max = 9;
new_min = 1;
new_max = 100;
mean_ratings = ((new_max-new_min)/(old_max-old_min))*(mean_ratings - old_max)+100;

%now load sequence data for this subject
load( [data_folder filesep seq_files.name], 'rows_data','output' );  %get the list of face numbers in the sequences

%Get numbers of draws
draws = cell2mat(output( find( cell2mat(output(:,6))==29 ), 3 ));  %draws

%need the list of values for each facenumber
seq_vals = zeros(size(rows_data));
for i=1:numel(rows_data);
    seq_vals(i) = mean_ratings( rows_data(i) );     %raw values
end;

sub_fail = 1;








function Generate_params = run_io(Generate_params);

for sub = 1:Generate_params.num_subs;
    
    disp(...
        sprintf('computing performance, ideal observer subject %d' ...
        , sub ...
        ) );
    
    for sequence = 1:Generate_params.num_seqs;
        
        clear sub_data;
        
        prior.mu =  mean(log(Generate_params.ratings(:,sub)));
        prior.sig = var(log(Generate_params.ratings(:,sub)));
        prior.kappa = 2;
        prior.nu = 1;
        
        list.flip = 0;
        list.vals = log(Generate_params.seq_vals(sequence,:,sub));
        list.length = size(list.vals,2);
        list.optimize = 0;
        params = 0; %Cs
        [choiceStop, choiceCont, difVal] = ...
            analyzeSecretaryNick3_test(prior,list,0,0,0);
        
        samples(sequence,sub) = find(difVal<0,1,'first');
        
        %rank of chosen option
        dataList = tiedrank(squeeze(Generate_params.seq_vals(sequence,:,sub))');    %ranks of sequence values
        ranks(sequence,sub) = dataList(samples(sequence,sub));
        
    end;    %sequence loop
    
end;    %sub loop

%add new io field to output struct
num_existing_models = size(Generate_params.model,2);
Generate_params.model(num_existing_models+1).name = 'Optimal';
Generate_params.model(num_existing_models+1).num_samples_est = samples;
Generate_params.model(num_existing_models+1).ranks_est = ranks;

fprintf('');
