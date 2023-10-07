
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = new_fit_code_av_2022_betafix;

%_betafix incorporates estimated beta into simulation of sampling rate

%new_fit_code_av_2022: Creates results for av study model fitting. 2022
%revamps it slightly to make plots more figure-worthy.

%v2: I tried to introduce log_or_not functionality

% new_fit_code_big_trust: I am now modifying my sweep code to fit
% parameters to human participants data with the biggest dataset that we
% have. Big trust.

%v3_sweep isn't an improvemnt on v2 buit starts a new branch. V2 (and its
%successors if any) will continue to run preconfigured models that simulate
%over versus undersampling. The v3_sweep branch will sweep across parameter
%values for each model and plot configured versus estimated [parameters and
%performance).

%v3 note: I have a bad habit of using identifier and indicator
%interchangeably.

%v2: I changed the biased values and biased rewards models to be single
%parameter (Just threshold with a fixed high slope to resemble a sharp
%threshold). Neither all draws nor last draw seem to vary both parameters
%as configured but mainly varies only slope, with some exception. So number
%of models changes

%v2: Also added to v1 some more visualisatioin tools: scatterplots to compare estimated anbd configured
%performance subject by subject and modifications to parameter visuation figure

%V1: originally implemented slope and threshold versions of BV and BR.

tic

%The idea was to save data in files along the way as needed. But as
%implemented now, later sections are dependent on previous ones.
check_params = 0;       %fit the same model that created the data and output estimated parameters
make_est_model_data = 1;
use_file_for_plots = 1; %Set the above to zero and this to 1 and it'll read in a file you specify (See filename_for_plots variable below) and make plots of whatever analyses are in the Generate_params structure in that file;
make_plots = 1;         %if 1, plots the results
all_draws_set = 1;          %You can toggle how the ll is computed here for all models at once if you want or go below and set different values for different models manually in structure
log_or_not = 0; %I'm changing things so all simulated data is logged at point of simulation (==1) or not
%1: cutoff 2: Cs 3: dummy (formerly IO in v2) 4: BV 5: BR 6: BPM 7: Opt 8: BPV
%(I keep model 3 as a legacy for IO because analyseSecertaryNick_2021
%looks for identifiers 4 and 5 for BV and BR and needs that for v2. Also it keeps the same color scheme as v2)
do_models = [1 2 4 5 7];    %These are now what v2 called model identifiers - Applies at the moment to both make_model_data and check_params;
comment = sprintf('av_betafix_Log%dvals%d',log_or_not);;    %The filename will already fill in basic parameters so only use special info for this.
outpath = 'C:\matlab_files\fiance\parameter_recovery\outputs';
%Unfortunately still needs to be typed in manually
filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_av_20212906.mat';
%These correspond to identifiers (not configured implementations like in v2) in the v3_sweep version
model_names = {'Cut off' 'Cs' 'IO' 'BV' 'BR' 'BPM' 'Opt' 'BPV' }; %IO is a placeholder, don't implement
num_model_identifiers = size(model_names,2);
subjects = 1:64;    %big trust sub nums
% subjects = 1;    %big trust sub nums
IC = 2; %1 if AIC, 2 if BIC
analyze_value_positions = 1;    %Create plots with psychometric curves, their thresholds (model fits) and their correlations (nbins_psi hardwired at function call)

if check_params == 1;
    
    disp('Getting subject data ...');
    num_subs_found = 0;
    for subject = subjects;
        
        %sub_fail 1*1, female 1*1, mean_ratings num_stim*1, seq_vals seqs*opts, output seqs*1
        %Note that ratings data are scaled 1 to 100 by this function.
        [sub_fail female mean_ratings seq_vals output] = get_sub_data_matlab(subject);
        
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
    Generate_params.analyze_value_positions = analyze_value_positions;  %make psychometric plots if a 1
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
    
    %         %How many models to do in this section?
    %     do_models = Generate_params.do_models_identifiers;
    
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
            
            %                 options = optimset('Display','iter');
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
                fminsearch(  @(params) f_fitparams( params, Generate_params ), params);
            
            %                     fminsearch(  @(params) f_fitparams( params, Generate_params ), params, options);
            
            
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




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%PLOT!%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if make_plots == 1;
    
    %Use data already in workspace because computed above by or ?
    %Or open a file where these things were computed on previous run?
    if use_file_for_plots == 1;
        
        load(filename_for_plots,'Generate_params');
        
    end;    %Plot data structure from a file?
    
    Generate_params.IC = IC;    %AIC (0) or BIC (1) correction?
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
        
        %         if Generate_params.model(Generate_params.current_model).log_or_not == 1;
        %             Generate_params.binEdges_reward = ...
        %                 linspace(log(Generate_params.rating_bounds(1)),log(Generate_params.rating_bounds(2)),Generate_params.nbins_reward+1);   %organise bins by min and max
        %         Generate_params.PriorMean = mean(log(Generate_params.ratings(:,num_subs_found)));
        %         Generate_params.PriorVar = var(log(Generate_params.ratings(:,num_subs_found)));
        %             Generate_params.BVrange = log(Generate_params.rating_bounds);   %Used for normalising BV
        %             list.allVals = log(squeeze(Generate_params.seq_vals(sequence,:,num_subs_found)));
        %         else
        
        %             Generate_params.BVrange = Generate_params.rating_bounds;    %Used for normalising BV
        list.allVals = squeeze(Generate_params.seq_vals(sequence,:,num_subs_found));
        Generate_params.PriorMean = mean(Generate_params.ratings(:,num_subs_found));
        Generate_params.PriorVar = var(Generate_params.ratings(:,num_subs_found));
        %         end;
        
        %ranks for this sequence
        dataList = tiedrank(squeeze(Generate_params.seq_vals(sequence,:,num_subs_found))');
        
        %         list.optimize = 0;
        %         list.flip = 1;
        list.vals =  list.allVals;
        %         list.length = Generate_params.seq_length;
        
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
            %find first index that is a candidate ....
%             num_samples(sequence,this_sub) = find(choiceStop == 1,1,'first');   %assign output num samples for cut off model
            
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
function [sub_fail rated_female mean_ratings seq_vals draws] = get_sub_data_matlab(subject);

%sub_fail indicates whether full data could be extracted from this
%subject or whether subject should be skipped in outer loop that
%calls get_sub_data. Can be used to decide whether to increment
%num_subs_found if you outout 1 for success and 0 for fail;

data_folder = 'C:\matlab_files\fiance\av_data';
rating_files = dir( [data_folder filesep sprintf('av_ratings_sub%02d*.mat',subject)] );
seq_files = dir( [data_folder filesep sprintf('aav_sequence_sub%02d*.mat',subject)] );

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
%new to this version
%scale them to be between 1 and 100 (so all studies will be on the same scale)
% min_ratings = min(mean_ratings);   %what is new min after trandform?
% max_ratings = max(mean_ratings);   %what is new max after transform?
% mean_ratings = (mean_ratings - min_ratings)/(max_ratings - min_ratings);
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





function analyze_value_position_functions(value_data,choice_trial,plot_cmap,binEdges_psi,legend_labels,param_to_fit,two_params);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Look at proportion choice, position and value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param_to_fit = [0 param_to_fit];    %the first model in this function is subject so add it to this list as zeroth model so it can be indexed (e.g., in colormaps)

%value_data was log(raw_seq_subs), contains ratings data in sequences and is seq*position*sub
nbins = size(binEdges_psi,2)-1;
num_subs = size(value_data,3);
num_positions = size(value_data,2);
num_seqs = size(value_data,1);
num_models = size(choice_trial,3);
serial_r = value_data;  %only for ratings plots, nothing else

f_a = 0.1; %face alpha
sw = 0.5;  %ppoint spread width
font_size = 12;

%ok. this time, let's be less efficient but more organised. I want to bin
%things right up front before anything so there is a ratings dataset
%(value_data) and a binned dataset (value_bins)
for sub = 1:num_subs;
    binEdges = binEdges_psi(sub,:);
    [dummy,value_bins(:,:,sub)] = histc(value_data(:,:,sub), [binEdges(1:end-1) Inf]);
end;

%nan mask has zeros for view decisions and 1s for take decisions and NaNs
%when no face was seen because of elapsed decision. Can use to mask other
%arrays later ...
nan_mask = NaN(num_seqs,num_positions,num_subs,num_models);
for model=1:num_models;
    for sub=1:num_subs;
        for seq=1:num_seqs;
            nan_mask(seq,1:choice_trial(sub,seq,model),sub,model) = 0;
            nan_mask(seq,choice_trial(sub,seq,model),sub,model) = 1;
        end;
    end;
end;

%now we have two seq*position*subject arrays, one ratings, one bins, now
%make supersubject seq*position versions by concatenating subjects. Effectively, each
%new subject justs adds new sequences so its a long list of sequences
value_data_sub = [];
value_bins_sub = [];
choice_trial_sub = [];

for sub=1:num_subs;
    value_data_sub = [value_data_sub; value_data(:,:,sub)];
    value_bins_sub = [value_bins_sub; value_bins(:,:,sub)]; %bins are still subject specific
    choice_trial_sub = [choice_trial_sub; squeeze(choice_trial(sub,:,:))];
end;

nan_mask_sub = [];
for model=1:num_models;
    temp = [];
    for sub=1:num_subs;
        temp = [temp; squeeze(nan_mask(:,:,sub,model))];
    end;
    nan_mask_sub(:,:,model) = temp;
end;

%yes, it's yet another subject loop. I'm being modular. This one prepares
%the average ratings * serial position data. It also computes the proportion choices *serial position data.
%It also computes proportion subject predicted * serial position data
%This one needed whether using a super subject or fitting all subjects, it's a separate analysis
model_predicted_choices = NaN(num_subs,num_positions,num_models-1); %for proportion correctly predicted subject choices
position_choices = NaN(num_subs,num_positions,num_models);   %for proportion responses serial positions
position_function = zeros(num_positions,num_subs,num_models);   %for average ratings as function of serial position
position_it = zeros(num_positions,num_subs,num_models);         %for average ratings as function of serial position
for sub=1:num_subs;
    
    this_subject_ratings = squeeze(serial_r(:,:,sub));  %only for plotting ratings by serial position
    
    for position=1:num_positions; %loop through the positions
        for model=1:num_models;
            
            sub_choices_this_position = nan_mask(:,position,sub,1);
            model_choices_this_position = nan_mask(:,position,sub,model);
            % %             %computes proportion responses for each position
            position_choices(sub,position,model) = sum( choice_trial(sub,:,model) == position )/size(choice_trial,2);
            
            %find average attractiveness of the choices in each position
            this_subject_choices = squeeze(choice_trial(sub,:,model));
            indices_into_choices = find(this_subject_choices==position);
            if ~isempty(indices_into_choices);
                for i=1:size(indices_into_choices,2);
                    position_function(position,sub,model) = position_function(position,sub,model)+this_subject_ratings(indices_into_choices(i),position);
                    position_it(position,sub,model) = position_it(position,sub,model)+1;
                end;    %loop through values for this position
            end;    %is there a valid position value?
            
        end;    %model
    end;        %position
end;            %subject

%individual subs
position_data_indiv = NaN(nbins,num_positions,num_subs,num_models);            %for value function analyses as function of serial position
ave_rating_per_bin_indiv = NaN(nbins,num_positions,num_subs,num_models);       %use this for curve fitting later

for sub=1:num_subs;
    
    for position=1:num_positions; %loop through the positions
        for model=1:num_models;
            
            this_subject_bins = value_bins(:,:,sub);
            temp2 = squeeze(nan_mask(:,:,sub,model));
            this_subject_bins(isnan(temp2(:)))=NaN;
            
            for val_bin = 1:nbins;
                
                %find bins at this position and,if any, check what are the CHOICES and RATINGS for that bin/position
                trials_with_bins_in_this_position = [];
                trials_with_bins_in_this_position = find( this_subject_bins(:,position) == val_bin );   %on which sequences did a value in this bin occur in this position?
                num_trial_with_vals = numel(trials_with_bins_in_this_position);                     %how many sequences have this value in this position?
                position_data_indiv(val_bin,position,sub,model) = sum(choice_trial(sub,trials_with_bins_in_this_position,model)==position)/ num_trial_with_vals ; %Now I need the number of CHOICES for this positon and bin
                ave_rating_per_bin_indiv(val_bin,position,sub,model) = nanmean(value_data(trials_with_bins_in_this_position,position,sub));   %need this for regression later (no sense of model here)
                
            end;    %value bin
        end;    %model
    end;        %position
end;            %subject


position_data = position_data_indiv;
ave_rating_per_bin = ave_rating_per_bin_indiv;

%loop again, this time through positions and fit averages over subjects
for model=1:num_models;
    for position=1:num_positions;
        
        %computes value slopes for each position, and model
        this_position_no_subs = nanmean(squeeze( position_data(:,position,:,model) ),2);   %returns bin values for a subject in a position
        
        y_this_position = this_position_no_subs(~isnan(this_position_no_subs));
        x_this_position = [1:nbins];
        x_this_position = x_this_position(~isnan(this_position_no_subs))';
        clear f_position;
        if numel(x_this_position)<3 | position==num_positions | sum(this_position_no_subs) == 0;    %if there are too many nans and not enough datapoints, if its that last position with the flat line or all ones, or if no response was ever made
            
            b1(position,model) = NaN;
            b2(position,model) = NaN;
        else
            %             f_position=fit(x_this_position,y_this_position,'1./(1+exp(-p1*(x-p2)))','StartPoint',[1 5],'Lower',[0 1],'Upper',[Inf 8]);
            %             temp_coef = coeffvalues(f_position);
            %             b1(position,model) = temp_coef(1);  %if only slope and mid are free
            %             b2(position,model) = temp_coef(2); %if only slope and mid are free
            if two_params == 1;
                %             Two params free
                f_position=fit(x_this_position,y_this_position,'1./(1+exp(-p1*(x-p2)))','StartPoint',[1 5],'Lower',[0 1],'Upper',[Inf 8]);
                temp_coef = coeffvalues(f_position);
                b1(position,model) = temp_coef(1);  %if only slope and mid are free
                b2(position,model) = temp_coef(2); %if only slope and mid are free
                
            else
                
                % %             %Three params free
                f_position=fit(x_this_position,y_this_position,'p1./(1+exp(-p3*(x-p4)))','StartPoint',[1 1 5],'Lower',[0 0 1],'Upper',[1 Inf 8]);
                temp_coef = coeffvalues(f_position);
                b1(position,model) = temp_coef(2);  %if only slope and mid are free
                b2(position,model) = temp_coef(3); %if only slope and mid are free
            end;
            
        end;    %check is there enough data to do a fit
        
    end;    %loop through positions
end;    %loop through models

b_ci = zeros(size(b2));
b = b2;

%%%%%%%new part: correlation position data for each model with subjects
r_graph = zeros(1,num_models);
r_ci_graph = zeros(1,num_models);
for model = 1:num_models;
    
    if model ~=1;
        
        for sub=1:num_subs;
            
            clear this_subject_data this_model_data this_subject_data_rs this_model_data_rs
            
            %extract_data
            this_subject_data = squeeze(position_data(:,:,sub,1));
            this_model_data = squeeze(position_data(:,:,sub,model));
            
            %reshape data
            this_subject_data_rs = reshape(this_subject_data,prod(size(this_subject_data)),1);
            this_model_data_rs = reshape(this_model_data,prod(size(this_model_data)),1);
            
            %correlate them
            [temp1 temp2] = corrcoef(this_subject_data,this_model_data,'rows','complete');
            r(sub,model-1) = temp1(2,1);
            p(sub,model-1) = temp2(2,1);
            sub_nums(sub,model-1) = sub;
            mod_nums(sub,model-1) = model-1;
            
        end;    %loop through subs
        
    end;    %only consider models other than subjects
    
end;    %models

r_graph = [0 nanmean(r,1)];
r_ci_graph = [0 1.96*(nanstd(r)/sqrt(size(r,1)))];


%average proportion responses, over subjects
mean_position_choices = squeeze(mean(position_choices,1));
ci_position_choices = squeeze(1.96*(std(position_choices,1,1)/sqrt(size(position_choices,1))));
%average ratings as function of serial position
clear ave_ratings ave ci;
ave_ratings = position_function./position_it;

%serial postion plots: average rating, proportion correct and value sensitivity slopes
h3 = figure; set(gcf,'Color',[1 1 1]);  %For serial position/PSE plots/correlation plots
h4 = figure; set(gcf,'Color',[1 1 1]);  %For psychometric function plots
for model = 1:size(choice_trial,3);
    
    markersize = 3;
    
    %average rating as function of serial positon
    legend_locs = [0.5:-0.05:(0.5 - (0.05*5))];
    %
    %     %proportion choices
    %     figure(h3); subplot( 2,2,3); hold on;
    %     sph = shadedErrorBar(1:size(mean_position_choices,1),mean_position_choices(:,model),ci_position_choices(:,model),{'MarkerFaceColor',plot_cmap(param_to_fit(model)+1,:),'MarkerEdgeColor',plot_cmap(param_to_fit(model)+1,:),'Marker','o','MarkerSize',markersize,'LineStyle','-'},1); hold on;
    %     set(sph.mainLine,'Color',plot_cmap(param_to_fit(model)+1,:));
    %     set(sph.patch,'FaceColor',plot_cmap(param_to_fit(model)+1,:));
    %     set(sph.edge(1),'Color',plot_cmap(param_to_fit(model)+1,:));
    %     set(sph.edge(2),'Color',plot_cmap(param_to_fit(model)+1,:));
    %     %         text(3,legend_locs(model),legend_names{model},'Color',plot_cmap(param_to_fit(model)+1,:),'FontSize',12,'FontName','Arial');
    %     box off;
    %     %axis square;
    %     set(gca,'FontSize',12,'FontName','Arial','xtick',[1:size(b,1)],'ytick',[0.1:0.1:0.8],'Ylim',[0 0.5],'Xlim',[1 size(b,1)],'LineWidth',2);
    %     xlabel('Position in Sequence'); ylabel('Proportion Choices');
    %
    %psychometric function parameters
    figure(h3); subplot( 1,2,1 ); hold on;
    legend_positions = [3:-.5:0];
    errorbar(1:size(b,1),b(:,model),b_ci(:,model),'Color',plot_cmap(param_to_fit(model)+1,:),'MarkerFaceColor',plot_cmap(param_to_fit(model)+1,:),'MarkerEdgeColor',plot_cmap(param_to_fit(model)+1,:),'Marker','o','MarkerSize',markersize,'LineStyle','-','LineWidth',1); hold on;
    if model == 1;
        text(2,legend_positions(model),'Participants', 'Color',plot_cmap(1,:), 'FontSize',12,'FontName','Arial');
    else
        text(2,legend_positions(model),legend_labels{model-1}, 'Color',plot_cmap(param_to_fit(model)+1,:), 'FontSize',12,'FontName','Arial');
    end;
    box off;
    set(gca,'FontSize',12,'FontName','Arial','xtick',[1:size(b,1)],'Xlim',[1 size(b,1)],'YLim',[0 num_positions],'LineWidth',2);
    xlabel('Position in sequence'); ylabel('Point of subjective equality');
    
    if model ~=1;   %no subjects
        
        %model correlations with proportion choice
        figure(h3); subplot( 1,2,2 ); hold on;
        %         legend_positions = [1.1:-.05:0];
        
        handles = plotSpread(r(:,model-1), ...
            'xValues',model,'distributionColors',plot_cmap(param_to_fit(model)+1,:),'distributionMarkers','.', 'spreadWidth', sw);
        
        bar(model,r_graph(model), ...
            'FaceColor',plot_cmap(param_to_fit(model)+1,:),'FaceAlpha',f_a,'EdgeColor',[0 0 0] );
        
        % %         text(1.5,legend_positions(model),legend_labels{param_to_fit(model)}, 'Color',plot_cmap(param_to_fit(model)+1,:), 'FontSize',12,'FontName','Arial');
        %          text(1.5,legend_positions(model),legend_labels{model-1}, 'Color',plot_cmap(param_to_fit(model)+1,:), 'FontSize',12,'FontName','Arial');
        %
        set(gca,'FontSize',12,'FontName','Arial', 'XTick',[],'xticklabel',{[]},'YTick',[0:.2:1],'yticklabel',{[0:.2:1]},'LineWidth',2);
        ylabel('Model-participant correlation');
        ylim([0 1.0]);
        xlim([1 numel(r_graph)+0.5]);
        
        x_axis_test_offset = .05;
        this_offset = -x_axis_test_offset*diff(ylim);
        text( model, this_offset ...
            ,sprintf('%s',legend_labels{model-1}) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',12 ...
            ,'Rotation',25 ...
            ,'HorizontalAlignment','right' ...
            );
        
    end;    %If not a subject
    
    %value psychometric functions (in a different figure with different colormap), with lines for each position
    figure(h4); subplot(1,num_models,model);
    pm_line_colors = cool(size(position_data,2)+1);
    
    for position_line = 1:size(position_data,2)-1;
        
        if numel(size(position_data))==4;
            h = plot( nanmean(squeeze(position_data(:,position_line,:,model)),2) ); hold on;
        else
            h = plot( position_data(:,position_line,model) ); hold on;
        end;
        axis square;
        set(h,'Marker','o','MarkerSize',6,'MarkerEdgeColor',pm_line_colors(position_line,:),'MarkerFaceColor',pm_line_colors(position_line,:),'Color',pm_line_colors(position_line,:),'LineStyle','-','LineWidth',2);
        set(gca,'FontSize',12,'FontName','Arial','xtick',[1:size(position_data,1)],'xlim',[0.5 size(position_data,1)+0.5],'ylim',[0 1.1],'ytick',[0:0.2:1],'LineWidth',2);
        xlabel('Rating level'); ylabel('Proportion Choices'); box off;
        
    end;    %position lines
    
    if model == num_models;
        legend('Position 1','Position 2','Position 3','Position 4','Position 5','Position 6','Position 7','Position 8','Position 9','Position 10','Position 11','Position 12');
    end;
    
end;    %loop through models


%the model loop is done but we need to add the sig tests using the accumulated model data
%stole and modified code from BIC plot
%note: this num_models variable counts participants as a model
if num_models-1 ~= 1 & num_subs > 1;
    
    %run and plot ttests
    figure(h3); subplot( 1,2,2 );
    pairs = nchoosek(1:num_models-1,2);
    num_pairs = size(pairs,1);
    [a In] = sort(diff(pairs')','descend');  %lengths of connecting lines
    line_pair_order = pairs(In,:);    %move longest connections to top
    
    %Where to put top line?
    y_inc = .025;
    ystart = max(max(r)) + y_inc*num_pairs;
    line_y_values = ystart:-y_inc:0;
    
    fprintf(' ');
    
    for pair = 1:num_pairs;
%         
%         %run ttest this pair (with Fisher r to z)
%         [h r_pvals(pair) ci stats] = ttest(atanh(r(:,line_pair_order(pair,1))), atanh(r(:,line_pair_order(pair,2))));

          %get Bayes factor too
        [bf10(pair),r_pvals(pair),ci,stats] = ...
            bf.ttest( atanh(r(:,line_pair_order(pair,1))) - atanh(r(:,line_pair_order(pair,2))) );
        
        %plot result
        %             subplot(2,4,6); hold on;
        set(gca,'Ylim',[0 ystart]);

        if r_pvals(pair) < 0.05/size(pairs,1);
            plot([line_pair_order(pair,1)+1 line_pair_order(pair,2)+1],...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',4,'Color',[0 0 0]);
        end;
        
        if bf10(pair) < (1/10);
            plot([line_pair_order(pair,1)+1 line_pair_order(pair,2)+1],...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',1,'Color',[1 0 1]);
        end;
        if bf10(pair) > 10;
            plot([line_pair_order(pair,1)+1 line_pair_order(pair,2)+1],...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',1,'Color',[0 1 0]);
        end;
%         
%             plot([line_pair_order(pair,1)+1 line_pair_order(pair,2)+1],...
%                 [line_y_values(pair) line_y_values(pair)],'LineWidth',2,'Color',[0 0 0]);
            
        
    end;    %loop through ttest pairs
    
end;    %Only compute ttests if there is at least one pair of models








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = plot_data(Generate_params);

%set up plot appearance
%For now I'll try to match model identifiers to colors. Which means this
%colormap needs to scale to the total possible number of models, not the
%number of models
plot_cmap = hsv(8+1);  %models + subjects
f_a = 0.1; %face alpha
sw = 1;  %ppoint spread width
graph_font = 12;
x_axis_test_offset = .075;   %What percentage of the y axis range should x labels be shifted below the x axis?

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%SAMPLES AND RANKS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Plot samples for participants and different models
h10 = figure('NumberTitle', 'off', 'Name',['parameters ' Generate_params.outname]);
set(gcf,'Color',[1 1 1]);

%Need to accumulate these for analyze_value_position_functions below.
all_choice_trial(:,:,1) = Generate_params.num_samples'; %Should be subs*seqs*models. num_samples here (human draws) is seqs*subs
model_strs = {};
for this_bar = 1:Generate_params.num_models+1;   %+1 is participants
    
%     for perf_measure = 1:2;   %samples or ranks
%     for perf_measure = 1;   %samples

        
%         if perf_measure == 1;   %If samples
            subplot(2,3,1); hold on; %Samples plot
            y_string = 'Samples to decision';
            
            if this_bar == 1;   %If participants
                these_data = nanmean(Generate_params.num_samples)';
                plot_color = [1 0 0];
                model_label = 'Participants';
            else;   %if model
                these_data = nanmean(Generate_params.model(this_bar-1).num_samples_est)';
                plot_color = plot_cmap(Generate_params.model(this_bar-1).identifier+1,:);
                model_label = Generate_params.model(this_bar-1).name;
            end;    %partricipants or model?
            
            %accumulate data for output to file later on
            samples_accum(:,this_bar) = these_data;
            
%         else;   %If ranks
% %             subplot(2,2,2); hold on; %Samples plot
% %             y_string = 'Rank of chosen option';
% %             if this_bar == 1;   %If participants
% %                 these_data = nanmean(Generate_params.ranks)';
% %                 plot_color = [1 0 0];
% %                 model_label = 'Participants';
% %             else;   %if model
% %                 these_data = nanmean(Generate_params.model(this_bar-1).ranks_est)';
% %                 plot_color = plot_cmap(Generate_params.model(this_bar-1).identifier+1,:);
% %                 model_label = Generate_params.model(this_bar-1).name;
% %             end;    %partricipants or model?
%             
%         end;    %samples or ranks?
        
        %average over sequences (rows) but keep sub data (cols) for scatter points
        handles = plotSpread(these_data ...
            ,'xValues',this_bar ...
            ,'distributionColors',plot_color ...
            ,'distributionMarkers','.' ...
            , 'spreadWidth', sw ...
            );
        
        bar(this_bar,nanmean(these_data) ...
            ,'FaceColor',plot_color ...
            ,'FaceAlpha',f_a ...
            ,'EdgeColor',[0 0 0] ...
            );
        
        set(gca ...
            ,'XTick',[] ...
            ,'fontSize',graph_font ...
            ,'FontName','Arial',...
            'XLim',[0 Generate_params.num_models+2] ...
            ,'YLim',[0 Generate_params.seq_length] ...
            ,'LineWidth',2 ...
            );
        ylabel(y_string);
        
        this_offset = -x_axis_test_offset*diff(ylim);
        text( this_bar, this_offset ...
            ,sprintf('%s',model_label) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',graph_font ...
            ,'Rotation',25 ...
            ,'HorizontalAlignment','right' ...
            );
        
%     end;    %switch between samples and ranks
    
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%LOG LIKELIHOOD ANALYSIS
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     
    %Still inside model (this_bar) loop
    if this_bar ~=1;
%         
%         %%%%%%%%%%%%%%%%%%%%
%         %%%%Plot of raw ll's
%         subplot(2,4,2);
%         
%         %accumulate ll's for output to file later
        ll_accum(:,this_bar-1) = Generate_params.model(this_bar-1).ll;
%         
%         handles = plotSpread(Generate_params.model(this_bar-1).ll ...
%             ,'xValues',this_bar ...
%             ,'distributionColors',plot_color ...
%             ,'distributionMarkers','.' ...
%             , 'spreadWidth', sw ...
%             );
%         
%         bar(this_bar,nanmean(Generate_params.model(this_bar-1).ll) ...
%             ,'FaceColor',plot_color ...
%             ,'FaceAlpha',f_a ...
%             ,'EdgeColor',[0 0 0] ...
%             );
%         
%         set(gca ...
%             ,'XTick',[] ...
%             ,'fontSize',graph_font ...
%             ,'FontName','Arial',...
%             'XLim',[1 Generate_params.num_models+2] ...
%             ,'LineWidth',2 ...
%             );
%         %                     ,'YLim',[0 Generate_params.seq_length]...0
%         ylabel('Log-likelihood');
%         
%         this_offset = -x_axis_test_offset*diff(ylim);
%         text( this_bar, this_offset ...
%             ,sprintf('%s',model_label) ...
%             ,'Fontname','Arial' ...
%             ,'Fontsize',graph_font ...
%             ,'Rotation',25 ...
%             ,'HorizontalAlignment','right' ...
%             );
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        %Plot of AIC/BIC (Not so relevant if they all have two parameters though)
        subplot(2,3,2);
        
        %Model IC
        no_params = numel( Generate_params.model(this_bar-1).this_models_free_parameters ) + 1; %+1 for beta
        lla = Generate_params.model(this_bar-1).ll;
        if Generate_params.IC == 1; %If AIC (per participant)
            IC_pps = 2*no_params + 2*lla;
            %             IC_sum = nansum(IC_pps);
            %             IC_sum = 2*no_params(param_to_fit(model)) + 2*nansum(lla);
            a_label = 'AIC';
            %             IC_ylims = [800 1350];
        elseif Generate_params.IC == 2; %If BIC (per participant)
            IC_pps = no_params*log(Generate_params.num_seqs) + 2*lla;
            %             IC_sum = nansum(IC_pps);
            %             IC_sum = no_params(param_to_fit(model))*log(numel(lla)*28) + 2*nansum(lla);
            a_label = 'BIC';
            %             IC_ylims = [750 1250];
        end;
        
        handles = plotSpread(IC_pps ...
            , 'xValues',this_bar...
            ,'distributionColors',plot_color ...
            ,'distributionMarkers','.' ...
            , 'spreadWidth', sw ...
            );
        
        bar(this_bar,nanmean(IC_pps), ...
            'FaceColor',plot_color,'FaceAlpha',f_a,'EdgeColor',[0 0 0] );
        
        set(gca ...
            ,'XTick',[] ...
            ,'fontSize',graph_font ...
            ,'FontName','Arial',...
            'XLim',[1 Generate_params.num_models+2] ...
            ,'LineWidth',2 ...
            );
        ylabel(a_label);
        
        this_offset = -x_axis_test_offset*diff(ylim);
        text( this_bar, this_offset ...
            ,sprintf('%s',model_label) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',graph_font ...
            ,'Rotation',25 ...
            ,'HorizontalAlignment','right' ...
            );
        
        %We need to accumulate these data over models in this loop to do the
        %next step more easily
        IC_pps_all_models(:,this_bar-1) = IC_pps';
        %These we need to accumulate so they can be passed into
        %analyze_value_position_functions below
        all_choice_trial(:,:,this_bar) = Generate_params.model(this_bar-1).num_samples_est';
        model_strs{this_bar-1} = Generate_params.model(this_bar-1).name;
        
    end;    %If not participants (this bar ~=1)
    
end;    %loop through models

%the model loop is done but we are still working on the AIC/BIC plot and we
%need to add the sig tests using the accumulated model data
if Generate_params.num_models ~= 1 & Generate_params.num_subs > 1;
    
    %Paint ttest / Bayes factor results onto samples graph (This one only
    %needs participants compared against each of the models
    %Where to put top line?
    subplot(2,3,1);
    
    num_pairs = numel(Generate_params.do_models_identifiers);
    y_inc = .5;
    ystart = max(max(samples_accum)) + y_inc*num_pairs;
    line_y_values = ystart:-y_inc:0;
    for pair = 1:num_pairs;
        
        [bf10(pair),samples_pvals(pair),ci,stats] = ...
            bf.ttest( samples_accum(:,1) - samples_accum(:,num_pairs + 2 - pair) );
        
        set(gca,'Ylim',[0 ystart]);
        
        %distance on plot
        distance_on_plot = [1 num_pairs + 2 - pair];
        
        if samples_pvals(pair) < 0.05/num_pairs;
            plot(distance_on_plot,...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',4,'Color',[0 0 0]);
        end;
        
        if bf10(pair) < (1/10);
            plot(distance_on_plot,...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',1,'Color',[1 0 1]);
        end;
        if bf10(pair) > 10;
            plot(distance_on_plot,...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',1,'Color',[0 1 0]);
        end;
        
        
    end;   %pairs: Loop through comparisons between participants and models
    
    %run and plot ttests and Bayesfactor results on the plots above for BIC
    %graph (This one needs comparisons among all models)
    subplot(2,3,2);
    pairs = nchoosek(1:Generate_params.num_models,2);
    num_pairs = size(pairs,1);
    [a In] = sort(diff(pairs')','descend');  %lengths of connecting lines
    line_pair_order = pairs(In,:);    %move longest connections to top
    
     %Where to put top line?
     y_inc = 7;
     ystart = max(max(IC_pps_all_models)) + y_inc*num_pairs;
     line_y_values = ystart:-y_inc:0;
     
     for pair = 1:num_pairs;
        %
        %         %run ttest this pair
        %         [h IC_pp_pvals(pair) ci stats] = ttest(IC_pps_all_models(:,line_pair_order(pair,1)), IC_pps_all_models(:,line_pair_order(pair,2)));
        %
        %get Bayes factor too
        [bf10(pair),IC_pp_pvals(pair),ci,stats] = bf.ttest( IC_pps_all_models(:,line_pair_order(pair,1)) - IC_pps_all_models(:,line_pair_order(pair,2)) );
        
        %plot result
        %             subplot(2,4,6); hold on;
        set(gca,'Ylim',[0 ystart]);
        
        %         if bf10(pair) < (1/10);
        %             pair_color = [1 0 1];   %magenta
        %         elseif IC_pp_pvals(pair) < 0.05/size(pairs,1) & bf10(pair) > 10;
        %             pair_color = [0 .5 0];
        %         elseif IC_pp_pvals(pair) < 0.05/size(pairs,1) & bf10(pair) < 10;
        %             pair_color = [0 0 0];
        %         elseif IC_pp_pvals(pair) > 0.05/size(pairs,1) & bf10(pair) > 10;
        %              pair_color = [0 1 0];
        %         else
        %             pair_color = [1 1 1];
        %         end;
        
        if IC_pp_pvals(pair) < 0.05/size(pairs,1);
            plot([line_pair_order(pair,1)+1 line_pair_order(pair,2)+1],...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',4,'Color',[0 0 0]);
        end;
        
        if bf10(pair) < (1/10);
            plot([line_pair_order(pair,1)+1 line_pair_order(pair,2)+1],...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',1,'Color',[1 0 1]);
        end;
        if bf10(pair) > 10;
            plot([line_pair_order(pair,1)+1 line_pair_order(pair,2)+1],...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',1,'Color',[0 1 0]);
        end;
        
    end;    %loop through ttest pairs
    
end;    %Only compute ttests if there is at least one pair of models

%%%%%%%%%%%%%%%%%%%%%%
%Plot of numbers of winning subs for each model

subplot(2,3,3); hold on; box off;

%winning models
[a pps_indices] = min(IC_pps_all_models');

for model = 1:Generate_params.num_models;
    
    bar(model,numel(find(pps_indices==model)), ...
        'FaceColor', plot_cmap(Generate_params.model(model).identifier+1,:),'FaceAlpha',f_a,'EdgeColor',[0 0 0] );
    
    %get frequency of most common model so I can set the Y axis limits (and
    %thereby correctly judge distance of x axis labels from axis)
    max_freq = max(histc(pps_indices,Generate_params.do_models_identifiers));
    
    set(gca ...
        ,'XTick',[] ...
        ,'fontSize',graph_font ...
        ,'FontName','Arial'...
        ,'XLim',[0 Generate_params.num_models+1] ...
        ,'YLim',[0 max_freq] ...
        ,'LineWidth',2 ...
        );
    ylabel('Frequency');
    
    this_offset = -x_axis_test_offset*diff(ylim);
    text( model, this_offset ...
        ,sprintf('%s',Generate_params.model(model).name) ...
        ,'Fontname','Arial' ...
        ,'Fontsize',graph_font ...
        ,'Rotation',25 ...
        ,'HorizontalAlignment','right' ...
        );
    
end;    %models


%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Write out the data matrices used for plots so they can be analysed statistically in JASP
%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf(' ');

%big matrix
%cols 1-6: samples participants & 5 models; cols 7-11: ll 5 models; cols 12-16: BIC 5 models
out = [samples_accum ll_accum IC_pps_all_models];

writematrix(out,'C:\matlab_files\fiance\parameter_recovery\outputs\models_av_out01.csv');


%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%INDIVIDUAL PARTICIPANT DATA SCATTERPPLOTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Scatterplots of participant-model subject by subject relationships (sampling)
h11 = figure('NumberTitle', 'off', 'Name',['parameters ' Generate_params.outname]);
set(gcf,'Color',[1 1 1]);

%For scatterplot subplots
num_rows = floor(sqrt(numel(Generate_params.do_models_identifiers )) );
num_cols = ceil(numel(Generate_params.do_models_identifiers)/num_rows);

for identifier = 1:numel(Generate_params.do_models_identifiers);
    
    %     subplot(num_rows, num_cols, identifier); hold on; box off;
    subplot( 1, numel(Generate_params.do_models_identifiers), identifier); hold on; box off;
    
    scatter( ...
        nanmean(Generate_params.num_samples)' ...
        , nanmean(Generate_params.model(identifier).num_samples_est)' ...
        , 'MarkerEdgeColor', plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:) ...
        , 'MarkerFaceColor', plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:) ...
        );
    
    %regression line
    [b,bint,r,rint,stats] = regress( ...
        nanmean(Generate_params.model(identifier).num_samples_est)' ...
        , [ones(Generate_params.num_subs,1) nanmean(Generate_params.num_samples)'] ...
        );
    %     x_vals = [min(nanmean(Generate_params.num_samples)') max(nanmean(Generate_params.num_samples)')];
    x_vals = [0 Generate_params.seq_length];
    
    y_hat = b(1) + b(2)*x_vals;
    
    %Put Rsquared on plot
    text( 0.5, Generate_params.seq_length + .5 ...
        , sprintf('R squared = %0.2f',stats(1)) ...
        , 'Fontname','Arial' ...
        , 'Fontsize',graph_font ...
        , 'FontWeight','normal' ...
        );
    
    %Plot regression line
    plot( x_vals, y_hat ...
        , 'Color', plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:) ...
        );
    
    %Plot a diagonal of inequality too
    plot( ...
        [0 Generate_params.seq_length] ...
        , [0 Generate_params.seq_length] ...
        , 'Color', [.5 .5 .5] ...
        );
    
    set(gca ...
        , 'Fontname','Arial' ...
        , 'Fontsize',graph_font ...
        , 'FontWeight','normal' ...
        , 'YTick',[0:2:Generate_params.seq_length] ...
        , 'XTick',[0:2:Generate_params.seq_length] ...
        , 'LineWidth',2 ...
        );
    
    ylim([0 Generate_params.seq_length+1]);
    xlim([0 Generate_params.seq_length]);
    ylabel('Predicted sampling');
    xlabel('Participant sampling');
    
    title( ...
        sprintf('%s',Generate_params.model(identifier ).name) ...
        , 'Fontname','Arial' ...
        , 'Fontsize',graph_font ...
        , 'FontWeight','normal' ...
        );
    
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Analysis of thresholds!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     if Generate_params.analyze_value_positions == 1;
%             analyze_value_position_functions(raw_seqs_subs,all_choice_trial,plot_cmap,binEdges_psi,model_strs,param_to_fit,two_params);
nbins_psi = 9;
binEdges_psi(1:Generate_params.num_subs,:) = ...
    repmat(...
    linspace(...
    Generate_params.rating_bounds(1) ...
    ,Generate_params.rating_bounds(2) ...
    ,nbins_psi+1 ...
    ), ...
    numel(1:Generate_params.num_subs),1 ...
    );

analyze_value_position_functions(...
    Generate_params.seq_vals ...
    ,all_choice_trial ...
    ,plot_cmap ...
    ,binEdges_psi ...
    ,model_strs ...
    ,Generate_params.do_models_identifiers ...
    ,1 ...
    );

%     end;    %make threshold by serial position plot



