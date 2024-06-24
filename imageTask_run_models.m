function Generate_params = run_models(Generate_params, fitting_bounds)


[Generate_params fitting_bounds] = imageTask_setup_models(Generate_params);

%In the previous section, we only assigned to the Generate_param
%struct the models that were in the do_models in that section. So
%this can only operate on those models (until I start implementing these sections from datafiles later).
for model = 1:numel(Generate_params.do_models);

    for sub = 1:Generate_params.num_subs;

        %You want to fit one model for one subject at a time
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

        Generate_params = imageTask_do_fits_get_performance(params, Generate_params, model, sub);

    end;    %Loop through subs

    %Should save after each model completed
    save([Generate_params.outpath filesep Generate_params.outname], 'Generate_params');

end;   %loop through models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Generate_params fitting_bounds] = imageTask_setup_models(Generate_params);

%configures theoertical models anf prepares input struct for model fitting.
%This is an operation that all imageTask studies have in common and so each
%separate script calls this code

%These correspond to identifiers
model_names = {'CO' 'Cs' 'IO' 'BV' 'BR' 'BPM' 'Opt' 'BPV' }; %IO is a placeholder, don't implement
Generate_params.num_model_identifiers = size(model_names,2);

%Now that you have info on the subs, load up the main struct with all the basic info you might need
Generate_params.num_subs =  size(Generate_params.seq_vals,3);
Generate_params.num_seqs =  size(Generate_params.seq_vals,1);
Generate_params.seq_length =  size(Generate_params.seq_vals,2);
Generate_params.num_vals = size(Generate_params.ratings,1);
Generate_params.rating_bounds = [1 100];    %Either Gorilla so already 1 to 100 or the function that gets the matlab behavioural data rescaled it to be 1 to 100
Generate_params.BVrange = Generate_params.rating_bounds;

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
model_template.BVmid = 50;      %row 7, initialised to halfway through the rating scale (can't be used with log)
model_template.BRslope = 1;    %row 8
model_template.BRmid = 50;      %row 9
model_template.BP = 0;           %row 10
model_template.optimism = 0;    %row 11
model_template.BPV = 0;          %row 12
model_template.beta = 1;        %Just for parameter estimation.
model_template.name = 'template';


%Repmat the template to create a column for each model. For now, we are
%doing all possible models, not the ones specified in do_models. We'll
%reduce this matrix to just those below.
identifiers = 1:Generate_params.num_model_identifiers;
num_cols = Generate_params.num_model_identifiers;
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
param_config_default = param_config_default(:,Generate_params.do_models);
free_parameters = free_parameters(:,Generate_params.do_models);

%Save your work into struct
Generate_params.num_models = numel(Generate_params.do_models);
Generate_params.param_config_default = param_config_default;
Generate_params.free_parameters_matrix = free_parameters;
Generate_params.outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\outputs';

outname = [Generate_params.comment char(datetime('now','format','yyyyddMM')) '.mat'];
Generate_params.outname = outname;

disp( sprintf('Running %s', outname) );

%Now fill in default parameters to model fields
for model = 1:Generate_params.num_models;   %How many models are we implementing (do_models)?

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

    %Fill in this model's free parameters to be estimated later, if you get to the parameter estimatioin this run
    Generate_params.model(model).this_models_free_parameters = find(free_parameters(:,model)==1);
    Generate_params.model(model).this_models_free_parameter_default_vals = param_config_default(find(free_parameters(:,model)==1),model)';

end;    %loop through models

%set upper and lower bounds for the different models (A bit
%disorganised to put it here, I admit, but this is a new section of
%code grafted onto a much older structure so this bit will look a little more ad hoc

%I'll just specify all the bounds together here and then assign them to
%Generate_params model structures and pass them to fminsearch in the
%model loop immediately below
fitting_bounds.CO = [2 Generate_params.seq_length-1];   %cut off, it's a threshold that must be inside sequence
fitting_bounds.Cs = [-100 100];   %cost to sample
fitting_bounds.BV = [1 100]; %biased values (O), it's a threshold that must be inside rating bounds
fitting_bounds.BR = [1 100]; %biased reward (O), it's a threshold that must be inside rating bounds
fitting_bounds.BPM = [-100 100];  %biased prior, value can't exit the rating scale
fitting_bounds.Opt = [-100 100];  %optimism, value can't exit rating scale
fitting_bounds.BVar = [-100 100];  %biased variances, can't be wider than the whole rating scale
fitting_bounds.beta = [0 100];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Generate_params = imageTask_do_fits_get_performance(params, Generate_params, model, sub)

%enables all the different scripts for the different imageTask studies (e.g., fit_models_face_1_*.m) to
%use one common set of functions for model fitting and performance
%generation. To make this work, it's easiest to do away with the modular
%set up I had before so you must generate performance for each model that
%is fit now, with no use_file or switch to turn performance generation or
%model fitting on or off. It will also automatically do_io without any
%switch. These are all more or less obsolete now anyway. This works one sub
%at a time.


%Do model fit for this participant ....

warning('off');

disp(...
    sprintf('fitting modeli %d name %s subject %d' ...
    , model ...
    , Generate_params.model( model ).name ...
    , sub ...
    ) );

%Do fitting
[Generate_params.model(model).estimated_params(sub,:) ...
    ,  Generate_params.model(model).ll(sub,:) ...
    , exitflag, search_out] = ...
    fminsearchbnd(  @(params) f_fitparams( params, Generate_params ), ...
    params,...
    [Generate_params.model(model).lower_bound Generate_params.model(model).lower_bound_beta], ...
    [Generate_params.model(model).upper_bound Generate_params.model(model).upper_bound_beta] ...
    );


%get performance
Generate_params.current_model = model;
[temp1 temp2] = ...
    generate_a_models_data_est(Generate_params,sub);

Generate_params.model(model).num_samples_est(:,sub) = temp1';
Generate_params.model(model).ranks_est(:,sub) = temp2';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [num_samples ranks] = generate_a_models_data_est(Generate_params,sub);

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

    %...and beta (assume beta is last)
    Generate_params.model(Generate_params.current_model).beta = ...
        Generate_params.model(Generate_params.current_model).estimated_params(sub,end);

    disp(...
        sprintf('computing performance, fitted modeli %d name %s subject %d' ...
        , Generate_params.current_model ...
        , Generate_params.model( Generate_params.current_model ).name ...
        , sub ...
        ) );

    Generate_params.num_subs_to_run = sub;
    [num_samples ranks] = generate_a_models_data(Generate_params);
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

    %How many samples for this sequence and subject
    listDraws = ...
        Generate_params.num_samples(seq,Generate_params.num_subs_to_run);

    %Loop through trials to be modelled to get choice probabilities for each action value
    for drawi = 1 : listDraws
        %cprob seqpos*choice(draw/stay)
        cprob(drawi, :) = exp(b*choiceValues(drawi, :))./sum(exp(b*choiceValues(drawi, :)));
    end;

    %Compute ll
    if listDraws == 1;  %If only one draw
        ll = ll - 0 - log(cprob(listDraws, 2));
    else
        ll = ll - sum(log(cprob((1:listDraws-1), 1))) - log(cprob(listDraws, 2));
    end;

end;    %seq loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




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

        %just one sequence
        list.vals = squeeze(Generate_params.seq_vals(sequence,:,num_subs_found));

        %get prior dist moments
        Generate_params.PriorMean = nanmean(Generate_params.ratings(:,num_subs_found));
        Generate_params.PriorVar = nanvar(Generate_params.ratings(:,num_subs_found));

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
            %Reverse 0s and 1's
            choiceCont = double(~choiceStop);

        else;   %Any Bayesian models

            [choiceStop, choiceCont, difVal] = analyzeSecretary_imageTask_2024(Generate_params,list.vals);

        end;    %Cutoff or other model?


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

        %Accumulate action values too so you can compute ll outside this function if needed
        choiceStop_all(sequence, :, this_sub) = choiceStop;
        choiceCont_all(sequence, :, this_sub) = choiceCont;

    end;    %loop through sequences

    this_sub = this_sub + 1;

end;    %loop through subs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [choiceStop, choiceCont, difVal] = analyzeSecretary_imageTask_2024(Generate_params,sampleSeries);

N = Generate_params.seq_length;
Cs = Generate_params.model(Generate_params.current_model).Cs;      %Will already be set to zero unless Cs model

%Assign params. Hopefully everything about model is now pre-specified
%(I would update mean and variance directly from Generate_params 
prior.mu    = Generate_params.PriorMean + Generate_params.model(Generate_params.current_model).BP; %prior mean offset is zero unless biased prior model
prior.sig   = Generate_params.PriorVar + Generate_params.model(Generate_params.current_model).BPV;                        %Would a biased variance model be a distinct model?
if prior.sig < 1; prior.sig = 1; end;   %It can happen randomly that a subject has a low variance and subtracting the bias gives a negative variance. Here, variance is set to minimal possible value.
prior.kappa = Generate_params.model(Generate_params.current_model).kappa;   %prior mean update parameter
prior.nu    = Generate_params.model(Generate_params.current_model).nu;

%Apply BV transformation, if needed
if Generate_params.model(Generate_params.current_model).identifier == 4;   %If identifier is BV

%     sampleSeries = ...
%         logistic_transform(...
%         sampleSeries, ...
%         Generate_params.BVrange, ...
%         Generate_params.model(Generate_params.current_model).BVslope, ...
%         Generate_params.model(Generate_params.current_model).BVmid ...
%         );

%minimise input values below threshold
sampleSeries( find(sampleSeries <= Generate_params.model(Generate_params.current_model).BVmid)) = 0;

end;

[choiceStop, choiceCont, difVal, currentRnk] = computeSecretary(Generate_params, sampleSeries, prior, N, Cs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function vals = logistic_transform(data,range,slope,mid);
% 
% %Returns unsorted logistic transform of vals on 1 to 100 scale
% 
% 
% vals = (range(2) - range(1)) ./ ...
%     (1+exp(-slope*(data-mid))); %do logistic transform
% 
% %normalise
% old_min = (range(2) - range(1)) ./ ...
%     (1+exp(-slope*(range(1)-mid))); %do logistic transform
% 
% old_max = (range(2) - range(1)) ./ ...
%     (1+exp(-slope*(range(2)-mid))); %do logistic transform
% 
% new_min=1;
% new_max = 100;
% 
% vals = (((new_max-new_min)*(vals - old_min))/(old_max-old_min))+new_min;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [choiceStop, choiceCont, difVal, currentRnk] = computeSecretary(Generate_params, sampleSeries, priorProb, N, Cs)

sdevs = 8;
dx = 2*sdevs*sqrt(priorProb.sig)/100;
x = ((priorProb.mu - sdevs*sqrt(priorProb.sig)) + dx : dx : ...
    (priorProb.mu + sdevs*sqrt(priorProb.sig)))';

Nconsider = length(sampleSeries);
if Nconsider > N
    Nconsider = N;
end

difVal = zeros(1, Nconsider);
choiceCont = zeros(1, Nconsider);
choiceStop = zeros(1, Nconsider);
currentRnk = zeros(1, Nconsider);

for ts = 1 : Nconsider
    
    [expectedStop, expectedCont] = rnkBackWardInduction(sampleSeries, ts, priorProb, N, x, Cs, Generate_params);
    
    [rnkv, rnki] = sort(sampleSeries(1:ts), 'descend');
    z = find(rnki == ts);
        
    difVal(ts) = expectedCont(ts) - expectedStop(ts);
    
    choiceCont(ts) = expectedCont(ts);
    choiceStop(ts) = expectedStop(ts);
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [expectedStop, expectedCont, expectedUtility] = rnkBackWardInduction(sampleSeries, ts, priorProb, ...
    listLength, x, Cs,  Generate_params)

N = listLength;
Nx = length(x);

if Generate_params.model(Generate_params.current_model).identifier == 5;

%     rewards = ...
%         logistic_transform(...
%         sampleSeries, ...
%         Generate_params.BVrange, ...
%         Generate_params.model(Generate_params.current_model).BRslope, ...
%         Generate_params.model(Generate_params.current_model).BRmid ...
%         );

 %minimise payoff values below threshold
    payoff = sampleSeries;
    payoff( find(payoff <= Generate_params.model(Generate_params.current_model).BRmid)) = 0;
    payoff = sort(payoff,'descend');

else;

    payoff = sort(sampleSeries,'descend');

end;

maxPayRank = numel(payoff);
payoff = [payoff zeros(1, 20)];

data.n  = ts;

data.sig = var(sampleSeries(1:ts));
data.mu = mean(sampleSeries(1:ts));

utCont  = zeros(length(x), 1);
utility = zeros(length(x), N);

if ts == 0
    ts = 1;
end

[rnkvl, rnki] = sort(sampleSeries(1:ts), 'descend');
z = find(rnki == ts);
rnki = z;

ties = 0;
if length(unique(sampleSeries(1:ts))) < ts
    ties = 1;
end

mxv = ts;
if mxv > maxPayRank
    mxv = maxPayRank;
end

rnkv = [Inf*ones(1,1); rnkvl(1:mxv)'; -Inf*ones(20, 1)];

[postProb] = normInvChi(priorProb, data);

postProb.mu ...
    = postProb.mu ...
    + Generate_params.model(Generate_params.current_model).optimism; %...Then add constant to the posterior mean (will be zero if not optimism model)

px = posteriorPredictive(x, postProb);
px = px/sum(px);

Fpx = cumsum(px);
cFpx = 1 - Fpx;

for ti = N : -1 : ts
    
    if ti == N
        utCont = -Inf*ones(Nx, 1);
    elseif ti == ts
        utCont = ones(Nx, 1)*sum(px.*utility(:, ti+1));
    else
        utCont = computeContinue(utility(:, ti+1), postProb, x, ti);
    end
    
    %%%% utility when rewarded for best 3, $5, $2, $1
    utStop = NaN*ones(Nx, 1);
    
    rd = N - ti; %%% remaining draws
    id = max([(ti - ts - 1) 0]); %%% intervening draws
    td = rd + id;
    ps = zeros(Nx, maxPayRank);
    
    for rk = 0 : maxPayRank-1
        
        pf = prod(td:-1:(td-(rk-1)))/factorial(rk);
        
        ps(:, rk+1) = pf*(Fpx.^(td-rk)).*((cFpx).^rk);
        
    end

    for ri = 1 : maxPayRank+1
        
        z = find(x < rnkv(ri) & x >= rnkv(ri+1));
        utStop(z) = ps(z, 1:maxPayRank)*(payoff(1+(ri-1):maxPayRank+(ri-1))');
        
    end
    
    if sum(isnan(utStop)) > 0
        fprintf('Nan in utStop');
    end
    
    if ti == ts
        [zv, zi] = min(abs(x - sampleSeries(ts)));
        if zi + 1 > length(utStop)
            zi = length(utStop) - 1;
        end
        
        utStop = utStop(zi+1)*ones(Nx, 1);
        
    end
    
    utCont = utCont - Cs;
    
    utility(:, ti)      = max([utStop utCont], [], 2);
    expectedUtility(ti) = px'*utility(:,ti);
    
    expectedStop(ti)    = px'*utStop;
    expectedCont(ti)    = px'*utCont;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%









%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function utCont = computeContinue(utility, postProb0, x, ti)

postProb0.nu = ti-1;

utCont = zeros(length(x), 1);

expData.n   = 1;
expData.sig = 0;

for xi = 1 : length(x)
    
    expData.mu  = x(xi);
    
    postProb = normInvChi(postProb0, expData);
    spx = posteriorPredictive(x, postProb);
    spx = (spx/sum(spx));
        
    utCont(xi) = spx'*utility;
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [postProb] = normInvChi(prior, data)

postProb.nu    = prior.nu + data.n;

postProb.kappa = prior.kappa + data.n;

postProb.mu    = (prior.kappa/postProb.kappa)*prior.mu + (data.n/postProb.kappa)*data.mu;

postProb.sig   = (prior.nu*prior.sig + (data.n-1)*data.sig + ...
    ((prior.kappa*data.n)/(postProb.kappa))*(data.mu - prior.mu).^2)/postProb.nu;

if data.n == 0
    postProb.sig = prior.sig;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function prob_y = posteriorPredictive(y, postProb)

tvar = (1 + postProb.kappa)*postProb.sig/postProb.kappa;

sy = (y - postProb.mu)./sqrt(tvar);

prob_y = tpdf(sy, postProb.nu);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

















