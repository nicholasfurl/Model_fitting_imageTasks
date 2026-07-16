%%%%%%%%%%%%%Start main body%%%%%%%%%%%%%%%%%%%%%
function [] = imageTask_model_recovery_from_scratch_20241028;

%14 Nov 2024. Added plots of BIC per sim*fit model and simmed - fitted BIC matrix

%20241031 halloween - spooky! - saving new version because going to make
%bigger change. Previous version never worked (I think). This one
%simplifies by just randomly sampling new sets of parameters for every
%subject, rather than getting a whole subject sample for every parameter level.

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasksy\FMINSEARCHBND'))
addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\plotSpread'));
addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\klabhub-bayesFactor-3d1e8a5'));

simulate = 0;   %simulate data, fit models and produce output file? Or skip?
plots = 1;  %make plots of results? Or skip?
use_file = 1; %read an output file (to be specified inside plots block) for the plotting? Or just let the input structure created in simulate block be inherited by the plotting block without opening any file?

%%%start simulate block
if simulate == 1;

    beta_upper_bound = 100;

    %for filename, to be saved later
    comment = 'imageTask';
    out_file_name = ['C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\outputs\' 'MR_' comment '_' datestr(now, 'yyyymmddHHMM')]
    disp(sprintf('**************imageTask model recovery: %s*******************',out_file_name));

    %Simulate stimuli
    [input.simed_subs input.N] = make_sequences; %configure within function definition

    %prepare parameter levels for simulated models (adapted from param recovery)
    %I am trying to cover multiple studies whose estimated parameters might all be
    % %distributed somewhat differently and and I don't suspect the estimated
    %parameters will be distributed normally necessarily anyway, so I'm going
    %to randomly sample parameter pairs uniformly from the ranges capable of
    %modulating the sampling rate for the first parameter and something wide for beta

    input.models(1).name = 'CS';
    %     input.models(1).sim_parameter_range(1,:) = [-.05,.015]; %CS, tight range
    input.models(1).sim_parameter_range(1,:) = [-1,1]; %CS, wide range
    %     input.models(1).sim_parameter_range(2,:) = [0, 40];  %beta, tight range
    input.models(1).sim_parameter_range(2,:) = [0, 100];  %beta, wide range
    input.models(1).function = @compute_behaviour_CS;
    input.models(1).fit_bounds(1,:) = [-1 1];
    input.models(1).fit_bounds(2,:) = [0 beta_upper_bound];
    input.models(1).fit_initial_params = [0 1];

    input.models(2).name = 'CO';
    input.models(2).sim_parameter_range(1,:) = [2,8];    %CO
    %     input.models(2).sim_parameter_range(2,:) = [0, 40];  %beta, tight range
    input.models(2).sim_parameter_range(2,:) = [0, 100];  %beta, wide range
    input.models(2).function = @compute_behaviour_CO;
    input.models(2).fit_bounds(1,:) = [2 input.N-1];   %cut off, it's a threshold that must be inside sequence
    input.models(2).fit_bounds(2,:) = [0 beta_upper_bound];
    input.models(2).fit_initial_params = [ceil(exp(-1)*input.N) 1];


    input.models(3).name = 'BP';
%     input.models(3).sim_parameter_range(1,:) = [-50,50];   %BP    %tight      range
    input.models(3).sim_parameter_range(1,:) = [-100,100];   %BP, wide range
    %     input.models(3).sim_parameter_range(2,:) = [0, 40];  %beta, tight range
    input.models(3).sim_parameter_range(2,:) = [0, 100];  %beta, wide range
    input.models(3).function = @compute_behaviour_BP;
    input.models(3).fit_bounds(1,:) = [-100 100];
    input.models(3).fit_bounds(2,:) = [0 beta_upper_bound];
    input.models(3).fit_initial_params = [0 1];

    %simulate behaviour for each sequence and fit all three models to each simulated subject
    % for parameter_level = 1:num_param_levels;

    for simed_model = 1:size(input.models,2);
        for subject = 1:size(input.simed_subs,2);

            %generate new set of parameters for this subject and model

            %first parameter
            temp_min = input.models(simed_model).sim_parameter_range(1,1);
            temp_max = input.models(simed_model).sim_parameter_range(1,2);
            param1 = temp_min + (temp_max - temp_min) * rand(1, 1);

            %beta
            temp_min = input.models(simed_model).sim_parameter_range(2,1);
            temp_max = input.models(simed_model).sim_parameter_range(2,2);
            param2 = temp_min + (temp_max - temp_min) * rand(1, 1);

            % Combine into an Nx2 matrix for all combinations
            input.models(simed_model).sim_parameter_samples(subject,:) = [param1, param2];

            clear this_models_draws;

            for sequence = 1:size(input.simed_subs(subject).sequences,1);

                %set up sequence info (changes for every simulation of model)
                input.current_subject = subject;    %needed during model fitting to identify the correct set of sequences
                input.mu = input.simed_subs(subject).mu;
                input.sig = input.simed_subs(subject).sig;
                input.sampleSeries = squeeze(input.simed_subs(subject).sequences(sequence,:));
                input.N = numel(input.sampleSeries);

                %simulate model performance
                params(1) = input.models(simed_model).sim_parameter_samples(subject,1);    %CS
                params(2) = input.models(simed_model).sim_parameter_samples(subject,2);    %beta
                cprob = input.models(simed_model).function(params,input);
                [val index] = max(cprob');
                this_simed_models_draws(sequence,1) = find(index == 2,1);

            end;    %sequences

            %save the draws data for this simulated model for posterity but also keep
            %this_models_draws for now to pass into the next loop to fit
            %these simulated numbers of draws to the three fitted models
            input.models(simed_model).sim_num_draws(subject,:) = this_simed_models_draws;

            %Now fit the three models to the numbers of draws that we just computed above
            for fitted_model = 1:size(input.models,2);

                %get the name of the model to be fitted
                input.models(simed_model).fitted_models(fitted_model).name = input.models(fitted_model).name;
                %get the handle of the model to be fitted (and save to structure for debugging purposes)
                which_model = input.models(fitted_model).function;
                input.models(simed_model).fitted_models(fitted_model).which_model = which_model;
                %get the lower bounds of the two parameters for the model to be fitted
                lower_bounds = [input.models(fitted_model).fit_bounds(1,1) input.models(fitted_model).fit_bounds(2,1)];
                input.models(simed_model).fitted_models(fitted_model).lower_bounds = lower_bounds;
                %get the upper bounds of the two parameters for the model to be fitted
                upper_bounds = [input.models(fitted_model).fit_bounds(1,2) input.models(fitted_model).fit_bounds(2,2)];
                input.models(simed_model).fitted_models(fitted_model).upper_bounds = upper_bounds;
                %get the initial params for the model to be fitted
                params = input.models(fitted_model).fit_initial_params;
                input.models(simed_model).fitted_models(fitted_model).fit_initial_params = params;

                %for debugging purposes, let's just ensure exactly what data were fitted
                input.models(simed_model).fitted_models(fitted_model).fitted_draws(subject,:) = this_simed_models_draws;

                disp(sprintf('fitting subject %d, simed model %s fitted model %s',subject,input.models(simed_model).name, input.models(fitted_model).name));

                [input.models(simed_model).fitted_models(fitted_model).estimated_params(subject,:) ...
                    , input.models(simed_model).fitted_models(fitted_model).ll(subject,:) ...
                    , exitflag, search_out] = ...
                    fminsearchbnd(  @(params) f_fitparams(params, input, which_model, this_simed_models_draws), ...
                    params,...
                    lower_bounds, ...
                    upper_bounds ...
                    );

            end;    %fitted models
        end;    %subjects

        %let's do a save here
        save(out_file_name)

    end;    %simulated models
end;    %simulate?
%%%end simulate block


%%%start plot block
h2 = figure('Color',[1 1 1]);
% temp = hsv(9);
plot_cmap = [    0.6, 0.3, 0.6;    % Purple
    0.85, 0.33, 0.1;  % Orange
    0.4, 0.6, 0.2 ; %green
    0, 0.45, 0.7;    % Blue
    0.95, 0.9, 0.25;  % Yellow
    0.6, 0.4, 0.2  % Brown
    ];
% cmap = [0.85, 0.33, 0.1;  % Orange
%     0.6, 0.3, 0.6;    % Purple
%     0, 0.45, 0.7;    % Blue
%     0.95, 0.9, 0.25;  % Yellow
%     0.6, 0.4, 0.2;  % Brown
%     0.4, 0.6, 0.2 ;
% plot_cmap = temp(2:end,:);
% plot_cmap = plot_cmap([2 1 6],:);
model_labels = {'Cost to sample', 'Cut off', 'Biased prior'};
graph_font = 12;

if plots == 1;

    if use_file == 1;

%          load('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\outputs\MR_earlyTest_202411101407.mat'); %this one should have N = 1000, beta < 100 fit, big simulated parameter ranges , seq length 12
       load('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\outputs\MR_imageTask_202502281700.mat'); %this one should have N = 1000, beta < 100 fit, big simulated parameter ranges , seq length 12

    end;    %base plots on input struct in this file?

    %make confusion matrix
    confusion_matrix = zeros(size(input.models,2),size(input.models(simed_model).fitted_models,2));
    ll_matrix = zeros(size(input.models,2),size(input.models(simed_model).fitted_models,2));
    for simed_model = 1:size(input.models,2);   %Loop through simulated models
        for subject = 1:size(input.models(simed_model).fitted_models(fitted_model).ll,1);

            %pull together the ll's for the three fitted models for this subject
            fitted_lls_this_subject = [];
            for fitted_model = 1:size(input.models(simed_model).fitted_models,2);  %Loop through fitted models

                fitted_lls_this_subject(fitted_model) = input.models(simed_model).fitted_models(fitted_model).ll(subject,1);

            end;    %loop through fitted models;

            %model wins and their confusion matrix
            [~,winning_fitted_model] = min(fitted_lls_this_subject);    %find winning model this subject
            confusion_matrix(simed_model,winning_fitted_model) = confusion_matrix(simed_model,winning_fitted_model) + 1;

            %log likelihoods of the three models (for plotting comparisons of them), converted to BIC
            lls(subject,simed_model,:) = fitted_lls_this_subject;
            lls(subject,simed_model,:) = 2.*input.N + 2*fitted_lls_this_subject;

            %inelegant, but helps to loop through fitted_model again, even though I just did it
            for fitted_model = 1:size(input.models(simed_model).fitted_models,2);  %Loop through fitted models

                ll_diffs(subject,simed_model,fitted_model) = lls(subject,simed_model,fitted_model) - lls(subject,simed_model,simed_model);

                %populate this row of ll matrix for this subject
                ll_matrix(simed_model, fitted_model) = ll_matrix(simed_model,fitted_model) + (lls(subject,simed_model,fitted_model) - lls(subject,simed_model,simed_model));

            end;    %loop through fitted mnodels (again) to populate ll_matrix with differences from simuklated model;

        end;    %loop through subjects

        figure(h2);
        sp1 = subplot(1,3,simed_model); hold on;

        %make sure line at zero is visible
        plot([0 size(input.models(simed_model).fitted_models,2)],[0 0],'LineWidth',.5,'Color',[0 0 0]);

        for fitted_model = 1:size(input.models(simed_model).fitted_models,2);  %Loop through fitted models

            y_data = ll_diffs;

            boxplot_details.fig = figure(h2);
            boxplot_details.sp = sp1;
            boxplot_details.x_val = fitted_model;
            boxplot_details.these_data = squeeze(y_data(:,simed_model,fitted_model));
            boxplot_details.externalColor = [0.3 0.3 0.3];
            boxplot_details.internalColor = plot_cmap(fitted_model,:) + 0 * ([1, 1, 1] - plot_cmap(fitted_model,:));
            boxplot_details.width = .35;
            boxplot_details.alpha = .1;
            boxplot_details.lineWidth = .5;

            boxplot_nick(boxplot_details);
            %bar(fitted_model,mean(squeeze(lls(:,simed_model,fitted_model))), 'FaceAlpha',.3,'FaceColor',plot_cmap(fitted_model,:));

            handles = plotSpread(squeeze(y_data(:,simed_model,fitted_model)) ...
                , 'xValues',fitted_model ...
                ,'distributionColors',plot_cmap(fitted_model,:) ...
                ,'distributionMarkers','.' ...
                , 'spreadWidth', .45 ...
                );

            title(sprintf('Simulated model: \n %s',model_labels{simed_model}), 'FontSize', graph_font, 'FontName', 'Arial','FontWeight','normal', 'HorizontalAlignment', 'center');

            % Add x-axis label for current x-value
            set(gca, 'XTick', 1:fitted_model, 'XTickLabel', model_labels, 'FontName','Arial', 'FontSize', graph_font);
            set(gca, 'FontSize', graph_font, 'FontName', 'Arial');

            ylim([-10 50]);

            % Set the axis labels
            xlabel('Fitted model', 'FontSize', graph_font, 'FontName', 'Arial');

            ylabel(sprintf('BIC fitted model - \n BIC simulated model'), 'FontSize', graph_font, 'FontName', 'Arial');

        end;    %loopthrough fitted models

    end;    %loop through simed models

    %convert to proportion 100 participants per simulated model
    confusion_matrix = confusion_matrix/size(input.simed_subs,2);

    %convert to inverse confusion matrix

    % Step 1: Compute P(simulated model) as the row sums (assuming uniform priors for simulated models)
    P_simulated_model = sum(confusion_matrix, 2) / size(confusion_matrix, 2);

    % Step 2: Compute P(fitted model) as the column sums
    P_fitted_model = sum(confusion_matrix, 1)'/ size(confusion_matrix, 1);

    % Step 3: Initialize the inverse confusion matrix
    inverse_confusion_matrix = zeros(size(confusion_matrix));

    % Step 4: Apply Bayes' Rule to fill in inverse_C
    for i = 1:size(confusion_matrix, 1)  % Loop over simulated models
        for j = 1:size(confusion_matrix, 2)  % Loop over fitted models
            % Bayes' Rule calculation
            inverse_confusion_matrix(j, i) = (confusion_matrix(i, j) * P_simulated_model(i)) / P_fitted_model(j);
        end
    end


    graph_font = 12;
    h1 = figure('Color',[1 1 1]);

    rowNames = {'Cost to sample', 'Cut off', 'Biased prior'};
    colNames = {'Cost to sample', 'Cut off', 'Biased prior'};

    subplot(2,1,1);

    h = heatmap(confusion_matrix);  % Plot the heatmap

    h.CellLabelFormat = '%.2f';

    % Set the X and Y tick labels
    h.XDisplayLabels = colNames;
    h.YDisplayLabels = rowNames;

    % Set the X and Y axis labels
    h.XLabel = 'Fitted models';
    h.YLabel = 'Simulated models';

    h.FontSize = graph_font;
    h.FontName = 'Arial';

    subplot(2,1,2);

    % inv_confusions_table = array2table(inverse_confusion_matrix, 'RowNames', rowNames, 'VariableNames', colNames); % Convert matrix to a table with row and column names
    h = heatmap(inverse_confusion_matrix);  % Plot the heatmap

    h.CellLabelFormat = '%.2f';

    % Set the X and Y tick labels
    h.XDisplayLabels = colNames;
    h.YDisplayLabels = rowNames;

    % Set the X and Y axis labels
    h.XLabel = 'Fitted models';
    h.YLabel = 'Simulated models';

    h.FontSize = graph_font;
    h.FontName = 'Arial';

end;    %plot data?
fprintf('');
%%%end plot block






disp('audi5000');
%%%%%%%%%%%%%end main body%%%%%%%%%%%%%%%%%%%%%








%%%%%%%%%%%%%Start f_fitparams%%%%%%%%%%%%%%%%%%%%%
function ll = f_fitparams(params, input, which_model, this_simed_models_draws);

%takes the parameters, the input struct from which we need the
%sequence inputs, and which_model, which is the function handle to the
%model we'll use this time

ll = 0;
for sequence = 1:numel(this_simed_models_draws);

    %prepare model for fitting
    input.sampleSeries = squeeze(input.simed_subs(input.current_subject).sequences(sequence,:));    %mu ans sigma should still be loaded into input from preceding simulation code

    %get action probabilities for this sequence
    cprob = which_model(params, input);

    if sum(sum(isnan(cprob))) > 0;
        fprintf('');
    end;

    %get simed draws for this sequence
    listDraws = this_simed_models_draws(sequence);

    %Compute ll
    if listDraws == 1;  %If only one draw
        ll = ll - 0 - log(cprob(listDraws, 2));
    else
        ll = ll - sum(log(cprob((1:listDraws-1), 1))) - log(cprob(listDraws, 2));
    end;

end;    %sequence loop
%%%%%%%%%%%%%end f_fitparams%%%%%%%%%%%%%%%%%%%%%











%%%%%%%%%%%%%Start compute_behaviour_CS%%%%%%%%%%%%%%%%%%%%%
function cprob = compute_behaviour_CS(params,input);

%input.mu: prior mean (from simulation)
%input.sig: prior variance (from simulation)
%input.N: sequence length
%input.sample_series (sequence)

%computes probabilities of take / reject for one option sequence only

%params
input.Cs = params(1);
input.beta = params(2);

sdevs = 8;
dx = 2*sdevs*sqrt(input.sig)/100;
x = ((input.mu - sdevs*sqrt(input.sig)) + dx : dx : ...
    (input.mu + sdevs*sqrt(input.sig)))';

%inialise action values for stop and continue
choiceCont = zeros(1, input.N);
choiceStop = zeros(1, input.N);

for ts = 1:input.N;  %Loop through options

    %get action values for this option using backwards induction
    [expectedStop, expectedCont] = rnkBackWardInduction(input, ts, x);

    choiceCont(ts) = expectedCont(ts);
    choiceStop(ts) = expectedStop(ts);

    %save action probabilities
    choiceValues(ts,:) = [choiceCont(ts) choiceStop(ts)];
    cprob(ts,:) = exp(input.beta*choiceValues(ts,:))./sum(exp(input.beta*choiceValues(ts,:)));

end;


%     cprob(2,end) = 1; %force stopping on last option at least

fprintf('');
%%%%%%%%%%%%%End compute_behaviour_CS%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%Start compute_behaviour_CO%%%%%%%%%%%%%%%%%%%%%
function cprob = compute_behaviour_CO(params,input);

estimated_cutoff = round(params(1));
if estimated_cutoff < 1; estimated_cutoff = 1; end;
if estimated_cutoff > input.N; estimated_cutoff = input.N; end;

input.beta = params(2);

%initialise all sequence positions to zero/continue (value of stopping zero)
choiceStop = zeros(1,input.N);

%find seq vals greater than the max in the period before cutoff and give these candidates a maximal stopping value of 1
choiceStop(1,find( input.sampleSeries > max(input.sampleSeries(1:estimated_cutoff)) ) ) = 1;

%set the last position to 1, whether it's greater than the best in the learning period or not
choiceStop(1,input.N) = 1;

%find first index that is a candidate ....
num_samples = find(choiceStop == 1,1,'first');   %assign output num samples for cut off model

%Reverse 0s and 1's
choiceCont = double(~choiceStop);

%save action probabilities
choiceValues = [choiceCont; choiceStop];
cprob = exp(input.beta*choiceValues)./sum(exp(input.beta*choiceValues));
cprob(2,end) = 1; %force stopping on last option at least
cprob = cprob';

%%%%%%%%%%%%%End compute_behaviour_CO%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%Start compute_behaviour_BP%%%%%%%%%%%%%%%%%%%%%
function cprob = compute_behaviour_BP(params,input);

%input.mu: prior mean (from simulation)
%input.sig: prior variance (from simulation)
%input.N: sequence length
%input.sample_series (sequence)

%computes probabilities of take / reject for one option sequence only

%params
input.mu = input.mu + params(1);
input.beta = params(2);
input.Cs = 0;


sdevs = 8;
dx = 2*sdevs*sqrt(input.sig)/100;
x = ((input.mu - sdevs*sqrt(input.sig)) + dx : dx : ...
    (input.mu + sdevs*sqrt(input.sig)))';

%inialise action values for stop and continue
choiceCont = zeros(1, input.N);
choiceStop = zeros(1, input.N);

for ts = 1:input.N;  %Loop through options

    %get action values for this option using backwards induction
    [expectedStop, expectedCont] = rnkBackWardInduction(input, ts, x);

    choiceCont(ts) = expectedCont(ts);
    choiceStop(ts) = expectedStop(ts);

    %save action probabilities
    choiceValues(ts,:) = [choiceCont(ts) choiceStop(ts)];
    cprob(ts,:) = exp(input.beta*choiceValues(ts,:))./sum(exp(input.beta*choiceValues(ts,:)));

end;
%%%%%%%%%%%%%End compute_behaviour_BP%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%start rnkBackWardInduction%%%%%%%%%%%%%%%%%%%%%
% function [expectedStop, expectedCont, expectedUtility] = rnkBackWardInduction(sampleSeries, ts, priorProb, listLength, x, Cs, Generate_params)
function [expectedStop, expectedCont, expectedUtility] = rnkBackWardInduction(input, ts, x)

N = input.N;
sampleSeries = input.sampleSeries;
priorProb.mu = input.mu;
priorProb.sig = input.sig;
priorProb.kappa = 2;
priorProb.nu = 1;

Nx = length(x);

%set payoffs for different ranks to be the option values themselves
payoff = sort(sampleSeries,'descend');   %sort the sample value
payoff = (payoff - min(payoff))/(max(payoff)-min(payoff));
maxPayRank = numel(payoff);
temp = [payoff zeros(1, 1000)];
payoff = temp;

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

% ties = 0;
% if length(unique(sampleSeries(1:ts))) < ts
%     ties = 1;
% end

mxv = ts;
if mxv > maxPayRank
    mxv = maxPayRank;
end

rnkv = [Inf*ones(1,1); rnkvl(1:mxv)'; -Inf*ones(20, 1)];

[postProb] = normInvChi(priorProb, data);

postProb.mu ...
    = postProb.mu; %...Then add constant to the posterior mean (will be zero if not optimism model)

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
            %             fprintf('accessing utStop at %d value x %.2f\n', zi, x);
            zi = length(utStop) - 1;
        end

        utStop = utStop(zi+1)*ones(Nx, 1);

    end

    utCont = utCont - input.Cs;

    utility(:, ti)      = max([utStop utCont], [], 2);
    expectedUtility(ti) = px'*utility(:,ti);

    expectedStop(ti)    = px'*utStop;
    expectedCont(ti)    = px'*utCont;

end
%%%%%%%%%%%%%End rnkBackwardInduction%%%%%%%%%%%%%%%%%%%%%










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
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function prob_y = posteriorPredictive(y, postProb)

tvar = (1 + postProb.kappa)*postProb.sig/postProb.kappa;

sy = (y - postProb.mu)./sqrt(tvar);

prob_y = tpdf(sy, postProb.nu);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%Start make_sequences%%%%%%%%%%%%%%%%%%%%%
function [simed_subs, seq_length] = make_sequences;

%returns a structure with an element per subject containing field for the
%prior mean, prior var and sequences: num_sequences*seq_length matrix of simulated sequences


num_subs = 1000;   %So this will be per parameter value
num_seqs = 5;
seq_length = 8; %hybrid might've usually used 12 but imageTask always uses 8!
num_vals = 426;  %How many items in phase 1 and available as options in sequences? I've used before 426 or 90
rating_bounds = [1 100];    %What is min and max of rating scale?
rating_grand_mean = 40;     %Individual subjects' rating means will jitter around this (50 or 39.5. The latter comes from the midpoint between NEW hybrid SV ratings mean (30) and normalised price mean (49.2)
rating_mean_jitter = 5;     %How much to jitter participant ratings means on average?
rating_grand_std = 20;       %Individual subjects' rating std devs will jitter around this (5 or 18, the latter is the midpoint b/n NEW hybrid SV and OV)
rating_var_jitter = 2;     %How much to jitter participant ratings vars on average?

for sub = 1:num_subs;

    %Make the moments of the prior distribution slightly different for this subject
    this_sub_rating_mean = rating_grand_mean + normrnd( 0, rating_mean_jitter );
    this_sub_rating_std = rating_grand_std + normrnd( 0, rating_var_jitter );

    %Generate a truncated normal distribution of option values
    pd = truncate(makedist('Normal','mu',this_sub_rating_mean,'sigma',this_sub_rating_std),rating_bounds(1),rating_bounds(2));  %creates distribution object for population level of values
    phase1 = random(pd,num_vals,1); %generates values from distribution object to populate the ratings in phase 1 or total prices that could be sampled

    simed_subs(sub).mu = mean(phase1);
    simed_subs(sub).sig = var(phase1);

    simed_subs(sub).sequences = reshape(...
        phase1(1:num_seqs*seq_length,1) ...
        ,num_seqs ...
        ,seq_length ...
        );

end;    %Each subject to create stimuli
%%%%%%%%%%%%%End make_sequences%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%start, boxplot_nick%%%%%%%%%
function boxplot_nick(boxplot_details)

%I had to make my own boxplot function because matlab's built in one messes
%up the subplot sizes. Unusable. Also, it doesn't support transparent boxes so
%instead of adding a patch on a boxplot, here we just make a boxplot from a
%transparent patch.

figure(boxplot_details.fig);
hold(boxplot_details.sp, 'on');

data = boxplot_details.these_data;
x_val = boxplot_details.x_val;
box_width = boxplot_details.width;

% Get the statistics for each group (quartiles, median, etc.)
q = prctile(data, [25, 50, 75]);
iqr = q(3) - q(1);  % Interquartile range (IQR)
whisker_min = max(min(data), q(1) - 1.5 * iqr);  % Lower whiskermaxVal = max(data(:, x_val));
whisker_max = min(max(data), q(3) + 1.5 * iqr);  % Lower whiskermaxVal = max(data(:, x_val));

% Create the box using patch (which supports FaceAlpha for transparency)
box_x = [x_val - box_width, x_val + box_width, x_val + box_width, x_val - box_width];
box_y = [q(1), q(1), q(3), q(3)];
patch(box_x, box_y, boxplot_details.internalColor, 'EdgeColor', boxplot_details.externalColor, 'LineWidth', boxplot_details.lineWidth, 'FaceAlpha', boxplot_details.alpha, 'Parent', boxplot_details.sp);

% Plot the median line
plot([x_val-box_width, x_val+box_width], [q(2), q(2)], 'Color', boxplot_details.externalColor, 'LineWidth', boxplot_details.lineWidth);

% Plot the whiskers
plot([x_val, x_val], [whisker_min, q(1)], 'Color', boxplot_details.externalColor, 'LineWidth', boxplot_details.lineWidth+.5);
plot([x_val, x_val], [q(3), whisker_max], 'Color', boxplot_details.externalColor, 'LineWidth', boxplot_details.lineWidth+.5);
%%%%%%%%%%end, boxplot_nick%%%%%%%%%










