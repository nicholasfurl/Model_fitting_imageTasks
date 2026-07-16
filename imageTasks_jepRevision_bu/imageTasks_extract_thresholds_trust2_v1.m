function imageTasks_extract_thresholds_trust2_v1()
% imageTasks_extract_thresholds_trust2_v1
%
% Extracts fitted-model action values for Trustworthiness dataset 2.
%
% This script DOES NOT refit models. It loads the saved fitted parameters
% from out_imageTask_trust_2_COCSBMP2_20250103.mat and reconstructs the
% stop and continue action values for every participant, sequence, position
% and fitted Bayesian model.
%
% The continuation action value Q_continue functions as a model-derived
% acceptance threshold in action-value space: stopping becomes favoured when
% Q_stop(current option) exceeds Q_continue.
%
% Output:
%   trust2_threshold_q_values.mat
%   trust2_threshold_q_values_long.csv
%
% The companion script imageTasks_analyse_thresholds_trust2_v1.m loads these
% outputs, makes figures and fits linear/quadratic threshold-shape models.

clc;
warning('off','all');

% -------------------------------------------------------------------------
% CONFIG
% -------------------------------------------------------------------------
cfg.outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\outputs';
cfg.file    = [cfg.outpath filesep 'out_imageTask_trust_2_COCSBMP2_20250103.mat'];
cfg.outdir  = [cfg.outpath filesep 'thresholds_trust2_out'];

% Saved Trustworthiness 2 model order is generated from do_models = [1 2 6]:
%   model 1 = CO / cut off
%   model 2 = Cs / cost to sample
%   model 3 = BPM / biased prior
%
% We extract thresholds for the two Bayesian one-parameter models. The cut
% off heuristic has a different algorithm and its 0/1 action code is not a
% comparable continuation-value threshold.
cfg.threshold_model_indices = [2 3];
cfg.threshold_model_names   = {'Cost-to-sample','Biased-prior'};

% But use all three fitted models to define each participant's best-fitting
% group, matching the manuscript model comparison.
cfg.comparison_model_indices = [2 1 3];
cfg.comparison_model_names   = {'Cost-to-sample','Cut off','Biased-prior'};

% -------------------------------------------------------------------------
% LOAD
% -------------------------------------------------------------------------
if ~exist(cfg.outdir,'dir'); mkdir(cfg.outdir); end
if ~isfile(cfg.file)
    error('Could not find file:\n%s\nEdit cfg.file at the top of this script.', cfg.file);
end

S = load(cfg.file,'Generate_params');
Generate_params = S.Generate_params;

num_models = numel(cfg.threshold_model_indices);
num_seqs   = size(Generate_params.seq_vals,1);
seq_length = size(Generate_params.seq_vals,2);
num_subs   = size(Generate_params.seq_vals,3);

fprintf('\nTrustworthiness dataset 2 threshold/action-value extraction\n');
fprintf('File: %s\n', cfg.file);
fprintf('Sequences: %d | positions: %d | participants: %d\n', num_seqs, seq_length, num_subs);

% -------------------------------------------------------------------------
% BEST-FITTING MODEL GROUPS FROM SAVED FULL-SEQUENCE FITS
% -------------------------------------------------------------------------
NLL_compare = NaN(num_subs, numel(cfg.comparison_model_indices));
for mi = 1:numel(cfg.comparison_model_indices)
    model_index = cfg.comparison_model_indices(mi);
    NLL_compare(:,mi) = Generate_params.model(model_index).ll(:);
end
[~, best_col] = min(NLL_compare, [], 2);
best_fit_model = cfg.comparison_model_names(best_col)';

fprintf('\nBest-fitting groups from full-sequence NLLs:\n');
for mi = 1:numel(cfg.comparison_model_names)
    fprintf('  %s: %d participants\n', cfg.comparison_model_names{mi}, sum(best_col == mi));
end

% -------------------------------------------------------------------------
% EXTRACT ACTION VALUES
% -------------------------------------------------------------------------
Q_stop      = NaN(num_seqs, seq_length, num_subs, num_models);
Q_continue  = NaN(num_seqs, seq_length, num_subs, num_models);
Q_diff_stop_minus_continue = NaN(num_seqs, seq_length, num_subs, num_models);
choice_position = NaN(num_seqs, num_subs);
observed_action_code = NaN(num_seqs, seq_length, num_subs); % 1 = continue, 2 = stop/choose, NaN = not observed
reached_position = false(num_seqs, seq_length, num_subs);

for sub = 1:num_subs
    if mod(sub,10) == 0 || sub == 1
        fprintf('  Participant %d / %d\n', sub, num_subs);
    end

    for seq = 1:num_seqs
        listDraws = Generate_params.num_samples(seq, sub);
        if ~isfinite(listDraws) || listDraws < 1
            continue;
        end
        listDraws = round(listDraws);
        if listDraws > seq_length
            listDraws = seq_length;
        end
        choice_position(seq, sub) = listDraws;

        for pos = 1:listDraws
            reached_position(seq,pos,sub) = true;
            if pos < listDraws
                observed_action_code(seq,pos,sub) = 1; % continue
            else
                observed_action_code(seq,pos,sub) = 2; % stop/choose, including forced terminal choice
            end
        end
    end

    for mi = 1:num_models
        model_index = cfg.threshold_model_indices(mi);

        GP = Generate_params;
        GP.current_model = model_index;
        GP.num_subs_to_run = sub;
        GP = set_model_to_participant_estimates(GP, model_index, sub);

        for seq = 1:num_seqs
            [choiceStop, choiceCont] = get_choice_values_for_sequence(GP, seq, sub);
            Q_stop(seq,:,sub,mi) = choiceStop;
            Q_continue(seq,:,sub,mi) = choiceCont;
            Q_diff_stop_minus_continue(seq,:,sub,mi) = choiceStop - choiceCont;
        end
    end
end

% -------------------------------------------------------------------------
% LONG TABLE FOR ANALYSIS/PLOTTING
% -------------------------------------------------------------------------
Subject = [];
Sequence = [];
Position = [];
Model = {};
BestFitModel = {};
ChoicePosition = [];
Reached = [];
ObservedActionCode = [];
ObservedAction = {};
TerminalPosition = [];
PositionFromChoice = [];
StepsBeforeChoice = [];
ProportionToChoice = [];
QStop = [];
QContinue = [];
QStopMinusQContinue = [];

for sub = 1:num_subs
    for seq = 1:num_seqs
        cp = choice_position(seq, sub);
        for pos = 1:seq_length
            for mi = 1:num_models
                Subject(end+1,1) = sub; %#ok<AGROW>
                Sequence(end+1,1) = seq; %#ok<AGROW>
                Position(end+1,1) = pos; %#ok<AGROW>
                Model{end+1,1} = cfg.threshold_model_names{mi}; %#ok<AGROW>
                BestFitModel{end+1,1} = best_fit_model{sub}; %#ok<AGROW>
                ChoicePosition(end+1,1) = cp; %#ok<AGROW>
                Reached(end+1,1) = reached_position(seq,pos,sub); %#ok<AGROW>
                ObservedActionCode(end+1,1) = observed_action_code(seq,pos,sub); %#ok<AGROW>

                if observed_action_code(seq,pos,sub) == 1
                    ObservedAction{end+1,1} = 'continue'; %#ok<AGROW>
                elseif observed_action_code(seq,pos,sub) == 2
                    ObservedAction{end+1,1} = 'stop'; %#ok<AGROW>
                else
                    ObservedAction{end+1,1} = 'not reached'; %#ok<AGROW>
                end

                TerminalPosition(end+1,1) = (pos == seq_length); %#ok<AGROW>
                PositionFromChoice(end+1,1) = pos - cp; %#ok<AGROW> % 0 at choice, negative before choice, positive after choice
                StepsBeforeChoice(end+1,1) = cp - pos; %#ok<AGROW> % 0 at choice, positive before choice, negative after choice
                ProportionToChoice(end+1,1) = pos ./ cp; %#ok<AGROW>
                QStop(end+1,1) = Q_stop(seq,pos,sub,mi); %#ok<AGROW>
                QContinue(end+1,1) = Q_continue(seq,pos,sub,mi); %#ok<AGROW>
                QStopMinusQContinue(end+1,1) = Q_diff_stop_minus_continue(seq,pos,sub,mi); %#ok<AGROW>
            end
        end
    end
end

T = table(Subject, Sequence, Position, Model, BestFitModel, ChoicePosition, ...
    Reached, ObservedActionCode, ObservedAction, TerminalPosition, ...
    PositionFromChoice, StepsBeforeChoice, ProportionToChoice, ...
    QStop, QContinue, QStopMinusQContinue);

% -------------------------------------------------------------------------
% SAVE
% -------------------------------------------------------------------------
mat_file = [cfg.outdir filesep 'trust2_threshold_q_values.mat'];
csv_file = [cfg.outdir filesep 'trust2_threshold_q_values_long.csv'];
save(mat_file, 'cfg', 'Generate_params', 'Q_stop', 'Q_continue', ...
    'Q_diff_stop_minus_continue', 'choice_position', 'reached_position', ...
    'observed_action_code', 'best_fit_model', 'NLL_compare', 'T', '-v7.3');
writetable(T, csv_file);

fprintf('\nWrote MAT output:\n  %s\n', mat_file);
fprintf('Wrote long CSV:\n  %s\n', csv_file);
fprintf('\nDone. Use imageTasks_analyse_thresholds_trust2_v1.m for plotting/regressions.\n');

end % main function

% =========================================================================
% Helper: place one participant's estimated params into the model struct
% =========================================================================
function GP = set_model_to_participant_estimates(GP, model_index, sub)

it = 1;
fields = fieldnames(GP.model(model_index));
free_fields = GP.model(model_index).this_models_free_parameters;

for field = free_fields(:)'
    GP.model(model_index).(fields{field}) = GP.model(model_index).estimated_params(sub,it);
    it = it + 1;
end

% beta is stored last in estimated_params, after the theoretical parameter(s)
GP.model(model_index).beta = GP.model(model_index).estimated_params(sub,end);

end

% =========================================================================
% Helper: compute continue/stop choice values for one sequence
% This follows imageTask_run_models.m for the 2024 image-task fits.
% =========================================================================
function [choiceStop, choiceCont] = get_choice_values_for_sequence(GP, sequence, sub)

% just one sequence
list_vals = squeeze(GP.seq_vals(sequence,:,sub));

% get prior dist moments
GP.PriorMean = nanmean(GP.ratings(:,sub));
GP.PriorVar = nanvar(GP.ratings(:,sub));

% Do cutoff model, if needed. This branch is not used by the threshold
% extraction configuration, but is retained for fidelity/re-use.
if GP.model(GP.current_model).identifier == 1

    choiceStop = zeros(1,GP.seq_length);
    estimated_cutoff = round(GP.model(GP.current_model).cutoff);
    if estimated_cutoff < 1; estimated_cutoff = 1; end
    if estimated_cutoff > GP.seq_length; estimated_cutoff = GP.seq_length; end

    % find seq vals greater than the max in the period before cutoff
    choiceStop(1,find(list_vals > max(list_vals(1:estimated_cutoff)))) = 1;

    % set the last position to 1 whether or not it beats the previous best
    choiceStop(1,GP.seq_length) = 1;

    % Reverse 0s and 1s
    choiceCont = double(~choiceStop);

else

    [choiceStop, choiceCont] = analyzeSecretary_imageTask_2024(GP, list_vals);

end

choiceStop = choiceStop(:)';
choiceCont = choiceCont(:)';

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [choiceStop, choiceCont, difVal] = analyzeSecretary_imageTask_2024(Generate_params,sampleSeries)

N = Generate_params.seq_length;
Cs = Generate_params.model(Generate_params.current_model).Cs;      % zero unless Cs model

prior.mu    = Generate_params.PriorMean + Generate_params.model(Generate_params.current_model).BP; % prior mean offset zero unless biased prior model
prior.sig   = Generate_params.PriorVar + Generate_params.model(Generate_params.current_model).BPV;
if prior.sig < 1; prior.sig = 1; end
prior.kappa = Generate_params.model(Generate_params.current_model).kappa;
prior.nu    = Generate_params.model(Generate_params.current_model).nu;

% Apply BV transformation if needed. This branch should not be used for the
% Trustworthiness 2 threshold models, but is retained for fidelity to the shared
% analysis function.
if Generate_params.model(Generate_params.current_model).identifier == 4
    sampleSeries(find(sampleSeries <= Generate_params.model(Generate_params.current_model).BVmid)) = 0;
end

[choiceStop, choiceCont, difVal, currentRnk] = computeSecretary(Generate_params, sampleSeries, prior, N, Cs); %#ok<ASGLU>
end

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

    [rnkv, rnki] = sort(sampleSeries(1:ts), 'descend'); %#ok<ASGLU>
    z = find(rnki == ts); %#ok<NASGU>

    difVal(ts) = expectedCont(ts) - expectedStop(ts);

    choiceCont(ts) = expectedCont(ts);
    choiceStop(ts) = expectedStop(ts);

end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [expectedStop, expectedCont, expectedUtility] = rnkBackWardInduction(sampleSeries, ts, priorProb, ...
    listLength, x, Cs,  Generate_params)

N = listLength;
Nx = length(x);

if Generate_params.model(Generate_params.current_model).identifier == 5

    payoff = sampleSeries;
    payoff(find(payoff <= Generate_params.model(Generate_params.current_model).BRmid)) = 0;
    payoff = sort(payoff,'descend');

else

    payoff = sort(sampleSeries,'descend');

end

% added 28/2/2025: normalise between zero and 1
if all(payoff == payoff(1))
    payoff = payoff/100;
else
    payoff = (payoff - min(payoff))/(max(payoff)-min(payoff));
end

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
rnki = z; %#ok<NASGU>

ties = 0; %#ok<NASGU>
if length(unique(sampleSeries(1:ts))) < ts
    ties = 1; %#ok<NASGU>
end

mxv = ts;
if mxv > maxPayRank
    mxv = maxPayRank;
end

rnkv = [Inf*ones(1,1); rnkvl(1:mxv)'; -Inf*ones(20, 1)];

[postProb] = normInvChi(priorProb, data);

postProb.mu ...
    = postProb.mu ...
    + Generate_params.model(Generate_params.current_model).optimism;

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

    utStop = NaN*ones(Nx, 1);

    rd = N - ti;
    id = max([(ti - ts - 1) 0]);
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
        [zv, zi] = min(abs(x - sampleSeries(ts))); %#ok<ASGLU>
        if zi + 1 > length(utStop)
            zi = length(utStop) - 1;
        end

        utStop = utStop(zi+1)*ones(Nx, 1);

    end

    utCont = utCont - Cs;

    utility(:, ti)      = max([utStop utCont], [], 2);
    expectedUtility(ti) = px'*utility(:,ti); %#ok<AGROW>

    expectedStop(ti)    = px'*utStop; %#ok<AGROW>
    expectedCont(ti)    = px'*utCont; %#ok<AGROW>

end
end

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
end

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
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function prob_y = posteriorPredictive(y, postProb)

tvar = (1 + postProb.kappa)*postProb.sig/postProb.kappa;

sy = (y - postProb.mu)./sqrt(tvar);

prob_y = tpdf(sy, postProb.nu);
end
