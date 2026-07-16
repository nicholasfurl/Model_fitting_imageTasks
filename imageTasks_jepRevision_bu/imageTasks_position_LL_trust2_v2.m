function imageTasks_position_LL_trust2_v2()
% imageTasks_position_LL_trust2_v2
%
% Post hoc decomposition of fitted model likelihoods by sequence position
% for Trustworthiness dataset 2.
%
% This script DOES NOT refit models. It loads the saved fitted parameters
% from out_imageTask_trust_2_COCSBMP2_20250103.mat and re-scores the observed
% continue/stop decisions using the same choice-value machinery used by
% imageTask_run_models.m. The decision-wise negative log likelihoods are then
% aggregated by sequence position.
%
% Important interpretation:
%   - This is a position-wise decomposition of static fitted models.
%   - It is not a model-switching or strategy-transition model.
%   - Later positions are conditional on a participant/trial having reached
%     those positions, so N decreases across sequence position.

clc;
warning('off','all');

% -------------------------------------------------------------------------
% CONFIG
% -------------------------------------------------------------------------
cfg.outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\outputs';
cfg.file = [cfg.outpath filesep 'out_imageTask_trust_2_COCSBMP2_20250103.mat'];
cfg.outdir = [cfg.outpath filesep 'position_ll_trust2_out'];

% Saved Trustworthiness 2 model order is generated from do_models = [1 2 6]:
%   model 1 = CO / cutoff
%   model 2 = Cs / cost to sample
%   model 3 = BPM / biased prior
% Use the same conceptual order as the manuscript figures/RFX analyses.
cfg.model_indices = [2 1 3];
cfg.model_names = {'Cost to sample','Cut off','Biased prior'};

% Validation tolerance only controls warning text. The script still writes
% output even if validation fails, so inspect the validation messages first.
cfg.validation_warning_tol = 1e-4;

% -------------------------------------------------------------------------
% LOAD
% -------------------------------------------------------------------------
if ~exist(cfg.outdir,'dir'); mkdir(cfg.outdir); end
if ~isfile(cfg.file)
    error('Could not find file:\n%s\nEdit cfg.file at the top of this script.', cfg.file);
end

S = load(cfg.file,'Generate_params');
Generate_params = S.Generate_params;

num_models = numel(cfg.model_indices);
num_seqs   = size(Generate_params.seq_vals,1);
seq_length = size(Generate_params.seq_vals,2);
num_subs   = size(Generate_params.seq_vals,3);

fprintf('\nTrustworthiness dataset 2 sequence-position likelihood decomposition\n');
fprintf('File: %s\n', cfg.file);
fprintf('Sequences: %d | positions: %d | participants: %d\n', num_seqs, seq_length, num_subs);

% -------------------------------------------------------------------------
% RE-SCORE EACH MODEL USING SAVED FITTED PARAMETERS
% -------------------------------------------------------------------------
all_nll_decision = NaN(num_seqs, seq_length, num_subs, num_models);
all_recomputed_nll = NaN(num_subs, num_models);
validation = NaN(num_models,3); % max abs err, mean abs err, corr saved/recomputed

for mi = 1:num_models
    model_index = cfg.model_indices(mi);
    fprintf('\nRe-scoring saved model %d (%s) as %s\n', ...
        model_index, Generate_params.model(model_index).name, cfg.model_names{mi});

    [nll_decision, recomputed_nll] = rescore_model_by_position(Generate_params, model_index);
    all_nll_decision(:,:,:,mi) = nll_decision;
    all_recomputed_nll(:,mi) = recomputed_nll;

    saved_nll = Generate_params.model(model_index).ll(:);
    valid = isfinite(saved_nll) & isfinite(recomputed_nll(:));
    err = recomputed_nll(valid) - saved_nll(valid);
    validation(mi,1) = max(abs(err));
    validation(mi,2) = mean(abs(err));
    if sum(valid) > 2
        temp = corrcoef(saved_nll(valid), recomputed_nll(valid));
        validation(mi,3) = temp(1,2);
    end

    fprintf('  Validation against saved summed NLL: max abs err = %.12g, mean abs err = %.12g, r = %.6f\n', ...
        validation(mi,1), validation(mi,2), validation(mi,3));

    if validation(mi,1) > cfg.validation_warning_tol
        warning('Validation error for %s is larger than tolerance. Inspect before interpreting position-wise LLs.', cfg.model_names{mi});
    end
end

% -------------------------------------------------------------------------
% AGGREGATE BY SEQUENCE POSITION
% -------------------------------------------------------------------------
n_by_pos = NaN(seq_length, num_models);
sum_nll_by_pos = NaN(seq_length, num_models);
mean_nll_by_pos = NaN(seq_length, num_models);
mean_ll_by_pos = NaN(seq_length, num_models);

for mi = 1:num_models
    for pos = 1:seq_length
        this_slice = squeeze(all_nll_decision(:,pos,:,mi));
        valid = isfinite(this_slice);
        n_by_pos(pos,mi) = sum(valid(:));
        if n_by_pos(pos,mi) > 0
            sum_nll_by_pos(pos,mi) = sum(this_slice(valid));
            mean_nll_by_pos(pos,mi) = sum_nll_by_pos(pos,mi) ./ n_by_pos(pos,mi);
            mean_ll_by_pos(pos,mi) = -mean_nll_by_pos(pos,mi);
        end
    end
end

delta_mean_nll = mean_nll_by_pos - repmat(min(mean_nll_by_pos,[],2),1,num_models);
[~, best_idx] = min(mean_nll_by_pos,[],2);
best_model = cfg.model_names(best_idx)';

fprintf('\nBest model by sequence position using mean NLL per observed decision:\n');
for pos = 1:seq_length
    fprintf('  Position %d: %s (N decisions = %d)\n', pos, best_model{pos}, n_by_pos(pos,1));
end

% -------------------------------------------------------------------------
% WRITE TABLE
% -------------------------------------------------------------------------
Position = (1:seq_length)';
N_decisions = n_by_pos(:,1); % same across models when all models use all observed draws
Cost_to_sample_mean_NLL = mean_nll_by_pos(:,1);
Cut_off_mean_NLL        = mean_nll_by_pos(:,2);
Biased_prior_mean_NLL   = mean_nll_by_pos(:,3);
Cost_to_sample_delta_NLL = delta_mean_nll(:,1);
Cut_off_delta_NLL        = delta_mean_nll(:,2);
Biased_prior_delta_NLL   = delta_mean_nll(:,3);
Best_model = best_model;

T = table(Position, N_decisions, ...
    Cost_to_sample_mean_NLL, Cut_off_mean_NLL, Biased_prior_mean_NLL, ...
    Cost_to_sample_delta_NLL, Cut_off_delta_NLL, Biased_prior_delta_NLL, ...
    Best_model);

csv_file = [cfg.outdir filesep 'trust2_position_likelihood_summary.csv'];
writetable(T, csv_file);
fprintf('\nWrote summary CSV:\n  %s\n', csv_file);

% Also write a long table that is easier to plot outside MATLAB if desired.
LongPosition = [];
LongModel = {};
LongN = [];
LongMeanNLL = [];
LongDeltaNLL = [];
for pos = 1:seq_length
    for mi = 1:num_models
        LongPosition(end+1,1) = pos; %#ok<AGROW>
        LongModel{end+1,1} = cfg.model_names{mi}; %#ok<AGROW>
        LongN(end+1,1) = n_by_pos(pos,mi); %#ok<AGROW>
        LongMeanNLL(end+1,1) = mean_nll_by_pos(pos,mi); %#ok<AGROW>
        LongDeltaNLL(end+1,1) = delta_mean_nll(pos,mi); %#ok<AGROW>
    end
end
Tlong = table(LongPosition, LongModel, LongN, LongMeanNLL, LongDeltaNLL, ...
    'VariableNames', {'Position','Model','N_decisions','Mean_NLL','Delta_mean_NLL'});
long_csv_file = [cfg.outdir filesep 'trust2_position_likelihood_long.csv'];
writetable(Tlong, long_csv_file);
fprintf('Wrote long-format CSV:\n  %s\n', long_csv_file);

% -------------------------------------------------------------------------
% SAVE MAT OUTPUT
% -------------------------------------------------------------------------
mat_file = [cfg.outdir filesep 'trust2_position_likelihood_results.mat'];
save(mat_file, 'cfg', 'n_by_pos', 'sum_nll_by_pos', 'mean_nll_by_pos', ...
    'mean_ll_by_pos', 'delta_mean_nll', 'best_model', 'all_nll_decision', ...
    'all_recomputed_nll', 'validation');
fprintf('Wrote MAT results:\n  %s\n', mat_file);

% -------------------------------------------------------------------------
% PLOTS
% -------------------------------------------------------------------------
try
    f1 = figure('Color','w','Name','Trust 2 position log likelihood'); hold on; box off;
    for mi = 1:num_models
        plot(Position, mean_ll_by_pos(:,mi), '-o', 'LineWidth', 1.5, 'MarkerSize', 5);
    end
    xlabel('Sequence position');
    ylabel('Mean log likelihood per observed decision');
    title('Trustworthiness dataset 2: fit by sequence position');
    legend(cfg.model_names, 'Location', 'best');
    set(gca,'FontName','Arial','FontSize',10,'XTick',Position);
    saveas(f1, [cfg.outdir filesep 'trust2_mean_log_likelihood_by_position.png']);

    f2 = figure('Color','w','Name','Trust 2 position delta NLL'); hold on; box off;
    for mi = 1:num_models
        plot(Position, delta_mean_nll(:,mi), '-o', 'LineWidth', 1.5, 'MarkerSize', 5);
    end
    xlabel('Sequence position');
    ylabel('\Delta mean negative log likelihood');
    title('Trustworthiness dataset 2: relative fit by sequence position');
    legend(cfg.model_names, 'Location', 'best');
    set(gca,'FontName','Arial','FontSize',10,'XTick',Position);
    plot([min(Position) max(Position)],[0 0],'k:');
    saveas(f2, [cfg.outdir filesep 'trust2_delta_mean_NLL_by_position.png']);
catch ME
    warning('Could not make/save plots: %s', ME.message);
end

fprintf('\nDone. Outputs written to:\n  %s\n', cfg.outdir);

end % main function

% =========================================================================
% Helper: re-score one model by observed decision position
% =========================================================================
function [nll_decision, recomputed_nll] = rescore_model_by_position(Generate_params, model_index)

num_seqs   = size(Generate_params.seq_vals,1);
seq_length = size(Generate_params.seq_vals,2);
num_subs   = size(Generate_params.seq_vals,3);

nll_decision = NaN(num_seqs, seq_length, num_subs);
recomputed_nll = NaN(num_subs,1);

for sub = 1:num_subs

    GP = Generate_params;
    GP.current_model = model_index;
    GP.num_subs_to_run = sub;
    GP = set_model_to_participant_estimates(GP, model_index, sub);

    sub_total_nll = 0;

    for seq = 1:num_seqs
        listDraws = Generate_params.num_samples(seq, sub);
        if ~isfinite(listDraws) || listDraws < 1
            continue;
        end
        listDraws = round(listDraws);
        if listDraws > seq_length
            listDraws = seq_length;
        end

        [choiceStop, choiceCont] = get_choice_values_for_sequence(GP, seq, sub);
        b = GP.model(model_index).beta;

        for drawi = 1:listDraws

            zCont = b .* choiceCont(drawi);
            zStop = b .* choiceStop(drawi);
            maxz = max([zCont zStop]);
            denom = exp(zCont - maxz) + exp(zStop - maxz);
            pCont = exp(zCont - maxz) ./ denom;
            pStop = exp(zStop - maxz) ./ denom;

            if listDraws == 1
                % Observed action is stop immediately.
                action_code = 2;
            elseif drawi < listDraws
                % Observed action is continue until the final sampled item.
                action_code = 1;
            else
                % Observed action is stop at the participant's observed stopping position.
                action_code = 2;
            end

            if action_code == 1
                this_nll = -log(max(pCont, realmin));
            else
                this_nll = -log(max(pStop, realmin));
            end

            nll_decision(seq, drawi, sub) = this_nll;
            sub_total_nll = sub_total_nll + this_nll;
        end
    end

    recomputed_nll(sub) = sub_total_nll;
end

end

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

% Do cutoff model, if needed
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
% three Trustworthiness 2 models, but is retained for fidelity to the shared
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
