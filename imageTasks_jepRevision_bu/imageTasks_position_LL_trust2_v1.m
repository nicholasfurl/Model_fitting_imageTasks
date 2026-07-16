function imageTasks_position_LL_trust2_v1()
% imageTasks_position_LL_trust2_v1
%
% Post hoc decomposition of fitted model likelihoods by sequence position
% for Trustworthiness dataset 2. This script DOES NOT refit models. It
% reloads the saved fitted parameters, re-scores the observed continue/stop
% decisions using the same choice-value machinery used in fitting, and then
% aggregates negative log likelihood by sequence position.
%
% Requirements:
%   - out_imageTask_trust_2_COCSBMP2_20250103.mat on the configured path
%   - analyzeSecretaryNick_2021.m available on the MATLAB path
%
% Output:
%   - CSV summary by sequence position
%   - PNG plots of mean log likelihood and delta mean NLL by position
%   - MAT file containing arrays used for the analysis

clc;
warning('off','all');

% -------------------------------------------------------------------------
% CONFIG
% -------------------------------------------------------------------------
cfg.outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\outputs';
cfg.file = [cfg.outpath filesep 'out_imageTask_trust_2_COCSBMP2_20250103.mat'];
cfg.outdir = [cfg.outpath filesep 'position_ll_trust2_out'];

% The current model-file order used elsewhere in the JEPLMC plotting/RFX code:
% model 2 = Cost to sample, model 1 = Cut off, model 3 = Biased prior.
cfg.model_indices = [2 1 3];
cfg.model_names = {'Cost to sample','Cut off','Biased prior'};

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
    fprintf('\nRe-scoring model %d: %s\n', model_index, cfg.model_names{mi});

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

    fprintf('  Validation against saved summed NLL: max abs err = %.6g, mean abs err = %.6g, r = %.6f\n', ...
        validation(mi,1), validation(mi,2), validation(mi,3));
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
    fprintf('  Position %d: %s\n', pos, best_model{pos});
end

% -------------------------------------------------------------------------
% WRITE TABLE
% -------------------------------------------------------------------------
Position = (1:seq_length)';
N_decisions = n_by_pos(:,1); % should be same across models when all_draws settings match
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
    yline(0,'k:');
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

        [choiceStop, choiceCont] = get_choice_values_for_sequence(GP, model_index, seq, sub);
        b = GP.model(model_index).beta;

        % action_code: 1 = continue, 2 = stop, NaN = not included in the LL.
        action_code = NaN(1, seq_length);
        if listDraws == 1
            action_code(1) = 2;
        else
            if GP.model(model_index).all_draws == 1
                action_code(1:listDraws-1) = 1;
                action_code(listDraws) = 2;
            else
                action_code(listDraws-1) = 1;
                action_code(listDraws) = 2;
            end
        end

        for pos = find(isfinite(action_code))
            zCont = b .* choiceCont(pos);
            zStop = b .* choiceStop(pos);
            maxz = max([zCont zStop]);
            denom = exp(zCont - maxz) + exp(zStop - maxz);
            pCont = exp(zCont - maxz) ./ denom;
            pStop = exp(zStop - maxz) ./ denom;

            if action_code(pos) == 1
                this_nll = -log(max(pCont, realmin));
            else
                this_nll = -log(max(pStop, realmin));
            end

            nll_decision(seq, pos, sub) = this_nll;
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
% =========================================================================
function [choiceStop, choiceCont] = get_choice_values_for_sequence(GP, model_index, sequence, sub)

seq_length = size(GP.seq_vals,2);
list.allVals = squeeze(GP.seq_vals(sequence,:,sub));
list.vals = list.allVals;

GP.PriorMean = mean(GP.ratings(:,sub));
GP.PriorVar  = var(GP.ratings(:,sub));

% Cutoff heuristic
if GP.model(model_index).identifier == 1
    this_seq_vals = list.allVals;
    choiceStop = zeros(1, seq_length);
    estimated_cutoff = round(GP.model(model_index).cutoff);
    if estimated_cutoff < 1; estimated_cutoff = 1; end
    if estimated_cutoff > seq_length; estimated_cutoff = seq_length; end

    choiceStop(1, find(this_seq_vals > max(this_seq_vals(1:estimated_cutoff)))) = 1;
    choiceStop(1, seq_length) = 1;
    choiceCont = double(~choiceStop);
else
    [choiceStop, choiceCont] = analyzeSecretaryNick_2021(GP, list);
end

choiceStop = choiceStop(:)';
choiceCont = choiceCont(:)';

end
