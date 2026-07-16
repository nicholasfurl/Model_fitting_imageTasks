%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% trust2_cs_bp_population_diagnostics.m
%
% Descriptive heterogeneity analysis for Trustworthiness dataset 2.
% Splits participants by whole-sequence best-fitting model:
%   Cs vs Cutoff vs BP
% Then focuses on Cs winners vs BP winners.
%
% Requires:
%   1) out_imageTask_trust_2_COCSBMP2_20250103.mat
%   2) trust2_position_likelihood_results.mat
%
% Does NOT refit or re-score models.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% ------------------------------------------------------------------------
% Paths
% -------------------------------------------------------------------------
outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\outputs';

fit_file = [outpath filesep 'out_imageTask_trust_2_COCSBMP2_20250103.mat'];

pos_file = [outpath filesep 'position_ll_trust2_out' filesep ...
    'trust2_position_likelihood_results.mat'];

analysis_outdir = [outpath filesep 'trust2_cs_bp_population_diagnostics'];
if ~exist(analysis_outdir, 'dir')
    mkdir(analysis_outdir);
end

%% ------------------------------------------------------------------------
% Load fitted model output
% -------------------------------------------------------------------------
S = load(fit_file, 'Generate_params');
GP = S.Generate_params;

num_seqs   = GP.num_seqs;
seq_length = GP.seq_length;
num_subs   = GP.num_subs;

model_names_all = {GP.model.name};

idx_CO = find(strcmp(model_names_all,'CO') | strcmp(model_names_all,'Cut off'), 1);
idx_Cs = find(strcmp(model_names_all,'Cs') | strcmp(model_names_all,'Cost to sample'), 1);
idx_BP = find(strcmp(model_names_all,'BPM') | strcmp(model_names_all,'BP') | strcmp(model_names_all,'Biased prior'), 1);

if isempty(idx_CO) || isempty(idx_Cs) || isempty(idx_BP)
    error('Could not find CO, Cs, and BP/BPM models in Generate_params.model.');
end

% Optional optimal / ground-truth model, usually added by run_io.
idx_Opt = find(strcmp(model_names_all,'Optimal') | strcmp(model_names_all,'Opt'), 1);

has_opt = ~isempty(idx_Opt) && isfield(GP.model(idx_Opt),'num_samples_est') ...
    && ~isempty(GP.model(idx_Opt).num_samples_est);

%% ------------------------------------------------------------------------
% Whole-sequence model winners
% -------------------------------------------------------------------------
% GP.model(m).ll is actually negative log likelihood / loss.
% Lower is better. Because the three theoretical models all have the same
% number of fitted parameters, the NLL winner is also the BIC winner.
NLL_Cs = GP.model(idx_Cs).ll(:);
NLL_CO = GP.model(idx_CO).ll(:);
NLL_BP = GP.model(idx_BP).ll(:);

NLL_all = [NLL_Cs, NLL_CO, NLL_BP];
winner_labels = {'Cost to sample','Cut off','Biased prior'};

[~, winner_col] = min(NLL_all, [], 2);

is_Cs = winner_col == 1;
is_CO = winner_col == 2;
is_BP = winner_col == 3;

fprintf('\nWhole-sequence winning model counts\n');
fprintf('  Cost to sample: %d\n', sum(is_Cs));
fprintf('  Cut off:        %d\n', sum(is_CO));
fprintf('  Biased prior:   %d\n', sum(is_BP));

% Positive means BP fits better than Cs.
BP_adv_NLL = NLL_Cs - NLL_BP;
BP_adv_BIC = 2 * BP_adv_NLL; % same k and same n_obs, so BIC difference is 2*NLL difference

%% ------------------------------------------------------------------------
% Behavioural measures
% -------------------------------------------------------------------------
obs_samples = GP.num_samples; % sequence x participant

% Optional optimal samples
if has_opt
    opt_samples = GP.model(idx_Opt).num_samples_est;

    % Be robust to accidental transposition.
    if size(opt_samples,1) ~= num_seqs && size(opt_samples,2) == num_seqs
        opt_samples = opt_samples';
    end

    sampling_bias_opt = nanmean(obs_samples - opt_samples, 1)';
else
    sampling_bias_opt = NaN(num_subs,1);
end

mean_samples = nanmean(obs_samples, 1)';

early_stop_rate        = nanmean(double(obs_samples <= 4), 1)';
late_nonterminal_rate  = nanmean(double(obs_samples >= 5 & obs_samples <= 7), 1)';
terminal_stop_rate     = nanmean(double(obs_samples == seq_length), 1)';

% Chosen value and sequence best value.
chosen_value = NaN(num_seqs, num_subs);
best_value   = squeeze(max(GP.seq_vals, [], 2)); % sequence x participant

for sub = 1:num_subs
    for seq = 1:num_seqs
        chosen_pos = obs_samples(seq,sub);
        chosen_value(seq,sub) = GP.seq_vals(seq, chosen_pos, sub);
    end
end

mean_chosen_value = nanmean(chosen_value, 1)';
mean_best_value   = nanmean(best_value, 1)';
regret            = nanmean(best_value - chosen_value, 1)';

% GP.ranks was computed by tiedrank(seq_vals), so larger rank = better item.
mean_rank_ascending = nanmean(GP.ranks, 1)';

% A more reader-friendly rank: 1 = best, 8 = worst.
mean_rank_best1 = nanmean(seq_length + 1 - GP.ranks, 1)';

prop_best_chosen = nanmean(double(chosen_value == best_value), 1)';

% Top-2 chosen: no more than one item in sequence is strictly better.
top2 = NaN(num_seqs, num_subs);
for sub = 1:num_subs
    for seq = 1:num_seqs
        vals = squeeze(GP.seq_vals(seq,:,sub));
        top2(seq,sub) = sum(vals > chosen_value(seq,sub)) <= 1;
    end
end
prop_top2_chosen = nanmean(top2, 1)';

% Participant-normalised chosen value percentile.
chosen_percentile = NaN(num_seqs, num_subs);
for sub = 1:num_subs
    ratings = GP.ratings(:,sub);
    ratings = ratings(~isnan(ratings));

    for seq = 1:num_seqs
        chosen_percentile(seq,sub) = mean(ratings <= chosen_value(seq,sub));
    end
end
mean_chosen_percentile = nanmean(chosen_percentile, 1)';

%% ------------------------------------------------------------------------
% Position-wise likelihood measures from existing output
% -------------------------------------------------------------------------
early_Cs_adv = NaN(num_subs,1); % positive = Cs better than BP in positions 2-4
late_BP_adv  = NaN(num_subs,1); % positive = BP better than Cs in positions 5-7

if exist(pos_file, 'file')
    R = load(pos_file);

    if isfield(R, 'all_nll_decision')

        % Assumes the previous position-LL script used model order:
        %   1 = Cs, 2 = Cutoff, 3 = BP
        all_nll_decision = R.all_nll_decision;

        cs_col = 1;
        bp_col = 3;

        early_pos = 2:4;
        late_pos  = 5:7;

        for sub = 1:num_subs

            nll_Cs_early = all_nll_decision(:,early_pos,sub,cs_col);
            nll_BP_early = all_nll_decision(:,early_pos,sub,bp_col);

            nll_Cs_late = all_nll_decision(:,late_pos,sub,cs_col);
            nll_BP_late = all_nll_decision(:,late_pos,sub,bp_col);

            nll_Cs_early = nll_Cs_early(isfinite(nll_Cs_early));
            nll_BP_early = nll_BP_early(isfinite(nll_BP_early));
            nll_Cs_late  = nll_Cs_late(isfinite(nll_Cs_late));
            nll_BP_late  = nll_BP_late(isfinite(nll_BP_late));

            % Positive = named model better.
            early_Cs_adv(sub) = mean(nll_BP_early) - mean(nll_Cs_early);
            late_BP_adv(sub)  = mean(nll_Cs_late)  - mean(nll_BP_late);
        end
    else
        warning('Position likelihood file found, but all_nll_decision field not found.');
    end
else
    warning('Position likelihood results file not found. Skipping early/late LL measures.');
end

%% ------------------------------------------------------------------------
% Participant-level table
% -------------------------------------------------------------------------
winner_name = winner_labels(winner_col)';

Tsub = table( ...
    (1:num_subs)', ...
    winner_name, ...
    NLL_Cs, NLL_CO, NLL_BP, ...
    BP_adv_NLL, BP_adv_BIC, ...
    mean_samples, sampling_bias_opt, ...
    early_stop_rate, late_nonterminal_rate, terminal_stop_rate, ...
    mean_chosen_value, mean_chosen_percentile, ...
    mean_rank_ascending, mean_rank_best1, ...
    regret, prop_best_chosen, prop_top2_chosen, ...
    early_Cs_adv, late_BP_adv, ...
    'VariableNames', { ...
    'Participant','Winner', ...
    'NLL_Cs','NLL_Cutoff','NLL_BP', ...
    'BP_advantage_NLL','BP_advantage_BIC', ...
    'Mean_samples','Sampling_bias_vs_optimal', ...
    'Early_stop_rate_1_4','Late_nonterminal_stop_rate_5_7','Terminal_stop_rate_8', ...
    'Mean_chosen_value','Mean_chosen_percentile', ...
    'Mean_rank_larger_is_better','Mean_rank_1_is_best', ...
    'Regret','Prop_best_chosen','Prop_top2_chosen', ...
    'Early_Cs_advantage_2_4','Late_BP_advantage_5_7'} ...
    );

disp(Tsub);

writetable(Tsub, [analysis_outdir filesep 'trust2_participant_metrics.csv']);

%% ------------------------------------------------------------------------
% Cs-vs-BP group comparisons
% -------------------------------------------------------------------------
metrics = { ...
    'Mean_samples',                    mean_samples; ...
    'Sampling_bias_vs_optimal',         sampling_bias_opt; ...
    'Early_stop_rate_1_4',              early_stop_rate; ...
    'Late_nonterminal_stop_rate_5_7',   late_nonterminal_rate; ...
    'Terminal_stop_rate_8',             terminal_stop_rate; ...
    'Mean_chosen_value',                mean_chosen_value; ...
    'Mean_chosen_percentile',           mean_chosen_percentile; ...
    'Mean_rank_larger_is_better',       mean_rank_ascending; ...
    'Mean_rank_1_is_best',              mean_rank_best1; ...
    'Regret',                           regret; ...
    'Prop_best_chosen',                 prop_best_chosen; ...
    'Prop_top2_chosen',                 prop_top2_chosen; ...
    'Early_Cs_advantage_2_4',           early_Cs_adv; ...
    'Late_BP_advantage_5_7',            late_BP_adv; ...
    'BP_advantage_NLL',                 BP_adv_NLL ...
    };

n_metrics = size(metrics,1);

Metric = cell(n_metrics,1);
Mean_Cs = NaN(n_metrics,1);
SD_Cs = NaN(n_metrics,1);
Mean_BP = NaN(n_metrics,1);
SD_BP = NaN(n_metrics,1);
BP_minus_Cs = NaN(n_metrics,1);
Cohens_d_BP_minus_Cs = NaN(n_metrics,1);
P_ttest2 = NaN(n_metrics,1);

for i = 1:n_metrics
    Metric{i} = metrics{i,1};
    x = metrics{i,2};

    x_Cs = x(is_Cs);
    x_BP = x(is_BP);

    Mean_Cs(i) = nanmean(x_Cs);
    SD_Cs(i)   = nanstd(x_Cs);
    Mean_BP(i) = nanmean(x_BP);
    SD_BP(i)   = nanstd(x_BP);

    BP_minus_Cs(i) = Mean_BP(i) - Mean_Cs(i);

    pooled_sd = sqrt(((sum(~isnan(x_Cs))-1)*nanvar(x_Cs) + ...
                      (sum(~isnan(x_BP))-1)*nanvar(x_BP)) / ...
                      (sum(~isnan(x_Cs)) + sum(~isnan(x_BP)) - 2));

    Cohens_d_BP_minus_Cs(i) = BP_minus_Cs(i) / pooled_sd;

    try
        [~,P_ttest2(i)] = ttest2(x_Cs, x_BP);
    catch
        P_ttest2(i) = NaN;
    end
end

Tgroup = table( ...
    Metric, Mean_Cs, SD_Cs, Mean_BP, SD_BP, ...
    BP_minus_Cs, Cohens_d_BP_minus_Cs, P_ttest2);

disp(Tgroup);

writetable(Tgroup, [analysis_outdir filesep 'trust2_cs_vs_bp_group_comparisons.csv']);

%% ------------------------------------------------------------------------
% Correlations: does BP evidence advantage predict behaviour?
% -------------------------------------------------------------------------
corr_metrics = { ...
    'Mean_samples',                  mean_samples; ...
    'Sampling_bias_vs_optimal',       sampling_bias_opt; ...
    'Terminal_stop_rate_8',           terminal_stop_rate; ...
    'Mean_chosen_value',              mean_chosen_value; ...
    'Mean_chosen_percentile',         mean_chosen_percentile; ...
    'Mean_rank_larger_is_better',     mean_rank_ascending; ...
    'Mean_rank_1_is_best',            mean_rank_best1; ...
    'Regret',                         regret; ...
    'Prop_best_chosen',               prop_best_chosen; ...
    'Prop_top2_chosen',               prop_top2_chosen; ...
    'Late_BP_advantage_5_7',          late_BP_adv ...
    };

Predictor = {};
Outcome = {};
R_pearson = [];
P_pearson = [];
R_spearman = [];
P_spearman = [];

predictor_names = {'NLL_Cs','NLL_BP','BP_advantage_NLL','Late_BP_advantage_5_7'};
predictor_values = {NLL_Cs, NLL_BP, BP_adv_NLL, late_BP_adv};

row = 0;
for pidx = 1:numel(predictor_names)
    pred = predictor_values{pidx};

    for midx = 1:size(corr_metrics,1)
        outcome_name = corr_metrics{midx,1};
        y = corr_metrics{midx,2};

        ok = isfinite(pred) & isfinite(y);

        if sum(ok) > 3
            [r1,p1] = corr(pred(ok), y(ok), 'Type','Pearson');
            [r2,p2] = corr(pred(ok), y(ok), 'Type','Spearman');
        else
            r1 = NaN; p1 = NaN; r2 = NaN; p2 = NaN;
        end

        row = row + 1;
        Predictor{row,1} = predictor_names{pidx};
        Outcome{row,1} = outcome_name;
        R_pearson(row,1) = r1;
        P_pearson(row,1) = p1;
        R_spearman(row,1) = r2;
        P_spearman(row,1) = p2;
    end
end

Tcorr = table(Predictor, Outcome, R_pearson, P_pearson, R_spearman, P_spearman);

disp(Tcorr);

writetable(Tcorr, [analysis_outdir filesep 'trust2_evidence_behavior_correlations.csv']);

%% ------------------------------------------------------------------------
% Stop-position distributions by winner group
% -------------------------------------------------------------------------
groups_to_plot = {'Cost to sample','Biased prior'};
group_masks = {is_Cs, is_BP};

stop_dist = NaN(seq_length, numel(groups_to_plot));
reach_rate = NaN(seq_length, numel(groups_to_plot));

for g = 1:numel(groups_to_plot)
    subs = find(group_masks{g});

    for pos = 1:seq_length
        stop_dist(pos,g) = mean(obs_samples(:,subs) == pos, 'all');
        reach_rate(pos,g) = mean(obs_samples(:,subs) >= pos, 'all');
    end
end

figure('Color',[1 1 1], 'Name','Trust 2 Cs vs BP stop distributions');

subplot(1,2,1); hold on; box off;
plot(1:seq_length, stop_dist(:,1), '-o', 'LineWidth', 1.5);
plot(1:seq_length, stop_dist(:,2), '-o', 'LineWidth', 1.5);
xlabel('Stopping position');
ylabel('Proportion of sequences');
title('Observed stopping distribution');
legend(groups_to_plot, 'Location','best');
set(gca,'FontName','Arial','FontSize',12);

subplot(1,2,2); hold on; box off;
plot(1:seq_length, reach_rate(:,1), '-o', 'LineWidth', 1.5);
plot(1:seq_length, reach_rate(:,2), '-o', 'LineWidth', 1.5);
xlabel('Sequence position');
ylabel('Proportion of decisions reached');
title('Exposure by position');
legend(groups_to_plot, 'Location','best');
set(gca,'FontName','Arial','FontSize',12);

saveas(gcf, [analysis_outdir filesep 'trust2_cs_vs_bp_stop_distribution.png']);

%% ------------------------------------------------------------------------
% Simple scatter: BP evidence advantage vs sampling bias
% -------------------------------------------------------------------------
figure('Color',[1 1 1], 'Name','BP evidence advantage vs sampling bias');
hold on; box off;

scatter(BP_adv_NLL(is_Cs), sampling_bias_opt(is_Cs), 50, 'o', 'filled');
scatter(BP_adv_NLL(is_BP), sampling_bias_opt(is_BP), 50, 'o', 'filled');

xlabel('BP evidence advantage over Cs: NLL_{Cs} - NLL_{BP}');
ylabel('Sampling bias: observed samples - optimal samples');
title('Trustworthiness dataset 2');
legend({'Cs winners','BP winners'}, 'Location','best');
set(gca,'FontName','Arial','FontSize',12);

ok = isfinite(BP_adv_NLL) & isfinite(sampling_bias_opt);
if sum(ok) > 3
    b = regress(sampling_bias_opt(ok), [ones(sum(ok),1) BP_adv_NLL(ok)]);
    xline_vals = linspace(min(BP_adv_NLL(ok)), max(BP_adv_NLL(ok)), 100);
    yhat = b(1) + b(2)*xline_vals;
    plot(xline_vals, yhat, 'k-', 'LineWidth', 1.5);
end

saveas(gcf, [analysis_outdir filesep 'trust2_bp_evidence_vs_sampling_bias.png']);

%% ------------------------------------------------------------------------
% Save all outputs
% -------------------------------------------------------------------------
save([analysis_outdir filesep 'trust2_cs_bp_population_diagnostics.mat'], ...
    'Tsub','Tgroup','Tcorr','stop_dist','reach_rate', ...
    'is_Cs','is_CO','is_BP','winner_col','winner_labels');

fprintf('\nDone. Outputs written to:\n  %s\n', analysis_outdir);