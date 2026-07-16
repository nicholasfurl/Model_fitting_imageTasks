function imageTasks_empirical_thresholds_trust2_winners_v1()
% imageTasks_empirical_thresholds_trust2_winners_v1
%
% Lee/Furl-style empirical threshold analysis for Trustworthiness dataset 2.
%
% This script uses the output from:
%   imageTasks_extract_thresholds_trust2_v1.m
%
% It estimates choice-derived thresholds by:
%   1. binning option values,
%   2. estimating P(stop/choose) within each value bin and sequence position,
%   3. fitting an increasing logistic function to each position's psychometric curve,
%   4. taking the logistic inflection point / PSE as the threshold.
%
% The analysis is performed separately for:
%   - participants best fitted by the cost-to-sample model
%   - participants best fitted by the biased-prior model
%
% For each group, the figure compares:
%   - observed participant choices
%   - fitted winning-model stop probabilities
%
% Important: the default model curve uses model stop probabilities for the
% states actually reached by the participant. This makes the participant and
% model curves condition on the same encountered states. You can change this
% using cfg.model_state_set below.

clc;
warning('off','all');

% -------------------------------------------------------------------------
% CONFIG
% -------------------------------------------------------------------------
cfg.outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\outputs';
cfg.outdir  = [cfg.outpath filesep 'thresholds_trust2_out'];
cfg.infile  = [cfg.outdir filesep 'trust2_threshold_q_values.mat'];

% Groups to compare. Cut off winners are deliberately excluded here.
cfg.groups = {'Cost-to-sample','Biased-prior'};
cfg.group_labels = {'Cost-to-sample winners','Biased-prior winners'};

% Saved model indices in Generate_params, matching extraction script.
cfg.model_index.Cost_to_sample = 2;
cfg.model_index.Biased_prior   = 3;

% Logistic threshold analysis settings.
cfg.positions = 1:7;               % exclude forced terminal position
cfg.n_bins = 8;                    % matches older threshold-analysis convention
cfg.bin_method = 'participant_quantile'; 
% Options:
%   'participant_quantile' : bins each participant's phase-1 ratings into quantiles
%   'participant_equal_width' : equal-width bins within each participant's phase-1 rating range
%   'global_equal_width' : equal-width bins using all Trust2 sequence option values
%
% The threshold is expressed in value-bin units, not raw rating units.
% With participant_quantile, a threshold of 7 means the boundary lies near
% the upper quantile bins of that participant's own rating distribution.

% Which states should the model psychometric curve use?
cfg.model_state_set = 'participant_reached';
% Options:
%   'participant_reached' : model P(stop) only for states reached by participants
%   'all_positions'       : model P(stop) for every sequence position 1-7

cfg.min_trials_per_bin = 3;      % ignore sparse bins in logistic fitting
cfg.min_total_trials   = 20;     % minimum total observations to fit position
cfg.min_successes      = 1e-6;   % need at least some stopping probability
cfg.min_failures       = 1e-6;   % need at least some continuing probability

% Plot settings.
cfg.make_psychometric_figure = true;
cfg.save_figures = true;
cfg.use_greyscale = false;       % set true for grayscale output
cfg.plot_raw_bin_points = true;  % show threshold figure only; psychometric curves always show points

% -------------------------------------------------------------------------
% LOAD
% -------------------------------------------------------------------------
if ~isfile(cfg.infile)
    error(['Could not find:\n%s\n\nFirst run imageTasks_extract_thresholds_trust2_v1.m, ', ...
           'or edit cfg.infile/cfg.outpath.'], cfg.infile);
end
if ~exist(cfg.outdir,'dir'); mkdir(cfg.outdir); end

S = load(cfg.infile, 'T', 'Generate_params');
T = S.T;
Generate_params = S.Generate_params;

fprintf('\nTrustworthiness dataset 2 empirical threshold analysis\n');
fprintf('Loaded: %s\n', cfg.infile);
fprintf('Value binning: %s | bins: %d | positions: %s\n', cfg.bin_method, cfg.n_bins, mat2str(cfg.positions));
fprintf('Model state set: %s\n\n', cfg.model_state_set);

% -------------------------------------------------------------------------
% BUILD BINNED CHOICE TABLE
% -------------------------------------------------------------------------
ValueBin = assign_value_bins_to_rows(T, Generate_params, cfg);
T.ValueBin = ValueBin;

% Add model P(stop) from Q values and beta.
T.ModelPStop = compute_model_stop_probabilities(T, Generate_params, cfg);

% Keep only CS/BP winners.
BestFit = string(T.BestFitModel);
Model   = string(T.Model);
groups  = string(cfg.groups);

% Participant rows: use winner-model rows only to avoid duplicate rows from
% the two model entries in T. The observed choice itself is model-independent.
participant_rows = ismember(BestFit, groups) ...
    & Model == BestFit ...
    & ismember(T.Position, cfg.positions) ...
    & T.Reached == true ...
    & ismember(T.ObservedActionCode, [1 2]) ...
    & isfinite(T.ValueBin);

% Model rows: use each participant's winning model only.
model_rows = ismember(BestFit, groups) ...
    & Model == BestFit ...
    & ismember(T.Position, cfg.positions) ...
    & isfinite(T.ValueBin) ...
    & isfinite(T.ModelPStop);

if strcmpi(cfg.model_state_set, 'participant_reached')
    model_rows = model_rows & T.Reached == true;
elseif strcmpi(cfg.model_state_set, 'all_positions')
    % leave as all positions 1-7
else
    error('Unknown cfg.model_state_set: %s', cfg.model_state_set);
end

fprintf('Retained participant rows: %d\n', sum(participant_rows));
fprintf('Retained model rows:       %d\n\n', sum(model_rows));

% Long trial-level table for the threshold analysis.
D_part = make_threshold_long_table(T, participant_rows, 'Participants', cfg);
D_model = make_threshold_long_table(T, model_rows, 'Winning model', cfg);
D = [D_part; D_model];

% Aggregate choices/probabilities by group × source × position × bin.
Binned = aggregate_bins(D, cfg);

% Fit logistic thresholds.
Thresholds = fit_thresholds_from_binned_data(Binned, cfg);

% Save tables.
writetable(D, [cfg.outdir filesep 'trust2_empirical_threshold_trials_winners_v1.csv']);
writetable(Binned, [cfg.outdir filesep 'trust2_empirical_threshold_binned_winners_v1.csv']);
writetable(Thresholds, [cfg.outdir filesep 'trust2_empirical_threshold_pse_winners_v1.csv']);

fprintf('\nWrote:\n');
fprintf('  %s\n', [cfg.outdir filesep 'trust2_empirical_threshold_trials_winners_v1.csv']);
fprintf('  %s\n', [cfg.outdir filesep 'trust2_empirical_threshold_binned_winners_v1.csv']);
fprintf('  %s\n', [cfg.outdir filesep 'trust2_empirical_threshold_pse_winners_v1.csv']);

% -------------------------------------------------------------------------
% PRINT SUMMARY
% -------------------------------------------------------------------------
print_threshold_summary(Thresholds, Binned, cfg);

% -------------------------------------------------------------------------
% PLOTS
% -------------------------------------------------------------------------
make_threshold_plot(Thresholds, cfg);
if cfg.make_psychometric_figure
    make_psychometric_plot(Binned, Thresholds, cfg);
end

save([cfg.outdir filesep 'trust2_empirical_threshold_analysis_winners_v1.mat'], ...
    'cfg','D','Binned','Thresholds');

fprintf('\nDone. Outputs written to:\n  %s\n', cfg.outdir);

end % main function

% =========================================================================
% VALUE BINNING
% =========================================================================
function ValueBin = assign_value_bins_to_rows(T, Generate_params, cfg)

n_rows = height(T);
ValueBin = NaN(n_rows,1);

seq_vals = Generate_params.seq_vals;
ratings = Generate_params.ratings;

% Global edges if requested.
if strcmpi(cfg.bin_method, 'global_equal_width')
    all_vals = seq_vals(:);
    all_vals = all_vals(isfinite(all_vals));
    global_edges = linspace(min(all_vals), max(all_vals), cfg.n_bins+1);
    global_edges(1) = -Inf;
    global_edges(end) = Inf;
else
    global_edges = [];
end

subjects = unique(T.Subject)';

for si = 1:numel(subjects)
    sub = subjects(si);

    if strcmpi(cfg.bin_method, 'participant_quantile')
        vals_for_edges = ratings(:,sub);
        vals_for_edges = vals_for_edges(isfinite(vals_for_edges));
        edges = quantile_edges(vals_for_edges, cfg.n_bins);
    elseif strcmpi(cfg.bin_method, 'participant_equal_width')
        vals_for_edges = ratings(:,sub);
        vals_for_edges = vals_for_edges(isfinite(vals_for_edges));
        edges = equal_width_edges(vals_for_edges, cfg.n_bins);
    elseif strcmpi(cfg.bin_method, 'global_equal_width')
        edges = global_edges;
    else
        error('Unknown cfg.bin_method: %s', cfg.bin_method);
    end

    rows = find(T.Subject == sub);
    for ri = 1:numel(rows)
        r = rows(ri);
        v = seq_vals(T.Sequence(r), T.Position(r), sub);
        ValueBin(r) = value_to_bin(v, edges, cfg.n_bins);
    end
end

end

function edges = quantile_edges(vals, n_bins)

vals = vals(isfinite(vals));
if isempty(vals)
    edges = linspace(0,1,n_bins+1);
    edges(1) = -Inf; edges(end) = Inf;
    return;
end

q = linspace(0,1,n_bins+1);
edges = quantile(vals, q);

% Quantile edges can duplicate if ratings are discrete or bunched.
% If too few unique edges, fall back to equal-width edges.
if numel(unique(edges)) < n_bins + 1
    edges = equal_width_edges(vals, n_bins);
else
    edges(1) = -Inf;
    edges(end) = Inf;
end

end

function edges = equal_width_edges(vals, n_bins)

vals = vals(isfinite(vals));
if isempty(vals)
    lo = 0; hi = 1;
else
    lo = min(vals); hi = max(vals);
    if lo == hi
        lo = lo - 0.5; hi = hi + 0.5;
    end
end
edges = linspace(lo, hi, n_bins+1);
edges(1) = -Inf;
edges(end) = Inf;

end

function b = value_to_bin(v, edges, n_bins)

b = NaN;
if ~isfinite(v); return; end

% Find the first upper edge >= v, respecting lower-open/upper-closed bins.
for bi = 1:n_bins
    if v > edges(bi) && v <= edges(bi+1)
        b = bi;
        return;
    end
end

% Fallback for pathological edge cases.
if v <= edges(1)
    b = 1;
elseif v > edges(end)
    b = n_bins;
end

end

% =========================================================================
% MODEL STOP PROBABILITIES
% =========================================================================
function PStop = compute_model_stop_probabilities(T, Generate_params, cfg)

PStop = NaN(height(T),1);

for r = 1:height(T)
    model_name = string(T.Model{r});
    if model_name == "Cost-to-sample"
        model_index = cfg.model_index.Cost_to_sample;
    elseif model_name == "Biased-prior"
        model_index = cfg.model_index.Biased_prior;
    else
        continue;
    end

    sub = T.Subject(r);
    beta = Generate_params.model(model_index).estimated_params(sub,end);

    q_stop = T.QStop(r);
    q_cont = T.QContinue(r);

    if ~isfinite(beta) || ~isfinite(q_stop) || ~isfinite(q_cont)
        continue;
    end

    % Stable two-action softmax probability of stopping.
    x = beta * (q_stop - q_cont);
    if x >= 0
        PStop(r) = 1 ./ (1 + exp(-x));
    else
        ex = exp(x);
        PStop(r) = ex ./ (1 + ex);
    end
end

end

% =========================================================================
% TRIAL TABLE
% =========================================================================
function D = make_threshold_long_table(T, rows, source_label, cfg)

idx = find(rows);
n = numel(idx);

Subject = T.Subject(idx);
Sequence = T.Sequence(idx);
Position = T.Position(idx);
BestFitModel = T.BestFitModel(idx);
Source = repmat({source_label}, n, 1);
ValueBin = T.ValueBin(idx);

if strcmpi(source_label, 'Participants')
    StopResponse = double(T.ObservedActionCode(idx) == 2);
else
    StopResponse = T.ModelPStop(idx);
end

D = table(Subject, Sequence, Position, BestFitModel, Source, ValueBin, StopResponse);

end

% =========================================================================
% BIN AGGREGATION
% =========================================================================
function B = aggregate_bins(D, cfg)

groups = string(cfg.groups);
sources = ["Participants","Winning model"];

BestFitModel = {};
Source = {};
Position = [];
ValueBin = [];
Success = [];
Trials = [];
PropStop = [];

for gi = 1:numel(groups)
    for si = 1:numel(sources)
        for pi = 1:numel(cfg.positions)
            pos = cfg.positions(pi);
            for bi = 1:cfg.n_bins
                f = string(D.BestFitModel)==groups(gi) ...
                    & string(D.Source)==sources(si) ...
                    & D.Position==pos ...
                    & D.ValueBin==bi ...
                    & isfinite(D.StopResponse);

                vals = D.StopResponse(f);
                if isempty(vals)
                    succ = NaN; tr = 0; prop = NaN;
                else
                    succ = sum(vals);
                    tr = numel(vals);
                    prop = succ ./ tr;
                end

                BestFitModel{end+1,1} = char(groups(gi)); %#ok<AGROW>
                Source{end+1,1} = char(sources(si)); %#ok<AGROW>
                Position(end+1,1) = pos; %#ok<AGROW>
                ValueBin(end+1,1) = bi; %#ok<AGROW>
                Success(end+1,1) = succ; %#ok<AGROW>
                Trials(end+1,1) = tr; %#ok<AGROW>
                PropStop(end+1,1) = prop; %#ok<AGROW>
            end
        end
    end
end

B = table(BestFitModel, Source, Position, ValueBin, Success, Trials, PropStop);

end

% =========================================================================
% FIT LOGISTIC THRESHOLDS
% =========================================================================
function Thresholds = fit_thresholds_from_binned_data(Binned, cfg)

groups = string(cfg.groups);
sources = ["Participants","Winning model"];

BestFitModel = {};
Source = {};
Position = [];
ThresholdPSE = [];
Slope = [];
Intercept = [];
NTrials = [];
NSuccess = [];
NFailure = [];
FitOK = [];
NegLogLik = [];

for gi = 1:numel(groups)
    for si = 1:numel(sources)
        for pi = 1:numel(cfg.positions)
            pos = cfg.positions(pi);

            f = string(Binned.BestFitModel)==groups(gi) ...
                & string(Binned.Source)==sources(si) ...
                & Binned.Position==pos ...
                & Binned.Trials >= cfg.min_trials_per_bin ...
                & isfinite(Binned.PropStop);

            x = Binned.ValueBin(f);
            success = Binned.Success(f);
            trials = Binned.Trials(f);

            [pse, slope, intercept, ok, nll] = fit_increasing_logistic_threshold(x, success, trials, cfg);

            BestFitModel{end+1,1} = char(groups(gi)); %#ok<AGROW>
            Source{end+1,1} = char(sources(si)); %#ok<AGROW>
            Position(end+1,1) = pos; %#ok<AGROW>
            ThresholdPSE(end+1,1) = pse; %#ok<AGROW>
            Slope(end+1,1) = slope; %#ok<AGROW>
            Intercept(end+1,1) = intercept; %#ok<AGROW>
            NTrials(end+1,1) = sum(trials); %#ok<AGROW>
            NSuccess(end+1,1) = sum(success); %#ok<AGROW>
            NFailure(end+1,1) = sum(trials-success); %#ok<AGROW>
            FitOK(end+1,1) = ok; %#ok<AGROW>
            NegLogLik(end+1,1) = nll; %#ok<AGROW>
        end
    end
end

Thresholds = table(BestFitModel, Source, Position, ThresholdPSE, Slope, ...
    Intercept, NTrials, NSuccess, NFailure, FitOK, NegLogLik);

end

function [pse, slope, intercept, ok, nll] = fit_increasing_logistic_threshold(x, success, trials, cfg)

pse = NaN; slope = NaN; intercept = NaN; ok = false; nll = NaN;

x = x(:); success = success(:); trials = trials(:);
valid = isfinite(x) & isfinite(success) & isfinite(trials) & trials > 0;
x = x(valid); success = success(valid); trials = trials(valid);

if numel(unique(x)) < 3 || sum(trials) < cfg.min_total_trials
    return;
end

if sum(success) <= cfg.min_successes || sum(trials-success) <= cfg.min_failures
    return;
end

% Initial threshold near the bin closest to .5, slope positive.
p_obs = success ./ trials;
[~,closest] = min(abs(p_obs - 0.5));
init_pse = x(closest);
init_slope = 1;
init_intercept = -init_slope * init_pse;
theta0 = [init_intercept, log(init_slope)];

opts = optimset('Display','off','MaxIter',5000,'MaxFunEvals',5000);
obj = @(th) logistic_nll(th, x, success, trials);
[theta, fval] = fminsearch(obj, theta0, opts);

intercept = theta(1);
slope = exp(theta(2)); % force positive/increasing psychometric curve
pse = -intercept ./ slope;
nll = fval;
ok = isfinite(pse) && isfinite(slope) && isfinite(nll);

end

function nll = logistic_nll(theta, x, success, trials)

intercept = theta(1);
slope = exp(theta(2));
eta = intercept + slope*x;

p = 1 ./ (1 + exp(-eta));
p = min(max(p, 1e-8), 1-1e-8);

nll = -sum(success .* log(p) + (trials-success) .* log(1-p));

if ~isfinite(nll)
    nll = realmax;
end

end

% =========================================================================
% PRINT SUMMARY
% =========================================================================
function print_threshold_summary(Thresholds, Binned, cfg)

fprintf('\n====================================================================\n');
fprintf('EMPIRICAL / CHOICE-DERIVED THRESHOLD SUMMARY\n');
fprintf('====================================================================\n');
fprintf('Thresholds are logistic inflection points / PSEs in value-bin units.\n');
fprintf('Terminal position 8 excluded.\n\n');

groups = string(cfg.groups);
sources = ["Participants","Winning model"];

for gi = 1:numel(groups)
    fprintf('%s\n', cfg.group_labels{gi});
    fprintf('%s\n', repmat('-',1,numel(cfg.group_labels{gi})));

    for si = 1:numel(sources)
        f = string(Thresholds.BestFitModel)==groups(gi) & string(Thresholds.Source)==sources(si);
        th = Thresholds(f,:);
        fprintf('  %s: %d/%d position-wise threshold fits OK\n', sources(si), sum(th.FitOK), height(th));
        for ri = 1:height(th)
            if th.FitOK(ri)
                fprintf('    pos %d: PSE = %.3f, slope = %.3f, trials = %d\n', ...
                    th.Position(ri), th.ThresholdPSE(ri), th.Slope(ri), th.NTrials(ri));
            else
                fprintf('    pos %d: fit failed/insufficient data, trials = %d\n', ...
                    th.Position(ri), th.NTrials(ri));
            end
        end
    end
    fprintf('\n');
end

% Optional bin-count warning.
sparse = Binned(Binned.Trials > 0 & Binned.Trials < cfg.min_trials_per_bin,:);
if ~isempty(sparse)
    fprintf('Note: %d non-empty bins had fewer than cfg.min_trials_per_bin observations and were excluded from logistic fits.\n', height(sparse));
end
fprintf('====================================================================\n\n');

end

% =========================================================================
% THRESHOLD PLOT
% =========================================================================
function make_threshold_plot(Thresholds, cfg)

groups = string(cfg.groups);
sources = ["Participants","Winning model"];

if cfg.use_greyscale
    cols = [0 0 0; .45 .45 .45];
else
    cols = [0 0 0; 0.8500 0.3250 0.0980];
end
markers = {'o','s'};
styles = {'-','--'};

f = figure('Color','w','Name','Trust2 choice-derived thresholds by winner group');
t = tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');

for gi = 1:numel(groups)
    ax = nexttile(t); hold(ax,'on'); box(ax,'off');

    for si = 1:numel(sources)
        rows = string(Thresholds.BestFitModel)==groups(gi) ...
            & string(Thresholds.Source)==sources(si) ...
            & Thresholds.FitOK==true;

        x = Thresholds.Position(rows);
        y = Thresholds.ThresholdPSE(rows);
        [x,ord] = sort(x); y = y(ord);

        plot(ax, x, y, styles{si}, 'Color', cols(si,:), ...
            'LineWidth', 2, 'Marker', markers{si}, ...
            'MarkerFaceColor', 'w', 'MarkerSize', 6);
    end

    xlabel(ax,'Sequence position');
    ylabel(ax,'Choice threshold (value-bin PSE)');
    title(ax,cfg.group_labels{gi});
    set(ax,'FontName','Arial','FontSize',10,'XTick',cfg.positions);
    ylim(ax,[0.5 cfg.n_bins+0.5]);
    xlim(ax,[min(cfg.positions)-0.25 max(cfg.positions)+0.25]);

    if gi == numel(groups)
        lgd = legend(ax, cellstr(sources), 'Location','best');
        lgd.Box = 'off';
    end
end

if cfg.save_figures
    base = [cfg.outdir filesep 'trust2_empirical_thresholds_winners_v1'];
    saveas(f, [base '.png']);
    print(f, [base '.tif'], '-dtiff', '-r600');
    fprintf('Wrote threshold figure:\n  %s\n  %s\n', [base '.png'], [base '.tif']);
end

end

% =========================================================================
% PSYCHOMETRIC CURVE PLOT
% =========================================================================
function make_psychometric_plot(Binned, Thresholds, cfg)

groups = string(cfg.groups);
sources = ["Participants","Winning model"];

% This is a compact diagnostic figure: columns are winner groups, rows are
% source (participants/model). Lines show sequence positions.
f = figure('Color','w','Name','Trust2 psychometric curves by winner group');
t = tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');

pos_cols = parula(numel(cfg.positions)+1);

for si = 1:numel(sources)
    for gi = 1:numel(groups)
        ax = nexttile(t); hold(ax,'on'); box(ax,'off');

        for pi = 1:numel(cfg.positions)
            pos = cfg.positions(pi);
            rows = string(Binned.BestFitModel)==groups(gi) ...
                & string(Binned.Source)==sources(si) ...
                & Binned.Position==pos ...
                & Binned.Trials > 0;

            x = Binned.ValueBin(rows);
            y = Binned.PropStop(rows);
            [x,ord] = sort(x); y = y(ord);

            plot(ax, x, y, '-o', 'Color', pos_cols(pi,:), ...
                'MarkerFaceColor', pos_cols(pi,:), 'MarkerSize', 4, ...
                'LineWidth', 1.1);
        end

        xlabel(ax,'Value bin');
        ylabel(ax,'Proportion stopping');
        title(ax, sprintf('%s: %s', cfg.group_labels{gi}, sources(si)));
        set(ax,'FontName','Arial','FontSize',9,'XTick',1:cfg.n_bins);
        xlim(ax,[0.5 cfg.n_bins+0.5]);
        ylim(ax,[0 1]);
    end
end

lgd = legend(arrayfun(@(p)sprintf('Position %d',p), cfg.positions, 'UniformOutput', false), ...
    'Location','eastoutside');
lgd.Box = 'off';

if cfg.save_figures
    base = [cfg.outdir filesep 'trust2_empirical_psychometric_curves_winners_v1'];
    saveas(f, [base '.png']);
    print(f, [base '.tif'], '-dtiff', '-r600');
    fprintf('Wrote psychometric figure:\n  %s\n  %s\n', [base '.png'], [base '.tif']);
end

end
