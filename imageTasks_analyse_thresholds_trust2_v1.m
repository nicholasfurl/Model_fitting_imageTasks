function imageTasks_analyse_thresholds_trust2_v1()
% imageTasks_analyse_thresholds_trust2_v1
%
% Loads Trustworthiness 2 fitted-model action values saved by
% imageTasks_extract_thresholds_trust2_v1.m, plots continuation-value
% thresholds and fits linear/quadratic threshold-shape regressions.
%
% The key dependent variable is QContinue, the model-derived value of
% sampling again. This functions as an acceptance threshold in action-value
% space: stopping is favoured when Q_stop(current option) exceeds Q_continue.
%
% The program reports two complementary threshold-shape analyses:
%
%   1. Absolute-position analysis:
%      Uses counterfactual model-derived thresholds at absolute sequence
%      positions 1-7 for every sequence, regardless of whether the human
%      participant actually reached that position. This estimates the fitted
%      model's threshold trajectory across the task.
%
%   2. Stop-aligned observed-trajectory analysis:
%      Uses only positions actually reached by the participant, aligned so
%      that 0 is the observed stopping/choice position. This estimates the
%      threshold trajectory along the realised path up to each choice.
%
% Outputs are written to the same thresholds_trust2_out directory.

clc;
warning('off','all');

% -------------------------------------------------------------------------
% CONFIG
% -------------------------------------------------------------------------
cfg.outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\outputs';
cfg.outdir  = [cfg.outpath filesep 'thresholds_trust2_out'];
cfg.infile  = [cfg.outdir filesep 'trust2_threshold_q_values.mat'];

cfg.model_names = {'Cost-to-sample','Biased-prior'};
cfg.bestfit_groups_to_plot = {'Cost-to-sample','Biased-prior'}; % exclude cut off winners from group comparisons
cfg.exclude_terminal_position = 1; % position 8 has forced choice / no meaningful sample-again option
cfg.make_figures = 1;

if ~isfile(cfg.infile)
    error('Could not find threshold MAT file:\n%s\nFirst run imageTasks_extract_thresholds_trust2_v1.m.', cfg.infile);
end

S = load(cfg.infile, 'T', 'cfg');
T = S.T;

if ~exist(cfg.outdir,'dir'); mkdir(cfg.outdir); end

fprintf('\nTrustworthiness dataset 2 threshold analysis\n');
fprintf('Loaded: %s\n', cfg.infile);

% Make string columns robust across MATLAB versions.
Model = string(T.Model);
BestFitModel = string(T.BestFitModel);

% Define core filters.
is_threshold_model = ismember(Model, string(cfg.model_names));
is_nonterminal = T.Position < 8;
is_group_of_interest = ismember(BestFitModel, string(cfg.bestfit_groups_to_plot));
finite_threshold = isfinite(T.QContinue);

% -------------------------------------------------------------------------
% FIGURES
% -------------------------------------------------------------------------
if cfg.make_figures
    make_absolute_position_fig(T, cfg, is_threshold_model & is_nonterminal & finite_threshold);
    make_absolute_position_by_group_fig(T, cfg, is_threshold_model & is_nonterminal & finite_threshold & is_group_of_interest);
    make_stop_aligned_fig(T, cfg, is_threshold_model & is_nonterminal & finite_threshold & is_group_of_interest & T.Reached);
end

% -------------------------------------------------------------------------
% REGRESSION ANALYSES
% -------------------------------------------------------------------------
% A. Absolute-position participant-level fits.
% Aggregate over sequences first, then fit threshold ~ position per participant/model.
AbsPart = participant_position_means(T, ...
    is_threshold_model & is_nonterminal & finite_threshold, ...
    'absolute');
AbsFits = fit_participant_curves(AbsPart, 'absolute');
write_table(AbsFits, cfg.outdir, 'trust2_threshold_absolute_participant_fits.csv');

% B. Stop-aligned participant-level fits.
% Use only observed/reached positions, align to observed stopping position (0 = choice).
StopPart = participant_position_means(T, ...
    is_threshold_model & is_nonterminal & finite_threshold & T.Reached, ...
    'stop_aligned');
StopFits = fit_participant_curves(StopPart, 'stop_aligned');
write_table(StopFits, cfg.outdir, 'trust2_threshold_stop_aligned_participant_fits.csv');

% C. Sequence-level stop-aligned fits, then summarise per participant.
SeqFits = fit_sequence_curves(T, is_threshold_model & is_nonterminal & finite_threshold & T.Reached);
write_table(SeqFits, cfg.outdir, 'trust2_threshold_stop_aligned_sequence_fits.csv');
SeqPartSummary = summarise_sequence_fits_by_participant(SeqFits);
write_table(SeqPartSummary, cfg.outdir, 'trust2_threshold_stop_aligned_sequence_fit_participant_summary.csv');

% -------------------------------------------------------------------------
% SECOND-LEVEL SUMMARIES / TESTS
% -------------------------------------------------------------------------
AbsSummary  = summarise_second_level(AbsFits, 'absolute');
StopSummary = summarise_second_level(StopFits, 'stop_aligned_participant');
SeqSummary  = summarise_second_level(SeqPartSummary, 'stop_aligned_sequence_mean');

write_table(AbsSummary, cfg.outdir, 'trust2_threshold_absolute_second_level_summary.csv');
write_table(StopSummary, cfg.outdir, 'trust2_threshold_stop_aligned_second_level_summary.csv');
write_table(SeqSummary, cfg.outdir, 'trust2_threshold_sequence_mean_second_level_summary.csv');

% Save everything in a MAT file too.
outmat = [cfg.outdir filesep 'trust2_threshold_analysis_results.mat'];
save(outmat, 'cfg', 'AbsPart', 'AbsFits', 'StopPart', 'StopFits', ...
    'SeqFits', 'SeqPartSummary', 'AbsSummary', 'StopSummary', 'SeqSummary');
fprintf('\nWrote MAT analysis output:\n  %s\n', outmat);

fprintf('\nDone. Outputs written to:\n  %s\n', cfg.outdir);

end % main function

% =========================================================================
% FIGURE HELPERS
% =========================================================================
function make_absolute_position_fig(T, cfg, filt)

Tf = T(filt,:);
models = string(cfg.model_names);
positions = unique(Tf.Position)';

f = figure('Color','w','Name','Thresholds by absolute position'); hold on; box off;
markers = {'-o','--s'};
for mi = 1:numel(models)
    y = NaN(size(positions));
    se = NaN(size(positions));
    for pi = 1:numel(positions)
        vals = Tf.QContinue(string(Tf.Model)==models(mi) & Tf.Position==positions(pi));
        y(pi) = mean(vals,'omitnan');
        se(pi) = std(vals,'omitnan') ./ sqrt(sum(isfinite(vals)));
    end
    plot(positions, y, markers{mi}, 'LineWidth', 1.5, 'MarkerSize', 5);
end
xlabel('Sequence position');
ylabel('Mean continuation-value threshold');
legend(cfg.model_names, 'Location','best'); legend boxoff;
set(gca,'FontName','Arial','FontSize',10,'XTick',positions);

saveas(f, [cfg.outdir filesep 'trust2_thresholds_absolute_by_model.png']);
print(f, [cfg.outdir filesep 'trust2_thresholds_absolute_by_model.tif'], '-dtiff', '-r600');
end

function make_absolute_position_by_group_fig(T, cfg, filt)

Tf = T(filt,:);
models = string(cfg.model_names);
groups = string(cfg.bestfit_groups_to_plot);
positions = unique(Tf.Position)';
markers = {'-o','--s'};

f = figure('Color','w','Name','Thresholds by absolute position and best-fit group');
t = tiledlayout(f,1,numel(groups),'TileSpacing','compact','Padding','compact');
for gi = 1:numel(groups)
    ax = nexttile(t); hold(ax,'on'); box(ax,'off');
    for mi = 1:numel(models)
        y = NaN(size(positions));
        for pi = 1:numel(positions)
            vals = Tf.QContinue(string(Tf.BestFitModel)==groups(gi) & ...
                string(Tf.Model)==models(mi) & Tf.Position==positions(pi));
            y(pi) = mean(vals,'omitnan');
        end
        plot(ax, positions, y, markers{mi}, 'LineWidth', 1.5, 'MarkerSize', 5);
    end
    xlabel(ax,'Sequence position');
    ylabel(ax,'Mean continuation-value threshold');
    title(ax, sprintf('Participants best fitted by\n%s model', groups(gi)));
    set(ax,'FontName','Arial','FontSize',10,'XTick',positions);
end
lgd = legend(ax, cfg.model_names, 'Location','best');
lgd.Box = 'off';

saveas(f, [cfg.outdir filesep 'trust2_thresholds_absolute_by_bestfit_group.png']);
print(f, [cfg.outdir filesep 'trust2_thresholds_absolute_by_bestfit_group.tif'], '-dtiff', '-r600');
end

function make_stop_aligned_fig(T, cfg, filt)

Tf = T(filt,:);
models = string(cfg.model_names);
groups = string(cfg.bestfit_groups_to_plot);
rel_positions = unique(Tf.PositionFromChoice)';
rel_positions = rel_positions(rel_positions <= 0); % up to and including observed choice
markers = {'-o','--s'};

f = figure('Color','w','Name','Stop-aligned thresholds by best-fit group');
t = tiledlayout(f,1,numel(groups),'TileSpacing','compact','Padding','compact');
for gi = 1:numel(groups)
    ax = nexttile(t); hold(ax,'on'); box(ax,'off');
    for mi = 1:numel(models)
        y = NaN(size(rel_positions));
        for ri = 1:numel(rel_positions)
            vals = Tf.QContinue(string(Tf.BestFitModel)==groups(gi) & ...
                string(Tf.Model)==models(mi) & Tf.PositionFromChoice==rel_positions(ri));
            y(ri) = mean(vals,'omitnan');
        end
        plot(ax, rel_positions, y, markers{mi}, 'LineWidth', 1.5, 'MarkerSize', 5);
    end
    xlabel(ax,'Sequence position relative to choice');
    ylabel(ax,'Mean continuation-value threshold');
    title(ax, sprintf('Participants best fitted by\n%s model', groups(gi)));
    set(ax,'FontName','Arial','FontSize',10,'XTick',rel_positions);
    xline(ax,0,'k:');
end
lgd = legend(ax, cfg.model_names, 'Location','best');
lgd.Box = 'off';

saveas(f, [cfg.outdir filesep 'trust2_thresholds_stop_aligned_by_bestfit_group.png']);
print(f, [cfg.outdir filesep 'trust2_thresholds_stop_aligned_by_bestfit_group.tif'], '-dtiff', '-r600');
end

% =========================================================================
% DATA AGGREGATION
% =========================================================================
function Part = participant_position_means(T, filt, mode)

Tf = T(filt,:);
Model = string(Tf.Model);
BestFitModel = string(Tf.BestFitModel);
subjects = unique(Tf.Subject)';
models = unique(Model)';

Subject = [];
ModelOut = {};
BestFitModelOut = {};
X = [];
MeanThreshold = [];
NObs = [];

for si = 1:numel(subjects)
    sub = subjects(si);
    for mi = 1:numel(models)
        this = Tf(Tf.Subject==sub & Model==models(mi),:);
        if isempty(this); continue; end
        bfm = string(this.BestFitModel{1});

        if strcmp(mode,'absolute')
            xvals = unique(this.Position)';
            xsource = this.Position;
        elseif strcmp(mode,'stop_aligned')
            xvals = unique(this.PositionFromChoice)';
            xvals = xvals(xvals <= 0);
            xsource = this.PositionFromChoice;
        else
            error('Unknown mode: %s', mode);
        end

        for xi = 1:numel(xvals)
            vals = this.QContinue(xsource == xvals(xi));
            vals = vals(isfinite(vals));
            if isempty(vals); continue; end
            Subject(end+1,1) = sub; %#ok<AGROW>
            ModelOut{end+1,1} = char(models(mi)); %#ok<AGROW>
            BestFitModelOut{end+1,1} = char(bfm); %#ok<AGROW>
            X(end+1,1) = xvals(xi); %#ok<AGROW>
            MeanThreshold(end+1,1) = mean(vals); %#ok<AGROW>
            NObs(end+1,1) = numel(vals); %#ok<AGROW>
        end
    end
end

Part = table(Subject, ModelOut, BestFitModelOut, X, MeanThreshold, NObs, ...
    'VariableNames', {'Subject','Model','BestFitModel','X','MeanThreshold','NObs'});
end

% =========================================================================
% FIT PARTICIPANT-LEVEL CURVES
% =========================================================================
function Fits = fit_participant_curves(Part, mode_label)

subjects = unique(Part.Subject)';
models = unique(string(Part.Model))';

Subject = [];
Model = {};
BestFitModel = {};
Mode = {};
NPoints = [];
LinearIntercept = [];
LinearSlope = [];
LinearR2 = [];
LinearSSE = [];
LinearAIC = [];
LinearBIC = [];
QuadIntercept = [];
QuadSlope = [];
QuadTerm = [];
QuadR2 = [];
QuadSSE = [];
QuadAIC = [];
QuadBIC = [];
DeltaBIC_QuadMinusLinear = [];

for si = 1:numel(subjects)
    sub = subjects(si);
    for mi = 1:numel(models)
        this = Part(Part.Subject==sub & string(Part.Model)==models(mi),:);
        if isempty(this); continue; end
        x = this.X;
        y = this.MeanThreshold;
        valid = isfinite(x) & isfinite(y);
        x = x(valid); y = y(valid);
        [x, order] = sort(x); y = y(order);
        n = numel(unique(x));

        lin = fit_poly_stats(x, y, 1);
        quad = fit_poly_stats(x, y, 2);

        Subject(end+1,1) = sub; %#ok<AGROW>
        Model{end+1,1} = char(models(mi)); %#ok<AGROW>
        BestFitModel{end+1,1} = this.BestFitModel{1}; %#ok<AGROW>
        Mode{end+1,1} = mode_label; %#ok<AGROW>
        NPoints(end+1,1) = n; %#ok<AGROW>

        LinearIntercept(end+1,1) = lin.intercept; %#ok<AGROW>
        LinearSlope(end+1,1) = lin.slope; %#ok<AGROW>
        LinearR2(end+1,1) = lin.R2; %#ok<AGROW>
        LinearSSE(end+1,1) = lin.SSE; %#ok<AGROW>
        LinearAIC(end+1,1) = lin.AIC; %#ok<AGROW>
        LinearBIC(end+1,1) = lin.BIC; %#ok<AGROW>

        QuadIntercept(end+1,1) = quad.intercept; %#ok<AGROW>
        QuadSlope(end+1,1) = quad.slope; %#ok<AGROW>
        QuadTerm(end+1,1) = quad.quad; %#ok<AGROW>
        QuadR2(end+1,1) = quad.R2; %#ok<AGROW>
        QuadSSE(end+1,1) = quad.SSE; %#ok<AGROW>
        QuadAIC(end+1,1) = quad.AIC; %#ok<AGROW>
        QuadBIC(end+1,1) = quad.BIC; %#ok<AGROW>
        DeltaBIC_QuadMinusLinear(end+1,1) = quad.BIC - lin.BIC; %#ok<AGROW>
    end
end

Fits = table(Subject, Model, BestFitModel, Mode, NPoints, ...
    LinearIntercept, LinearSlope, LinearR2, LinearSSE, LinearAIC, LinearBIC, ...
    QuadIntercept, QuadSlope, QuadTerm, QuadR2, QuadSSE, QuadAIC, QuadBIC, ...
    DeltaBIC_QuadMinusLinear);
end

% =========================================================================
% FIT STOP-ALIGNED CURVES PER SEQUENCE
% =========================================================================
function SeqFits = fit_sequence_curves(T, filt)

Tf = T(filt,:);
subjects = unique(Tf.Subject)';
models = unique(string(Tf.Model))';

Subject = [];
Sequence = [];
Model = {};
BestFitModel = {};
NPoints = [];
LinearSlope = [];
LinearR2 = [];
LinearBIC = [];
QuadTerm = [];
QuadR2 = [];
QuadBIC = [];
DeltaBIC_QuadMinusLinear = [];

for si = 1:numel(subjects)
    sub = subjects(si);
    seqs = unique(Tf.Sequence(Tf.Subject==sub))';
    for qi = 1:numel(seqs)
        seq = seqs(qi);
        for mi = 1:numel(models)
            this = Tf(Tf.Subject==sub & Tf.Sequence==seq & string(Tf.Model)==models(mi),:);
            if isempty(this); continue; end
            x = this.PositionFromChoice; % 0 at observed choice, negative before choice
            y = this.QContinue;
            valid = isfinite(x) & isfinite(y) & x <= 0;
            x = x(valid); y = y(valid);
            [x, order] = sort(x); y = y(order);
            n = numel(unique(x));

            lin = fit_poly_stats(x, y, 1);
            quad = fit_poly_stats(x, y, 2);

            Subject(end+1,1) = sub; %#ok<AGROW>
            Sequence(end+1,1) = seq; %#ok<AGROW>
            Model{end+1,1} = char(models(mi)); %#ok<AGROW>
            BestFitModel{end+1,1} = this.BestFitModel{1}; %#ok<AGROW>
            NPoints(end+1,1) = n; %#ok<AGROW>
            LinearSlope(end+1,1) = lin.slope; %#ok<AGROW>
            LinearR2(end+1,1) = lin.R2; %#ok<AGROW>
            LinearBIC(end+1,1) = lin.BIC; %#ok<AGROW>
            QuadTerm(end+1,1) = quad.quad; %#ok<AGROW>
            QuadR2(end+1,1) = quad.R2; %#ok<AGROW>
            QuadBIC(end+1,1) = quad.BIC; %#ok<AGROW>
            DeltaBIC_QuadMinusLinear(end+1,1) = quad.BIC - lin.BIC; %#ok<AGROW>
        end
    end
end

SeqFits = table(Subject, Sequence, Model, BestFitModel, NPoints, LinearSlope, ...
    LinearR2, LinearBIC, QuadTerm, QuadR2, QuadBIC, DeltaBIC_QuadMinusLinear);
end

function SeqPartSummary = summarise_sequence_fits_by_participant(SeqFits)

subjects = unique(SeqFits.Subject)';
models = unique(string(SeqFits.Model))';

Subject = [];
Model = {};
BestFitModel = {};
MeanLinearSlope = [];
MeanLinearR2 = [];
MeanQuadTerm = [];
MeanQuadR2 = [];
MeanDeltaBIC_QuadMinusLinear = [];
NSequencesLinear = [];
NSequencesQuadratic = [];

for si = 1:numel(subjects)
    sub = subjects(si);
    for mi = 1:numel(models)
        this = SeqFits(SeqFits.Subject==sub & string(SeqFits.Model)==models(mi),:);
        if isempty(this); continue; end
        Subject(end+1,1) = sub; %#ok<AGROW>
        Model{end+1,1} = char(models(mi)); %#ok<AGROW>
        BestFitModel{end+1,1} = this.BestFitModel{1}; %#ok<AGROW>
        MeanLinearSlope(end+1,1) = mean(this.LinearSlope,'omitnan'); %#ok<AGROW>
        MeanLinearR2(end+1,1) = mean(this.LinearR2,'omitnan'); %#ok<AGROW>
        MeanQuadTerm(end+1,1) = mean(this.QuadTerm,'omitnan'); %#ok<AGROW>
        MeanQuadR2(end+1,1) = mean(this.QuadR2,'omitnan'); %#ok<AGROW>
        MeanDeltaBIC_QuadMinusLinear(end+1,1) = mean(this.DeltaBIC_QuadMinusLinear,'omitnan'); %#ok<AGROW>
        NSequencesLinear(end+1,1) = sum(isfinite(this.LinearSlope)); %#ok<AGROW>
        NSequencesQuadratic(end+1,1) = sum(isfinite(this.QuadTerm)); %#ok<AGROW>
    end
end

SeqPartSummary = table(Subject, Model, BestFitModel, MeanLinearSlope, MeanLinearR2, ...
    MeanQuadTerm, MeanQuadR2, MeanDeltaBIC_QuadMinusLinear, ...
    NSequencesLinear, NSequencesQuadratic);
end

% =========================================================================
% SECOND-LEVEL SUMMARY
% =========================================================================
function Summary = summarise_second_level(Fits, analysis_label)

Model = string(Fits.Model);
models = unique(Model)';
BestFitModel = string(Fits.BestFitModel);

Analysis = {};
Subset = {};
ModelOut = {};
Measure = {};
N = [];
MeanValue = [];
SDValue = [];
Tstat = [];
DF = [];
P = [];

for mi = 1:numel(models)
    this_model = models(mi);

    % Select measure names depending on table type.
    if ismember('LinearSlope', Fits.Properties.VariableNames)
        slope_vals = Fits.LinearSlope(Model==this_model);
        delta_bic_vals = Fits.DeltaBIC_QuadMinusLinear(Model==this_model);
        quad_vals = Fits.QuadTerm(Model==this_model);
    else
        slope_vals = Fits.MeanLinearSlope(Model==this_model);
        delta_bic_vals = Fits.MeanDeltaBIC_QuadMinusLinear(Model==this_model);
        quad_vals = Fits.MeanQuadTerm(Model==this_model);
    end

    [Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P] = ...
        add_one_ttest_row(Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, ...
        analysis_label, 'All participants', char(this_model), 'Linear slope vs zero', slope_vals);

    [Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P] = ...
        add_one_ttest_row(Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, ...
        analysis_label, 'All participants', char(this_model), 'Quadratic term vs zero', quad_vals);

    [Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P] = ...
        add_one_ttest_row(Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, ...
        analysis_label, 'All participants', char(this_model), 'Delta BIC quad-minus-linear vs zero', delta_bic_vals);
end

% Group differences: cost-to-sample best-fitted participants versus biased-prior best-fitted participants.
groups = {'Cost-to-sample','Biased-prior'};
for mi = 1:numel(models)
    this_model = models(mi);
    if ismember('LinearSlope', Fits.Properties.VariableNames)
        vals = Fits.LinearSlope(Model==this_model);
        bfg = BestFitModel(Model==this_model);
    else
        vals = Fits.MeanLinearSlope(Model==this_model);
        bfg = BestFitModel(Model==this_model);
    end
    v1 = vals(bfg == groups{1});
    v2 = vals(bfg == groups{2});
    [h,p,ci,stats] = ttest2(v1(isfinite(v1)), v2(isfinite(v2))); %#ok<ASGLU>
    Analysis{end+1,1} = analysis_label; %#ok<AGROW>
    Subset{end+1,1} = 'Cost-to-sample vs biased-prior best-fit groups'; %#ok<AGROW>
    ModelOut{end+1,1} = char(this_model); %#ok<AGROW>
    Measure{end+1,1} = 'Linear slope group difference'; %#ok<AGROW>
    N(end+1,1) = sum(isfinite(v1)) + sum(isfinite(v2)); %#ok<AGROW>
    MeanValue(end+1,1) = mean(v1,'omitnan') - mean(v2,'omitnan'); %#ok<AGROW>
    SDValue(end+1,1) = NaN; %#ok<AGROW>
    Tstat(end+1,1) = stats.tstat; %#ok<AGROW>
    DF(end+1,1) = stats.df; %#ok<AGROW>
    P(end+1,1) = p; %#ok<AGROW>
end

Summary = table(Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P);
end

function [Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P] = ...
    add_one_ttest_row(Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, ...
    analysis_label, subset_label, model_label, measure_label, vals)

vals = vals(isfinite(vals));
if numel(vals) >= 2
    [h,p,ci,stats] = ttest(vals); %#ok<ASGLU>
    tstat = stats.tstat;
    df = stats.df;
else
    p = NaN; tstat = NaN; df = NaN;
end

Analysis{end+1,1} = analysis_label; %#ok<AGROW>
Subset{end+1,1} = subset_label; %#ok<AGROW>
ModelOut{end+1,1} = model_label; %#ok<AGROW>
Measure{end+1,1} = measure_label; %#ok<AGROW>
N(end+1,1) = numel(vals); %#ok<AGROW>
MeanValue(end+1,1) = mean(vals,'omitnan'); %#ok<AGROW>
SDValue(end+1,1) = std(vals,'omitnan'); %#ok<AGROW>
Tstat(end+1,1) = tstat; %#ok<AGROW>
DF(end+1,1) = df; %#ok<AGROW>
P(end+1,1) = p; %#ok<AGROW>
end

% =========================================================================
% POLYNOMIAL FIT UTILITY
% =========================================================================
function out = fit_poly_stats(x, y, degree)

out.intercept = NaN;
out.slope = NaN;
out.quad = NaN;
out.R2 = NaN;
out.SSE = NaN;
out.AIC = NaN;
out.BIC = NaN;

valid = isfinite(x) & isfinite(y);
x = x(valid);
y = y(valid);

if numel(unique(x)) < degree + 1 || numel(y) < degree + 1
    return;
end

p = polyfit(x, y, degree);
yhat = polyval(p, x);
resid = y - yhat;
SSE = sum(resid.^2);
SST = sum((y - mean(y)).^2);
if SST > 0
    R2 = 1 - SSE/SST;
else
    R2 = NaN;
end

n = numel(y);
k = degree + 1; % intercept plus coefficients
SSE_for_ic = max(SSE, eps);
AIC = n*log(SSE_for_ic/n) + 2*k;
BIC = n*log(SSE_for_ic/n) + k*log(n);

if degree == 1
    out.slope = p(1);
    out.intercept = p(2);
elseif degree == 2
    out.quad = p(1);
    out.slope = p(2);
    out.intercept = p(3);
end
out.R2 = R2;
out.SSE = SSE;
out.AIC = AIC;
out.BIC = BIC;
end

function write_table(T, outdir, filename)
file = [outdir filesep filename];
writetable(T, file);
fprintf('Wrote table:\n  %s\n', file);
end
