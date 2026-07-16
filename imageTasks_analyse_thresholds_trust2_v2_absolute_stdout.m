function imageTasks_analyse_thresholds_trust2_v2_absolute_stdout()
% imageTasks_analyse_thresholds_trust2_v2_absolute_stdout
%
% Loads Trustworthiness 2 fitted-model action values saved by
% imageTasks_extract_thresholds_trust2_v1.m, focuses on absolute-position
% continuation-value threshold trajectories, and prints participant-level
% second-level summaries separately for:
%
%   - all participants
%   - participants best fitted by the cost-to-sample model
%   - participants best fitted by the biased-prior model
%
% The dependent variable is QContinue, the fitted model's value of sampling
% again. In this analysis it is treated as a continuation-value threshold in
% action-value space: stopping is favoured when QStop exceeds QContinue.
%
% The program does not reconstruct Q values itself. Run
% imageTasks_extract_thresholds_trust2_v1.m first.

clc;
warning('off','all');

% -------------------------------------------------------------------------
% CONFIG
% -------------------------------------------------------------------------
cfg.outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\outputs';
cfg.outdir  = [cfg.outpath filesep 'thresholds_trust2_out'];
cfg.infile  = [cfg.outdir filesep 'trust2_threshold_q_values.mat'];

cfg.model_names = {'Cost-to-sample','Biased-prior'};
cfg.bestfit_groups_to_report = {'Cost-to-sample','Biased-prior'};
cfg.exclude_terminal_position = 1; % position 8 has forced choice / no meaningful sample-again option
cfg.make_figures = 1;

if ~isfile(cfg.infile)
    error('Could not find threshold MAT file:\n%s\nFirst run imageTasks_extract_thresholds_trust2_v1.m.', cfg.infile);
end

S = load(cfg.infile, 'T', 'cfg');
T = S.T;

if ~exist(cfg.outdir,'dir'); mkdir(cfg.outdir); end

fprintf('\nTrustworthiness dataset 2 absolute-position threshold analysis\n');
fprintf('Loaded: %s\n', cfg.infile);
fprintf('Outcome: QContinue, the continuation-value threshold in action-value space.\n');
fprintf('Positions analysed: 1-7. Terminal position 8 is excluded.\n\n');

% Make string columns robust across MATLAB versions.
Model = string(T.Model);
BestFitModel = string(T.BestFitModel);

% Filters.
is_threshold_model = ismember(Model, string(cfg.model_names));
is_nonterminal = T.Position < 8;
finite_threshold = isfinite(T.QContinue);
filt_absolute = is_threshold_model & is_nonterminal & finite_threshold;

% Print best-fitting group counts from the full-sequence fits.
fprintf('Best-fitting groups from full-sequence NLLs:\n');
all_groups = unique(BestFitModel)';
for gi = 1:numel(all_groups)
    fprintf('  %s: %d participants\n', all_groups(gi), numel(unique(T.Subject(BestFitModel == all_groups(gi)))));
end
fprintf('\n');

% -------------------------------------------------------------------------
% FIGURES
% -------------------------------------------------------------------------
if cfg.make_figures
    make_absolute_position_fig(T, cfg, filt_absolute);
    make_absolute_position_by_group_fig(T, cfg, filt_absolute & ismember(BestFitModel, string(cfg.bestfit_groups_to_report)));
    make_standardised_slope_scatter(T, cfg, filt_absolute & ismember(BestFitModel, string(cfg.bestfit_groups_to_report)));
end

% -------------------------------------------------------------------------
% PARTICIPANT-LEVEL ABSOLUTE-POSITION FITS
% -------------------------------------------------------------------------
AbsPart = participant_position_means(T, filt_absolute);
AbsFits = fit_participant_absolute_curves(AbsPart);

write_table(AbsPart, cfg.outdir, 'trust2_threshold_absolute_participant_position_means.csv');
write_table(AbsFits, cfg.outdir, 'trust2_threshold_absolute_participant_fits_v2.csv');

% -------------------------------------------------------------------------
% SECOND-LEVEL REPORTING
% -------------------------------------------------------------------------
Summary = summarise_absolute_second_level(AbsFits, cfg);
write_table(Summary, cfg.outdir, 'trust2_threshold_absolute_second_level_summary_v2.csv');

print_absolute_second_level(Summary, cfg);

% Save everything in a MAT file too.
outmat = [cfg.outdir filesep 'trust2_threshold_absolute_analysis_v2.mat'];
save(outmat, 'cfg', 'AbsPart', 'AbsFits', 'Summary');
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

f = figure('Color','w','Name','Absolute-position continuation thresholds'); hold on; box off;
markers = {'-o','--s'};
for mi = 1:numel(models)
    y = NaN(size(positions));
    for pi = 1:numel(positions)
        vals = Tf.QContinue(string(Tf.Model)==models(mi) & Tf.Position==positions(pi));
        y(pi) = mean(vals,'omitnan');
    end
    plot(positions, y, markers{mi}, 'LineWidth', 1.5, 'MarkerSize', 5);
end
xlabel('Sequence position');
ylabel('Mean continuation-value threshold');
legend(cfg.model_names, 'Location','best'); legend boxoff;
set(gca,'FontName','Arial','FontSize',10,'XTick',positions);

saveas(f, [cfg.outdir filesep 'trust2_thresholds_absolute_by_model_v2.png']);
print(f, [cfg.outdir filesep 'trust2_thresholds_absolute_by_model_v2.tif'], '-dtiff', '-r600');
end

function make_absolute_position_by_group_fig(T, cfg, filt)

Tf = T(filt,:);
models = string(cfg.model_names);
groups = string(cfg.bestfit_groups_to_report);
positions = unique(Tf.Position)';
markers = {'-o','--s'};

f = figure('Color','w','Name','Absolute-position thresholds by best-fit group');
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

saveas(f, [cfg.outdir filesep 'trust2_thresholds_absolute_by_bestfit_group_v2.png']);
print(f, [cfg.outdir filesep 'trust2_thresholds_absolute_by_bestfit_group_v2.tif'], '-dtiff', '-r600');
end

function make_standardised_slope_scatter(T, cfg, filt)

% Create this from participant-level fits, not from raw rows.
AbsPart = participant_position_means(T, filt);
AbsFits = fit_participant_absolute_curves(AbsPart);

models = string(cfg.model_names);
groups = string(cfg.bestfit_groups_to_report);

f = figure('Color','w','Name','Standardised linear slopes'); hold on; box off;
xpos = 0;
labels = {};
for mi = 1:numel(models)
    for gi = 1:numel(groups)
        xpos = xpos + 1;
        vals = AbsFits.LinearSlopeStd(string(AbsFits.Model)==models(mi) & ...
            string(AbsFits.BestFitModel)==groups(gi));
        vals = vals(isfinite(vals));
        xj = xpos + (rand(size(vals))-.5)*0.18;
        plot(xj, vals, 'ko', 'MarkerSize', 4, 'MarkerFaceColor', [0.75 0.75 0.75]);
        plot([xpos-.25 xpos+.25], [mean(vals) mean(vals)], 'k-', 'LineWidth', 2);
        labels{end+1} = sprintf('%s\n%s winners', models(mi), groups(gi)); %#ok<AGROW>
    end
end
plot([0 xpos+1],[0 0],'k:');
xlim([0 xpos+1]);
set(gca,'XTick',1:xpos,'XTickLabel',labels,'FontName','Arial','FontSize',9);
xtickangle(30);
ylabel('Standardised linear slope');
saveas(f, [cfg.outdir filesep 'trust2_threshold_standardised_slopes_v2.png']);
print(f, [cfg.outdir filesep 'trust2_threshold_standardised_slopes_v2.tif'], '-dtiff', '-r600');
end

% =========================================================================
% DATA AGGREGATION
% =========================================================================
function Part = participant_position_means(T, filt)

Tf = T(filt,:);
Model = string(Tf.Model);
subjects = unique(Tf.Subject)';
models = unique(Model)';

Subject = [];
ModelOut = {};
BestFitModelOut = {};
Position = [];
MeanThreshold = [];
NObs = [];

for si = 1:numel(subjects)
    sub = subjects(si);
    for mi = 1:numel(models)
        this = Tf(Tf.Subject==sub & Model==models(mi),:);
        if isempty(this); continue; end
        bfm = string(this.BestFitModel{1});
        positions = unique(this.Position)';

        for pi = 1:numel(positions)
            vals = this.QContinue(this.Position == positions(pi));
            vals = vals(isfinite(vals));
            if isempty(vals); continue; end

            Subject(end+1,1) = sub; %#ok<AGROW>
            ModelOut{end+1,1} = char(models(mi)); %#ok<AGROW>
            BestFitModelOut{end+1,1} = char(bfm); %#ok<AGROW>
            Position(end+1,1) = positions(pi); %#ok<AGROW>
            MeanThreshold(end+1,1) = mean(vals); %#ok<AGROW>
            NObs(end+1,1) = numel(vals); %#ok<AGROW>
        end
    end
end

Part = table(Subject, ModelOut, BestFitModelOut, Position, MeanThreshold, NObs, ...
    'VariableNames', {'Subject','Model','BestFitModel','Position','MeanThreshold','NObs'});
end

% =========================================================================
% FIT PARTICIPANT-LEVEL CURVES
% =========================================================================
function Fits = fit_participant_absolute_curves(Part)

subjects = unique(Part.Subject)';
models = unique(string(Part.Model))';

Subject = [];
Model = {};
BestFitModel = {};
NPoints = [];

LinearIntercept = [];
LinearSlope = [];
LinearSlopeStd = [];
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
DeltaR2_QuadMinusLinear = [];

for si = 1:numel(subjects)
    sub = subjects(si);
    for mi = 1:numel(models)
        this = Part(Part.Subject==sub & string(Part.Model)==models(mi),:);
        if isempty(this); continue; end

        x = this.Position;
        y = this.MeanThreshold;
        valid = isfinite(x) & isfinite(y);
        x = x(valid); y = y(valid);
        [x, order] = sort(x); y = y(order);
        n = numel(unique(x));

        lin = fit_poly_stats(x, y, 1);
        quad = fit_poly_stats(x, y, 2);

        Subject(end+1,1) = sub; %#ok<AGROW>
        Model{end+1,1} = this.Model{1}; %#ok<AGROW>
        BestFitModel{end+1,1} = this.BestFitModel{1}; %#ok<AGROW>
        NPoints(end+1,1) = n; %#ok<AGROW>

        LinearIntercept(end+1,1) = lin.intercept; %#ok<AGROW>
        LinearSlope(end+1,1) = lin.slope; %#ok<AGROW>
        LinearSlopeStd(end+1,1) = lin.slope_std; %#ok<AGROW>
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
        DeltaR2_QuadMinusLinear(end+1,1) = quad.R2 - lin.R2; %#ok<AGROW>
    end
end

Fits = table(Subject, Model, BestFitModel, NPoints, ...
    LinearIntercept, LinearSlope, LinearSlopeStd, LinearR2, LinearSSE, LinearAIC, LinearBIC, ...
    QuadIntercept, QuadSlope, QuadTerm, QuadR2, QuadSSE, QuadAIC, QuadBIC, ...
    DeltaBIC_QuadMinusLinear, DeltaR2_QuadMinusLinear);
end

% =========================================================================
% SECOND-LEVEL SUMMARY
% =========================================================================
function Summary = summarise_absolute_second_level(Fits, cfg)

models = string(cfg.model_names);
groups = string(cfg.bestfit_groups_to_report);

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
NLinearBetter = [];
NQuadraticBetter = [];

% Subsets: all participants, CS winners, BP winners.
subset_names = ["All participants", ...
    "Participants best fitted by cost-to-sample", ...
    "Participants best fitted by biased-prior"];
subset_filters = cell(1,3);
subset_filters{1} = true(height(Fits),1);
subset_filters{2} = string(Fits.BestFitModel) == groups(1);
subset_filters{3} = string(Fits.BestFitModel) == groups(2);

for si = 1:numel(subset_names)
    for mi = 1:numel(models)
        rowf = subset_filters{si} & string(Fits.Model)==models(mi);

        vals = Fits.LinearSlope(rowf);
        [Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, NLinearBetter, NQuadraticBetter] = ...
            add_one_ttest_row(Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, NLinearBetter, NQuadraticBetter, ...
            'absolute_position', char(subset_names(si)), char(models(mi)), 'Raw linear slope vs zero', vals, []);

        vals = Fits.LinearSlopeStd(rowf);
        [Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, NLinearBetter, NQuadraticBetter] = ...
            add_one_ttest_row(Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, NLinearBetter, NQuadraticBetter, ...
            'absolute_position', char(subset_names(si)), char(models(mi)), 'Standardised linear slope vs zero', vals, []);

        vals = Fits.LinearR2(rowf);
        [Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, NLinearBetter, NQuadraticBetter] = ...
            add_descriptive_row(Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, NLinearBetter, NQuadraticBetter, ...
            'absolute_position', char(subset_names(si)), char(models(mi)), 'Linear R2', vals);

        vals = Fits.QuadR2(rowf);
        [Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, NLinearBetter, NQuadraticBetter] = ...
            add_descriptive_row(Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, NLinearBetter, NQuadraticBetter, ...
            'absolute_position', char(subset_names(si)), char(models(mi)), 'Quadratic R2', vals);

        vals = Fits.DeltaR2_QuadMinusLinear(rowf);
        [Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, NLinearBetter, NQuadraticBetter] = ...
            add_one_ttest_row(Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, NLinearBetter, NQuadraticBetter, ...
            'absolute_position', char(subset_names(si)), char(models(mi)), 'Delta R2 quad-minus-linear vs zero', vals, []);

        vals = Fits.DeltaBIC_QuadMinusLinear(rowf);
        [Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, NLinearBetter, NQuadraticBetter] = ...
            add_one_ttest_row(Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, NLinearBetter, NQuadraticBetter, ...
            'absolute_position', char(subset_names(si)), char(models(mi)), 'Delta BIC quad-minus-linear vs zero', vals, vals);
    end
end

% Between-group tests for CS winners versus BP winners, separately for each fitted model.
for mi = 1:numel(models)
    this_model = models(mi);
    model_filter = string(Fits.Model)==this_model;
    g1 = string(Fits.BestFitModel)==groups(1) & model_filter;
    g2 = string(Fits.BestFitModel)==groups(2) & model_filter;

    measures = {'LinearSlopeStd','DeltaBIC_QuadMinusLinear','DeltaR2_QuadMinusLinear'};
    labels = {'Standardised linear slope group difference', ...
        'Delta BIC quad-minus-linear group difference', ...
        'Delta R2 quad-minus-linear group difference'};

    for ii = 1:numel(measures)
        v1 = Fits.(measures{ii})(g1);
        v2 = Fits.(measures{ii})(g2);
        [p,tstat,df,mean_diff,n_total] = safe_ttest2(v1, v2);

        Analysis{end+1,1} = 'absolute_position'; %#ok<AGROW>
        Subset{end+1,1} = 'Cost-to-sample vs biased-prior best-fitted participants'; %#ok<AGROW>
        ModelOut{end+1,1} = char(this_model); %#ok<AGROW>
        Measure{end+1,1} = labels{ii}; %#ok<AGROW>
        N(end+1,1) = n_total; %#ok<AGROW>
        MeanValue(end+1,1) = mean_diff; %#ok<AGROW>
        SDValue(end+1,1) = NaN; %#ok<AGROW>
        Tstat(end+1,1) = tstat; %#ok<AGROW>
        DF(end+1,1) = df; %#ok<AGROW>
        P(end+1,1) = p; %#ok<AGROW>
        NLinearBetter(end+1,1) = NaN; %#ok<AGROW>
        NQuadraticBetter(end+1,1) = NaN; %#ok<AGROW>
    end
end

Summary = table(Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, ...
    NLinearBetter, NQuadraticBetter);
end

function [Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, NLinearBetter, NQuadraticBetter] = ...
    add_one_ttest_row(Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, NLinearBetter, NQuadraticBetter, ...
    analysis_label, subset_label, model_label, measure_label, vals, delta_bic_vals)

[p,tstat,df,m,sd,n] = safe_ttest(vals);

Analysis{end+1,1} = analysis_label; %#ok<AGROW>
Subset{end+1,1} = subset_label; %#ok<AGROW>
ModelOut{end+1,1} = model_label; %#ok<AGROW>
Measure{end+1,1} = measure_label; %#ok<AGROW>
N(end+1,1) = n; %#ok<AGROW>
MeanValue(end+1,1) = m; %#ok<AGROW>
SDValue(end+1,1) = sd; %#ok<AGROW>
Tstat(end+1,1) = tstat; %#ok<AGROW>
DF(end+1,1) = df; %#ok<AGROW>
P(end+1,1) = p; %#ok<AGROW>

if isempty(delta_bic_vals)
    NLinearBetter(end+1,1) = NaN; %#ok<AGROW>
    NQuadraticBetter(end+1,1) = NaN; %#ok<AGROW>
else
    db = delta_bic_vals(isfinite(delta_bic_vals));
    % DeltaBIC = BIC_quad - BIC_linear. Positive means linear lower BIC.
    NLinearBetter(end+1,1) = sum(db > 0); %#ok<AGROW>
    NQuadraticBetter(end+1,1) = sum(db < 0); %#ok<AGROW>
end
end

function [Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, NLinearBetter, NQuadraticBetter] = ...
    add_descriptive_row(Analysis, Subset, ModelOut, Measure, N, MeanValue, SDValue, Tstat, DF, P, NLinearBetter, NQuadraticBetter, ...
    analysis_label, subset_label, model_label, measure_label, vals)

vals = vals(isfinite(vals));

Analysis{end+1,1} = analysis_label; %#ok<AGROW>
Subset{end+1,1} = subset_label; %#ok<AGROW>
ModelOut{end+1,1} = model_label; %#ok<AGROW>
Measure{end+1,1} = measure_label; %#ok<AGROW>
N(end+1,1) = numel(vals); %#ok<AGROW>
MeanValue(end+1,1) = mean(vals,'omitnan'); %#ok<AGROW>
SDValue(end+1,1) = std(vals,'omitnan'); %#ok<AGROW>
Tstat(end+1,1) = NaN; %#ok<AGROW>
DF(end+1,1) = NaN; %#ok<AGROW>
P(end+1,1) = NaN; %#ok<AGROW>
NLinearBetter(end+1,1) = NaN; %#ok<AGROW>
NQuadraticBetter(end+1,1) = NaN; %#ok<AGROW>
end

% =========================================================================
% STDOUT REPORT
% =========================================================================
function print_absolute_second_level(Summary, cfg)

models = string(cfg.model_names);
subsets = ["All participants", ...
    "Participants best fitted by cost-to-sample", ...
    "Participants best fitted by biased-prior"];

fprintf('\n====================================================================\n');
fprintf('ABSOLUTE-POSITION SECOND-LEVEL THRESHOLD RESULTS\n');
fprintf('====================================================================\n');
fprintf('Outcome: continuation-value threshold QContinue.\n');
fprintf('Linear-vs-quadratic comparison uses DeltaBIC = BIC_quadratic - BIC_linear.\n');
fprintf('  DeltaBIC > 0 favours the linear model.\n');
fprintf('  DeltaBIC < 0 favours the quadratic model.\n');
fprintf('Standardised slope is the participant-level linear slope after putting\n');
fprintf('position and threshold on z-score scales, equivalent to the linear r.\n\n');

for si = 1:numel(subsets)
    fprintf('\n%s\n', subsets(si));
    fprintf('%s\n', repmat('-',1,strlength(subsets(si))));

    for mi = 1:numel(models)
        fprintf('\n  Fitted threshold model: %s\n', models(mi));

        print_row(Summary, subsets(si), models(mi), 'Standardised linear slope vs zero', ...
            '    Standardised linear slope');

        print_row(Summary, subsets(si), models(mi), 'Raw linear slope vs zero', ...
            '    Raw linear slope');

        print_row(Summary, subsets(si), models(mi), 'Linear R2', ...
            '    Linear R2');

        print_row(Summary, subsets(si), models(mi), 'Quadratic R2', ...
            '    Quadratic R2');

        print_row(Summary, subsets(si), models(mi), 'Delta R2 quad-minus-linear vs zero', ...
            '    Delta R2 (quadratic - linear)');

        print_row(Summary, subsets(si), models(mi), 'Delta BIC quad-minus-linear vs zero', ...
            '    Delta BIC (quadratic - linear)');
    end
end

fprintf('\nBetween-group comparisons: cost-to-sample best-fitted participants vs biased-prior best-fitted participants\n');
fprintf('-----------------------------------------------------------------------------------------------\n');
for mi = 1:numel(models)
    fprintf('\n  Fitted threshold model: %s\n', models(mi));
    print_group_row(Summary, models(mi), 'Standardised linear slope group difference', ...
        '    Standardised slope group difference');
    print_group_row(Summary, models(mi), 'Delta BIC quad-minus-linear group difference', ...
        '    Delta BIC group difference');
    print_group_row(Summary, models(mi), 'Delta R2 quad-minus-linear group difference', ...
        '    Delta R2 group difference');
end

fprintf('\n====================================================================\n\n');
end

function print_row(Summary, subset, model, measure, label)

row = string(Summary.Subset)==subset & string(Summary.ModelOut)==model & string(Summary.Measure)==measure;
if ~any(row)
    fprintf('%s: not found\n', label);
    return;
end
r = Summary(row,:);

if contains(measure, 'R2') && ~contains(measure, 'Delta')
    fprintf('%s: M = %.4f, SD = %.4f, N = %d\n', ...
        label, r.MeanValue, r.SDValue, r.N);
elseif contains(measure, 'Delta BIC')
    fprintf('%s: M = %.4f, SD = %.4f, t(%g) = %.3f, p = %.4f, N = %d; linear better = %g, quadratic better = %g\n', ...
        label, r.MeanValue, r.SDValue, r.DF, r.Tstat, r.P, r.N, r.NLinearBetter, r.NQuadraticBetter);
else
    fprintf('%s: M = %.4f, SD = %.4f, t(%g) = %.3f, p = %.4f, N = %d\n', ...
        label, r.MeanValue, r.SDValue, r.DF, r.Tstat, r.P, r.N);
end
end

function print_group_row(Summary, model, measure, label)

row = string(Summary.Subset)=="Cost-to-sample vs biased-prior best-fitted participants" & ...
    string(Summary.ModelOut)==model & string(Summary.Measure)==measure;
if ~any(row)
    fprintf('%s: not found\n', label);
    return;
end
r = Summary(row,:);
fprintf('%s: mean difference = %.4f, t(%g) = %.3f, p = %.4f, N = %d\n', ...
    label, r.MeanValue, r.DF, r.Tstat, r.P, r.N);
end

% =========================================================================
% POLYNOMIAL FIT UTILITY
% =========================================================================
function out = fit_poly_stats(x, y, degree)

out.intercept = NaN;
out.slope = NaN;
out.slope_std = NaN;
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
    sx = std(x);
    sy = std(y);
    if sx > 0 && sy > 0
        out.slope_std = out.slope * sx / sy;
    else
        out.slope_std = NaN;
    end
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

% =========================================================================
% STATS HELPERS
% =========================================================================
function [p,tstat,df,m,sd,n] = safe_ttest(vals)

vals = vals(isfinite(vals));
n = numel(vals);
m = mean(vals,'omitnan');
sd = std(vals,'omitnan');

if n >= 2 && sd > 0
    [~,p,~,stats] = ttest(vals);
    tstat = stats.tstat;
    df = stats.df;
else
    p = NaN;
    tstat = NaN;
    df = NaN;
end
end

function [p,tstat,df,mean_diff,n_total] = safe_ttest2(v1, v2)

v1 = v1(isfinite(v1));
v2 = v2(isfinite(v2));
n_total = numel(v1) + numel(v2);
mean_diff = mean(v1,'omitnan') - mean(v2,'omitnan');

if numel(v1) >= 2 && numel(v2) >= 2 && (std(v1) > 0 || std(v2) > 0)
    [~,p,~,stats] = ttest2(v1, v2);
    tstat = stats.tstat;
    df = stats.df;
else
    p = NaN;
    tstat = NaN;
    df = NaN;
end
end

function write_table(T, outdir, filename)

file = [outdir filesep filename];
writetable(T, file);
fprintf('Wrote table:\n  %s\n', file);
end
