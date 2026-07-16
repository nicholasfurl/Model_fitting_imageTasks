% =========================================================================
% IMAGE TASK RFX BAYESIAN MODEL SELECTION
% -------------------------------------------------------------------------
% Standalone random-effects Bayesian model-selection analysis for the
% image-task model fits. This intentionally does NOT modify the figure code.
%
% It reuses the plotting-code infrastructure only in the sense that it uses
% the same fit-file paths and the same model ordering:
%   Cost to sample, Cut off, Biased prior
%
% It reuses Christina's RFX BMS infrastructure:
%   BIC -> approximate log model evidence, lme = -0.5*BIC
%   Random-effects Bayesian model selection
%   Expected frequencies, exceedance probabilities, protected exceedance
%   probabilities, and BOR.
%
% Outputs:
%   imageTask_RFX_BMS_out/
%       all_group_summary.csv
%       all_per_subject_IC.csv
%       <analysis_set>_group_summary.csv
%       <analysis_set>_per_subject_IC.csv
%       <analysis_set>_BMS.png
%       imageTask_RFX_BMS_results.mat
%
% Nicholas / ChatGPT draft v1, June 2026.
% =========================================================================

function imageTasks_RFX_BMS_v1()

clc;

% ----------------------------- CONFIG ------------------------------------
cfg.outdir = fullfile(pwd, 'imageTask_RFX_BMS_out');
cfg.Nsamp  = 1e6;      % Monte-Carlo samples for exceedance probabilities
cfg.seed   = 1;
cfg.make_figures = 1;

% Use the standard BIC penalty. This is the value RFX BMS should receive,
% especially if any model comparison later includes models with different k.
cfg.use_standard_BIC = 1;   % 1 = 2*NLL + k*log(n); 0 = legacy plotting scale 2*NLL + k*n

% Original model indices inside Generate_params.model, in the order used in
% the manuscript model-comparison panels after excluding Human/Ground truth.
% In the plotting code, bar_order = [4 0 2 1 3], where 4 = ground truth,
% 0 = humans, 2 = cost to sample, 1 = cut off, 3 = biased prior.
cfg.model_indices = [2 1 3];
cfg.model_names   = {'Cost to sample','Cut off','Biased prior'};

% Output path and fit files copied from the current manuscript plotting code.
outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\outputs';
cfg.file_paths = {...
    [outpath filesep 'out_imageTask_face_1_COCSBMP2_20252802.mat']
    [outpath filesep 'out_imageTask_face_2_COCSBMP2_20252802.mat']
    [outpath filesep 'out_imageTask_face_3_COCSBMP2_20252802.mat']
    [outpath filesep 'out_imageTask_matchmaker_COCSBMP2_20252802.mat']
    [outpath filesep 'out_imageTask_trust_1_COCSBMP2_20252802.mat']
    [outpath filesep 'out_imageTask_trust_2_COCSBMP2_20250103.mat']
    [outpath filesep 'out_imageTask_food_1_COCSBMP2_20250103.mat']
    [outpath filesep 'out_imageTask_food_2_COCSBMP2_20250103.mat']
    [outpath filesep 'out_imageTask_holiday_1_COCSBMP2_20250103.mat']
    [outpath filesep 'out_imageTask_holiday_2_COCSBMP2_20250103.mat']
    };

cfg.study_names = {...
    'Facial attractiveness dataset 1'
    'Facial attractiveness dataset 2'
    'Facial attractiveness dataset 3'
    'Matchmaker dataset'
    'Trustworthiness dataset 1'
    'Trustworthiness dataset 2'
    'Foods dataset 1'
    'Foods dataset 2'
    'Vacations dataset 1'
    'Vacations dataset 2'
    };

cfg.study_short = {...
    'face_1'
    'face_2'
    'face_3'
    'matchmaker'
    'trust_1'
    'trust_2'
    'foods_1'
    'foods_2'
    'vacations_1'
    'vacations_2'
    };

% Analysis sets. The first ten are the individual datasets. The grouped
% analyses are useful summaries, but treat pooled participants/tasks as
% exchangeable, so describe them as descriptive unless you decide otherwise.
cfg.analysis_sets = struct('name',{},'file_indices',{},'description',{});
for i = 1:numel(cfg.file_paths)
    cfg.analysis_sets(i).name = cfg.study_short{i};
    cfg.analysis_sets(i).file_indices = i;
    cfg.analysis_sets(i).description = cfg.study_names{i};
end
cfg.analysis_sets(end+1).name = 'faces_all';
cfg.analysis_sets(end).file_indices = [1 2 3];
cfg.analysis_sets(end).description = 'Facial attractiveness datasets 1-3 pooled';

cfg.analysis_sets(end+1).name = 'match_trust_all';
cfg.analysis_sets(end).file_indices = [4 5 6];
cfg.analysis_sets(end).description = 'Matchmaker and trustworthiness datasets pooled';

cfg.analysis_sets(end+1).name = 'foods_vacations_all';
cfg.analysis_sets(end).file_indices = [7 8 9 10];
cfg.analysis_sets(end).description = 'Foods and vacations datasets pooled';

cfg.analysis_sets(end+1).name = 'all_datasets';
cfg.analysis_sets(end).file_indices = 1:10;
cfg.analysis_sets(end).description = 'All image-task datasets pooled';

% -------------------------------------------------------------------------

if ~exist(cfg.outdir, 'dir'); mkdir(cfg.outdir); end
rng(cfg.seed);

fprintf('\nIMAGE TASK RFX BMS\n');
fprintf('Output folder: %s\n', cfg.outdir);
fprintf('RFX exceedance samples: %g\n', cfg.Nsamp);

% ---- preflight: check files exist ----
for f = 1:numel(cfg.file_paths)
    if ~isfile(cfg.file_paths{f})
        error('Fit file not found:\n  %s\nEdit cfg.file_paths in imageTasks_RFX_BMS_v1.m.', cfg.file_paths{f});
    end
end

% ---- load each individual dataset once ----
dataset = struct([]);
for f = 1:numel(cfg.file_paths)
    fprintf('\nLoading %s\n', cfg.study_names{f});
    dataset(f) = load_image_task_evidence(cfg.file_paths{f}, cfg.study_names{f}, cfg.study_short{f}, cfg);
end

% ---- run all requested analysis sets ----
results = struct();
allGroupRows = table();
allSubjectRows = table();

for s = 1:numel(cfg.analysis_sets)
    set_cfg = cfg.analysis_sets(s);
    this_datasets = dataset(set_cfg.file_indices);
    R = run_rfx_set(this_datasets, set_cfg, cfg);
    results.(safe_field(set_cfg.name)) = R;

    groupT = make_group_table(R, cfg);
    subT   = make_subject_table(R, cfg);

    writetable(groupT, fullfile(cfg.outdir, sprintf('%s_group_summary.csv', safe_file(set_cfg.name))));
    writetable(subT,   fullfile(cfg.outdir, sprintf('%s_per_subject_IC.csv', safe_file(set_cfg.name))));

    allGroupRows = [allGroupRows; groupT];
    allSubjectRows = [allSubjectRows; subT];

    if cfg.make_figures
        save_bms_figure(R, cfg);
    end
end

writetable(allGroupRows, fullfile(cfg.outdir, 'all_group_summary.csv'));
writetable(allSubjectRows, fullfile(cfg.outdir, 'all_per_subject_IC.csv'));

save(fullfile(cfg.outdir, 'imageTask_RFX_BMS_results.mat'), 'results', 'dataset', 'cfg');

fprintf('\nDone. Outputs written to:\n  %s\n', cfg.outdir);

end % main


% =========================================================================
% Dataset extraction
% =========================================================================
function D = load_image_task_evidence(file_path, study_name, study_short, cfg)

S = load(file_path, 'Generate_params');
if ~isfield(S, 'Generate_params')
    error('File %s does not contain Generate_params.', file_path);
end
G = S.Generate_params;

M = numel(cfg.model_indices);
Nsub = size(G.num_samples, 2);

% n_obs is the number of modelled binary choices entering the likelihood.
% When all_draws == 1, each sequence contributes one continue/stop decision
% for each sampled option, including the final stop decision.
n_obs = nansum(G.num_samples, 1)';

NLL = nan(Nsub, M);
kvec = nan(1, M);
model_names_from_file = cell(1, M);

for m = 1:M
    model_idx = cfg.model_indices(m);

    if model_idx > numel(G.model)
        error('Model index %d not found in %s.', model_idx, file_path);
    end

    NLL(:,m) = G.model(model_idx).ll(:);
    kvec(m) = numel(G.model(model_idx).this_models_free_parameters) + 1; % + beta / decision noise
    model_names_from_file{m} = char(G.model(model_idx).name);
end

keep = isfinite(n_obs) & n_obs > 0 & all(isfinite(NLL), 2);
if any(~keep)
    fprintf('  Dropping %d subject(s) with non-finite n_obs/NLL.\n', sum(~keep));
end

n_obs = n_obs(keep);
NLL = NLL(keep,:);
subject_index = find(keep)';

Nsub_keep = size(NLL,1);

AIC = 2*NLL + 2*repmat(kvec, Nsub_keep, 1);
if cfg.use_standard_BIC
    BIC = 2*NLL + repmat(log(n_obs),1,M).*repmat(kvec,Nsub_keep,1);
else
    BIC = 2*NLL + repmat(n_obs,1,M).*repmat(kvec,Nsub_keep,1);
end

fprintf('  Subjects: %d | n_obs min/median/max = %g / %g / %g\n', ...
    Nsub_keep, min(n_obs), median(n_obs), max(n_obs));
fprintf('  k values: %s\n', mat2str(kvec));

D = struct();
D.name = study_name;
D.short = study_short;
D.file_path = file_path;
D.subject_index = subject_index(:);
D.n_obs = n_obs;
D.NLL = NLL;
D.AIC = AIC;
D.BIC = BIC;
D.k = kvec;
D.model_names_from_file = model_names_from_file;

end


% =========================================================================
% RFX set analysis
% =========================================================================
function R = run_rfx_set(datasets, set_cfg, cfg)

M = numel(cfg.model_names);

NLL = [];
AIC = [];
BIC = [];
n_obs = [];
dataset_name = strings(0,1);
dataset_short = strings(0,1);
subject_index = [];

for d = 1:numel(datasets)
    Nd = size(datasets(d).NLL, 1);

    NLL = [NLL; datasets(d).NLL];
    AIC = [AIC; datasets(d).AIC];
    BIC = [BIC; datasets(d).BIC];
    n_obs = [n_obs; datasets(d).n_obs];

    dataset_name = [dataset_name; repmat(string(datasets(d).name), Nd, 1)];
    dataset_short = [dataset_short; repmat(string(datasets(d).short), Nd, 1)];
    subject_index = [subject_index; datasets(d).subject_index(:)];
end

Nsub = size(BIC,1);
kvec = datasets(1).k;

% Basic IC summaries
sumNLL = sum(NLL, 1);
sumAIC = sum(AIC, 1);
sumBIC = sum(BIC, 1);

dAIC = sumAIC - min(sumAIC);
dBIC = sumBIC - min(sumBIC);

wAIC_group = ic_weights(sumAIC);
wBIC_group = ic_weights(sumBIC);

[~, bestAIC] = min(AIC, [], 2);
[~, bestBIC] = min(BIC, [], 2);
cntAIC = accumarray(bestAIC, 1, [M 1])';
cntBIC = accumarray(bestBIC, 1, [M 1])';

% Participant-level BIC weights
BIC_weight = row_ic_weights(BIC);
sum_BIC_weight = sum(BIC_weight, 1);

% Runner-up diagnostics
[BIC_sorted, ~] = sort(BIC, 2, 'ascend');
delta_runnerup = BIC_sorted(:,2) - BIC_sorted(:,1);
max_BIC_weight = max(BIC_weight, [], 2);

% RFX BMS using BIC approximation to log model evidence
lme = -0.5 * BIC;
[alpha, exp_r, xp, pxp, bor] = spm_BMS_local(lme, cfg.Nsamp);

fprintf('\n=====================================================\n');
fprintf(' ANALYSIS SET: %s\n', upper(set_cfg.name));
fprintf(' %s\n', set_cfg.description);
fprintf('=====================================================\n');
fprintf('  Subjects/rows: %d\n', Nsub);
fprintf('  Random-effects BMS (BIC evidence), BOR = %.3f\n', bor);
fprintf('  %-18s %5s %10s %10s %10s %10s %10s %10s\n', ...
    'Model','k','#bestBIC','sum_wBIC','exp_r','xp','pxp','dBIC');
for m = 1:M
    fprintf('  %-18s %5d %10d %10.2f %10.3f %10.3f %10.3f %10.1f\n', ...
        cfg.model_names{m}, kvec(m), cntBIC(m), sum_BIC_weight(m), ...
        exp_r(m), xp(m), pxp(m), dBIC(m));
end
fprintf('  Median runner-up delta BIC = %.2f\n', median(delta_runnerup));
fprintf('  %% delta runner-up > 2 / 6 / 10 = %.1f / %.1f / %.1f\n', ...
    100*mean(delta_runnerup > 2), 100*mean(delta_runnerup > 6), 100*mean(delta_runnerup > 10));
fprintf('  Mean max BIC weight = %.3f\n', mean(max_BIC_weight));

R = struct();
R.name = set_cfg.name;
R.description = set_cfg.description;
R.dataset_name = dataset_name;
R.dataset_short = dataset_short;
R.subject_index = subject_index;
R.model_names = cfg.model_names;
R.k = kvec;
R.n_obs = n_obs;
R.NLL = NLL;
R.AIC = AIC;
R.BIC = BIC;
R.BIC_weight = BIC_weight;
R.bestAIC = bestAIC;
R.bestBIC = bestBIC;
R.cntAIC = cntAIC;
R.cntBIC = cntBIC;
R.sumNLL = sumNLL;
R.sumAIC = sumAIC;
R.sumBIC = sumBIC;
R.dAIC = dAIC;
R.dBIC = dBIC;
R.wAIC_group = wAIC_group;
R.wBIC_group = wBIC_group;
R.sum_BIC_weight = sum_BIC_weight;
R.delta_runnerup = delta_runnerup;
R.max_BIC_weight = max_BIC_weight;
R.alpha = alpha;
R.exp_r = exp_r;
R.xp = xp;
R.pxp = pxp;
R.bor = bor;

end


% =========================================================================
% Output tables
% =========================================================================
function T = make_group_table(R, cfg)

M = numel(R.model_names);

T = table();
T.analysis_set = repmat(string(R.name), M, 1);
T.description = repmat(string(R.description), M, 1);
T.N_subject_rows = repmat(size(R.BIC,1), M, 1);
T.model_idx = (1:M)';
T.model = string(R.model_names(:));
T.k = R.k(:);
T.sumNLL = R.sumNLL(:);
T.sumAIC = R.sumAIC(:);
T.sumBIC = R.sumBIC(:);
T.dAIC = R.dAIC(:);
T.dBIC = R.dBIC(:);
T.group_wAIC = R.wAIC_group(:);
T.group_wBIC = R.wBIC_group(:);
T.n_bestAIC = R.cntAIC(:);
T.n_bestBIC = R.cntBIC(:);
T.summed_subject_BIC_weight = R.sum_BIC_weight(:);
T.exp_r = R.exp_r(:);
T.xp = R.xp(:);
T.pxp = R.pxp(:);
T.BOR = repmat(R.bor, M, 1);
T.median_runnerup_delta_BIC = repmat(median(R.delta_runnerup), M, 1);
T.pct_runnerup_delta_BIC_gt2 = repmat(100*mean(R.delta_runnerup > 2), M, 1);
T.pct_runnerup_delta_BIC_gt6 = repmat(100*mean(R.delta_runnerup > 6), M, 1);
T.pct_runnerup_delta_BIC_gt10 = repmat(100*mean(R.delta_runnerup > 10), M, 1);
T.mean_max_subject_BIC_weight = repmat(mean(R.max_BIC_weight), M, 1);

end


function T = make_subject_table(R, cfg)

N = size(R.BIC,1);
M = numel(R.model_names);
safe_model = matlab.lang.makeValidName(R.model_names);

T = table();
T.analysis_set = repmat(string(R.name), N, 1);
T.dataset = R.dataset_name;
T.dataset_short = R.dataset_short;
T.subject_index = R.subject_index(:);
T.n_obs = R.n_obs(:);
T.best_AIC = string(R.model_names(R.bestAIC))';
T.best_BIC = string(R.model_names(R.bestBIC))';
T.runnerup_delta_BIC = R.delta_runnerup(:);
T.max_BIC_weight = R.max_BIC_weight(:);

for m = 1:M
    T.(['NLL_' safe_model{m}]) = R.NLL(:,m);
    T.(['AIC_' safe_model{m}]) = R.AIC(:,m);
    T.(['BIC_' safe_model{m}]) = R.BIC(:,m);
    T.(['BIC_weight_' safe_model{m}]) = R.BIC_weight(:,m);
end

end


% =========================================================================
% Figures
% =========================================================================
function save_bms_figure(R, cfg)

try
    f = figure('Visible','off','Color','w','Position',[100 100 860 320]);

    subplot(1,2,1);
    bar(R.exp_r, 'FaceColor',[0.30 0.30 0.30], 'EdgeColor',[0 0 0]);
    ylim([0 1]);
    set(gca, 'XTick', 1:numel(R.model_names), 'XTickLabel', R.model_names, ...
        'XTickLabelRotation', 30, 'FontName','Arial');
    ylabel('Expected model frequency');
    title(sprintf('%s: RFX frequencies', strrep(R.name,'_',' ')), 'Interpreter','none');

    subplot(1,2,2);
    bar(R.pxp, 'FaceColor',[0.30 0.30 0.30], 'EdgeColor',[0 0 0]);
    ylim([0 1]);
    set(gca, 'XTick', 1:numel(R.model_names), 'XTickLabel', R.model_names, ...
        'XTickLabelRotation', 30, 'FontName','Arial');
    ylabel('Protected exceedance probability');
    title(sprintf('PXP (BOR = %.2f)', R.bor));

    saveas(f, fullfile(cfg.outdir, sprintf('%s_BMS.png', safe_file(R.name))));
    close(f);
catch ME
    warning('Could not save BMS figure for %s: %s', R.name, ME.message);
end

end


% =========================================================================
% Generic helpers
% =========================================================================
function w = ic_weights(ic)
% Akaike/Schwarz weights from a vector of summed IC values.
d = ic - min(ic);
w = exp(-0.5*d);
w = w ./ sum(w);
end


function W = row_ic_weights(IC)
% Participant-level IC weights. Rows sum to 1.
D = IC - min(IC, [], 2);
W = exp(-0.5*D);
W = W ./ sum(W, 2);
end


function s = safe_file(s)
s = char(s);
s = regexprep(s, '[^A-Za-z0-9_]+', '_');
end


function s = safe_field(s)
s = matlab.lang.makeValidName(char(s));
end


% =========================================================================
% RFX Bayesian model selection helpers from Christina's code
% =========================================================================
function [alpha, exp_r, xp, pxp, bor] = spm_BMS_local(lme, Nsamp)
% Random-effects Bayesian model selection (variational Dirichlet).
% Re-implementation of Stephan et al. (2009) + protected exceedance
% probability / BOR (Rigoux et al., 2014). Base-MATLAB only.
%
%   lme   : [Nsub x Nmodel] log model evidence (here -0.5*BIC)
%   alpha : Dirichlet concentration parameters of q(r)
%   exp_r : expected posterior model frequencies
%   xp    : exceedance probabilities
%   pxp   : protected exceedance probabilities
%   bor   : Bayesian Omnibus Risk (posterior prob of H0: r = 1/K)

if nargin < 2, Nsamp = 1e6; end
[Ni, Nk] = size(lme);
alpha0 = ones(1, Nk);
alpha  = alpha0;

for it = 1:10000
    alpha_prev = alpha;
    Elogr = psi(alpha) - psi(sum(alpha));     % 1 x Nk
    beta  = zeros(1, Nk);
    for i = 1:Ni
        lu = lme(i,:) + Elogr;
        lu = lu - max(lu);                    % numerical stabilisation
        u  = exp(lu);
        beta = beta + u / sum(u);             % responsibilities g_ik
    end
    alpha = alpha0 + beta;
    if norm(alpha - alpha_prev) < 1e-6, break; end
end

exp_r = alpha / sum(alpha);
xp    = dirichlet_exceedance(alpha, Nsamp);

% BOR / protected exceedance probability
F1 = free_energy_H1(lme, alpha, alpha0);
F0 = sum(logsumexp_rows(lme) - log(Nk));
bor = 1 / (1 + exp(F1 - F0));
pxp = (1 - bor) * xp + bor / Nk;

end


function F = free_energy_H1(lme, alpha, alpha0)
% Negative variational free energy (lower bound on ln p(Y|H1)).
Ni    = size(lme,1);
Elogr = psi(alpha) - psi(sum(alpha));

ELJ = 0; Sqm = 0;
for i = 1:Ni
    lu = lme(i,:) + Elogr;
    g  = exp(lu - max(lu));
    g  = g / sum(g);
    ELJ = ELJ + sum(g .* (lme(i,:) + Elogr));
    Sqm = Sqm - sum(g .* log(g + eps));
end

Eqlogpr = gammaln(sum(alpha0)) - sum(gammaln(alpha0)) + sum((alpha0-1).*Elogr);
Eqlogqr = gammaln(sum(alpha))  - sum(gammaln(alpha))  + sum((alpha -1).*Elogr);
F = ELJ + Eqlogpr - Eqlogqr + Sqm;

end


function y = logsumexp_rows(X)
m = max(X, [], 2);
y = m + log(sum(exp(X - m), 2));
end


function xp = dirichlet_exceedance(alpha, Nsamp)
% Exceedance probabilities by Monte-Carlo sampling of Dirichlet(alpha).
Nk = numel(alpha);
r  = zeros(Nsamp, Nk);
for k = 1:Nk
    r(:,k) = gamrnd_local(alpha(k), Nsamp);   % Gamma(alpha_k, 1)
end
r = r ./ sum(r, 2);                           % normalise -> Dirichlet
[~, w] = max(r, [], 2);
xp = accumarray(w, 1, [Nk 1])' / Nsamp;
end


function g = gamrnd_local(a, n)
% n x 1 samples from Gamma(shape a, scale 1) via Marsaglia & Tsang (2000).
% Base-MATLAB only (no Statistics Toolbox).
if a < 1
    g = gamrnd_local(a + 1, n) .* (rand(n,1) .^ (1/a));
    return;
end

d = a - 1/3;
c = 1 / sqrt(9*d);
g = zeros(n,1);
need = true(n,1);

while any(need)
    idx = find(need);
    m   = numel(idx);
    x   = randn(m,1);
    v   = (1 + c*x).^3;
    u   = rand(m,1);
    acc = (v > 0) & ( log(u) < 0.5*x.^2 + d - d*v + d*log(max(v, realmin)) );
    g(idx(acc))    = d * v(acc);
    need(idx(acc)) = false;
end

end
