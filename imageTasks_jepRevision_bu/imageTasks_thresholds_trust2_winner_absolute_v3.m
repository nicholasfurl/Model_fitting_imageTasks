function imageTasks_thresholds_trust2_winner_absolute_v3()
% Absolute-position threshold-shape analysis for Trustworthiness dataset 2.
% Requires output from imageTasks_extract_thresholds_trust2_v1.m.
% Analyses only each participant's winning model:
%   CS winners: CS QContinue thresholds
%   BP winners: BP QContinue thresholds
% Cut-off winners and loser-model thresholds are excluded.
% BIC convention:
%   Quadratic advantage = BIC_linear - BIC_quadratic
%   Positive values favour quadratic.

clc;
warning('off','all');

cfg.outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\outputs';
cfg.outdir  = [cfg.outpath filesep 'thresholds_trust2_out'];
cfg.infile  = [cfg.outdir filesep 'trust2_threshold_q_values.mat'];
cfg.groups = {'Cost-to-sample','Biased-prior'};
cfg.group_labels = {'Participants best fitted by cost-to-sample model', ...
                    'Participants best fitted by biased-prior model'};
cfg.group_short = {'Cost-to-sample winners','Biased-prior winners'};
cfg.cols = [0.0000 0.4470 0.7410; 0.8500 0.3250 0.0980];
cfg.pos = 1:7;

if ~isfile(cfg.infile)
    error('Cannot find %s. Run imageTasks_extract_thresholds_trust2_v1 first.', cfg.infile);
end
if ~exist(cfg.outdir,'dir'); mkdir(cfg.outdir); end

S = load(cfg.infile,'T');
T = S.T;

fprintf('\nTrustworthiness dataset 2 winner-model threshold analysis\n');
fprintf('Loaded: %s\n', cfg.infile);
fprintf('Outcome: QContinue, the continuation-value threshold in action-value space.\n');
fprintf('Absolute positions 1-7 only; terminal position excluded.\n');
fprintf('Only each participant''s winning model is analysed. Cut-off winners are excluded.\n\n');

Model = string(T.Model);
Best = string(T.BestFitModel);
groups = string(cfg.groups);
keep = ismember(Best,groups) & Model==Best & ismember(T.Position,cfg.pos) & isfinite(T.QContinue);
Tw = T(keep,:);

fprintf('Rows retained: %d\n', height(Tw));
for g = 1:numel(groups)
    subs = unique(Tw.Subject(string(Tw.BestFitModel)==groups(g)));
    fprintf('  %s: %d participants\n', groups(g), numel(subs));
end
fprintf('\n');

Part = participant_means(Tw,cfg);
Fits = participant_fits(Part);
Summary = second_level(Fits,cfg);

writetable(Part, [cfg.outdir filesep 'trust2_threshold_winner_absolute_participant_means_v3.csv']);
writetable(Fits, [cfg.outdir filesep 'trust2_threshold_winner_absolute_fits_v3.csv']);
writetable(Summary, [cfg.outdir filesep 'trust2_threshold_winner_absolute_summary_v3.csv']);
save([cfg.outdir filesep 'trust2_threshold_winner_absolute_v3.mat'],'cfg','Tw','Part','Fits','Summary');

print_summary(Summary,cfg);
make_curve_fig(Part,Fits,cfg,'linear');
make_curve_fig(Part,Fits,cfg,'quadratic');
make_point_fig(Fits,cfg);

fprintf('Done. Outputs written to:\n  %s\n', cfg.outdir);
end

function Part = participant_means(Tw,cfg)
Subject=[]; BestFitModel={}; Position=[]; MeanThreshold=[]; NObs=[];
for gi=1:numel(cfg.groups)
    group = string(cfg.groups{gi});
    subs = unique(Tw.Subject(string(Tw.BestFitModel)==group))';
    for si=1:numel(subs)
        sub = subs(si);
        for pi=1:numel(cfg.pos)
            pos = cfg.pos(pi);
            vals = Tw.QContinue(Tw.Subject==sub & Tw.Position==pos & string(Tw.BestFitModel)==group);
            vals = vals(isfinite(vals));
            if isempty(vals); continue; end
            Subject(end+1,1)=sub; %#ok<AGROW>
            BestFitModel{end+1,1}=char(group); %#ok<AGROW>
            Position(end+1,1)=pos; %#ok<AGROW>
            MeanThreshold(end+1,1)=mean(vals); %#ok<AGROW>
            NObs(end+1,1)=numel(vals); %#ok<AGROW>
        end
    end
end
Part = table(Subject,BestFitModel,Position,MeanThreshold,NObs);
end

function Fits = participant_fits(Part)
subs = unique(Part.Subject)';
Subject=[]; BestFitModel={}; NPositions=[];
LinIntercept=[]; LinSlope=[]; LinSlopeStd=[]; LinR2=[]; LinBIC=[];
QuadIntercept=[]; QuadSlope=[]; QuadTerm=[]; QuadR2=[]; QuadBIC=[];
QuadAdvBIC=[]; QuadAdvR2=[];
for si=1:numel(subs)
    sub = subs(si);
    this = Part(Part.Subject==sub,:);
    [x,ord] = sort(this.Position); y = this.MeanThreshold(ord);
    lin = poly_stats(x,y,1); quad = poly_stats(x,y,2);
    Subject(end+1,1)=sub; %#ok<AGROW>
    BestFitModel{end+1,1}=this.BestFitModel{1}; %#ok<AGROW>
    NPositions(end+1,1)=numel(unique(x)); %#ok<AGROW>
    LinIntercept(end+1,1)=lin.intercept; %#ok<AGROW>
    LinSlope(end+1,1)=lin.slope; %#ok<AGROW>
    LinSlopeStd(end+1,1)=lin.slope_std; %#ok<AGROW>
    LinR2(end+1,1)=lin.R2; %#ok<AGROW>
    LinBIC(end+1,1)=lin.BIC; %#ok<AGROW>
    QuadIntercept(end+1,1)=quad.intercept; %#ok<AGROW>
    QuadSlope(end+1,1)=quad.slope; %#ok<AGROW>
    QuadTerm(end+1,1)=quad.quad; %#ok<AGROW>
    QuadR2(end+1,1)=quad.R2; %#ok<AGROW>
    QuadBIC(end+1,1)=quad.BIC; %#ok<AGROW>
    QuadAdvBIC(end+1,1)=lin.BIC - quad.BIC; %#ok<AGROW>
    QuadAdvR2(end+1,1)=quad.R2 - lin.R2; %#ok<AGROW>
end
Fits = table(Subject,BestFitModel,NPositions,LinIntercept,LinSlope,LinSlopeStd,LinR2,LinBIC, ...
    QuadIntercept,QuadSlope,QuadTerm,QuadR2,QuadBIC,QuadAdvBIC,QuadAdvR2);
end

function out = poly_stats(x,y,deg)
out.intercept=NaN; out.slope=NaN; out.slope_std=NaN; out.quad=NaN; out.R2=NaN; out.BIC=NaN;
x=x(:); y=y(:); valid=isfinite(x)&isfinite(y); x=x(valid); y=y(valid);
if numel(unique(x)) < deg+1 || numel(y) < deg+1; return; end
p=polyfit(x,y,deg); yhat=polyval(p,x); resid=y-yhat;
SSE=sum(resid.^2); SST=sum((y-mean(y)).^2);
if SST>0; out.R2=1-SSE/SST; end
n=numel(y); k=deg+1; out.BIC=n*log(max(SSE,eps)/n)+k*log(n);
if deg==1
    out.slope=p(1); out.intercept=p(2);
    if std(x)>0 && std(y)>0; out.slope_std=out.slope*std(x)/std(y); end
else
    out.quad=p(1); out.slope=p(2); out.intercept=p(3);
end
end

function Summary = second_level(Fits,cfg)
Subset={}; Measure={}; N=[]; MeanValue=[]; SDValue=[]; Tstat=[]; DF=[]; P=[]; NQuadraticBetter=[]; NLinearBetter=[];
for gi=1:numel(cfg.groups)
    group=string(cfg.groups{gi}); f=string(Fits.BestFitModel)==group;
    [Subset,Measure,N,MeanValue,SDValue,Tstat,DF,P,NQuadraticBetter,NLinearBetter]=add_t(Subset,Measure,N,MeanValue,SDValue,Tstat,DF,P,NQuadraticBetter,NLinearBetter,char(group),'Standardised linear slope vs zero',Fits.LinSlopeStd(f),[]);
    [Subset,Measure,N,MeanValue,SDValue,Tstat,DF,P,NQuadraticBetter,NLinearBetter]=add_t(Subset,Measure,N,MeanValue,SDValue,Tstat,DF,P,NQuadraticBetter,NLinearBetter,char(group),'Raw linear slope vs zero',Fits.LinSlope(f),[]);
    [Subset,Measure,N,MeanValue,SDValue,Tstat,DF,P,NQuadraticBetter,NLinearBetter]=add_d(Subset,Measure,N,MeanValue,SDValue,Tstat,DF,P,NQuadraticBetter,NLinearBetter,char(group),'Linear R2',Fits.LinR2(f));
    [Subset,Measure,N,MeanValue,SDValue,Tstat,DF,P,NQuadraticBetter,NLinearBetter]=add_d(Subset,Measure,N,MeanValue,SDValue,Tstat,DF,P,NQuadraticBetter,NLinearBetter,char(group),'Quadratic R2',Fits.QuadR2(f));
    [Subset,Measure,N,MeanValue,SDValue,Tstat,DF,P,NQuadraticBetter,NLinearBetter]=add_t(Subset,Measure,N,MeanValue,SDValue,Tstat,DF,P,NQuadraticBetter,NLinearBetter,char(group),'Quadratic R2 advantage vs zero',Fits.QuadAdvR2(f),[]);
    [Subset,Measure,N,MeanValue,SDValue,Tstat,DF,P,NQuadraticBetter,NLinearBetter]=add_t(Subset,Measure,N,MeanValue,SDValue,Tstat,DF,P,NQuadraticBetter,NLinearBetter,char(group),'Quadratic BIC advantage vs zero',Fits.QuadAdvBIC(f),Fits.QuadAdvBIC(f));
end
cs=string(Fits.BestFitModel)==string(cfg.groups{1}); bp=string(Fits.BestFitModel)==string(cfg.groups{2});
fields={'LinSlopeStd','LinSlope','QuadAdvBIC','QuadAdvR2','LinR2','QuadR2'};
labels={'Standardised linear slope group difference','Raw linear slope group difference','Quadratic BIC advantage group difference','Quadratic R2 advantage group difference','Linear R2 group difference','Quadratic R2 group difference'};
for ii=1:numel(fields)
    v1=Fits.(fields{ii})(cs); v2=Fits.(fields{ii})(bp); [p,t,df,md,n]=ttest2_safe(v1,v2);
    Subset{end+1,1}='Cost-to-sample minus biased-prior winners'; %#ok<AGROW>
    Measure{end+1,1}=labels{ii}; %#ok<AGROW>
    N(end+1,1)=n; MeanValue(end+1,1)=md; SDValue(end+1,1)=NaN; Tstat(end+1,1)=t; DF(end+1,1)=df; P(end+1,1)=p; %#ok<AGROW>
    NQuadraticBetter(end+1,1)=NaN; NLinearBetter(end+1,1)=NaN; %#ok<AGROW>
end
Summary=table(Subset,Measure,N,MeanValue,SDValue,Tstat,DF,P,NQuadraticBetter,NLinearBetter);
end

function [Subset,Measure,N,MeanValue,SDValue,Tstat,DF,P,NQuadraticBetter,NLinearBetter]=add_t(Subset,Measure,N,MeanValue,SDValue,Tstat,DF,P,NQuadraticBetter,NLinearBetter,subset,measure,vals,bicvals)
[p,t,df,m,sd,n]=ttest_safe(vals);
Subset{end+1,1}=subset; Measure{end+1,1}=measure; N(end+1,1)=n; MeanValue(end+1,1)=m; SDValue(end+1,1)=sd; Tstat(end+1,1)=t; DF(end+1,1)=df; P(end+1,1)=p; %#ok<AGROW>
if isempty(bicvals)
    NQuadraticBetter(end+1,1)=NaN; NLinearBetter(end+1,1)=NaN; %#ok<AGROW>
else
    b=bicvals(isfinite(bicvals));
    NQuadraticBetter(end+1,1)=sum(b>0); NLinearBetter(end+1,1)=sum(b<0); %#ok<AGROW>
end
end

function [Subset,Measure,N,MeanValue,SDValue,Tstat,DF,P,NQuadraticBetter,NLinearBetter]=add_d(Subset,Measure,N,MeanValue,SDValue,Tstat,DF,P,NQuadraticBetter,NLinearBetter,subset,measure,vals)
vals=vals(isfinite(vals));
Subset{end+1,1}=subset; Measure{end+1,1}=measure; N(end+1,1)=numel(vals); MeanValue(end+1,1)=mean(vals); SDValue(end+1,1)=std(vals); Tstat(end+1,1)=NaN; DF(end+1,1)=NaN; P(end+1,1)=NaN; NQuadraticBetter(end+1,1)=NaN; NLinearBetter(end+1,1)=NaN; %#ok<AGROW>
end

function make_curve_fig(Part,Fits,cfg,kind)
if strcmp(kind,'linear'); outfile='trust2_threshold_winner_absolute_linear_curves_v3'; fitlab='linear'; else; outfile='trust2_threshold_winner_absolute_quadratic_curves_v3'; fitlab='quadratic'; end
x=cfg.pos; xd=linspace(min(x),max(x),200);
f=figure('Color','w','Name',[fitlab ' winner-model threshold curves']); t=tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');
for gi=1:numel(cfg.groups)
    group=string(cfg.groups{gi}); subs=unique(Part.Subject(string(Part.BestFitModel)==group))';
    ax=nexttile(t,gi); hold(ax,'on'); box(ax,'off');
    for si=1:numel(subs)
        th=Part(Part.Subject==subs(si),:); [xx,oo]=sort(th.Position); yy=th.MeanThreshold(oo); plot(ax,xx,yy,'-','Color',[.78 .78 .78],'LineWidth',.75);
    end
    yg=group_curve(Fits,group,xd,kind); plot(ax,xd,yg,'-','Color',cfg.cols(gi,:),'LineWidth',3);
    xlabel(ax,'Sequence position'); ylabel(ax,'Continuation-value threshold'); title(ax,sprintf('%s\nparticipant means + mean %s fit',cfg.group_short{gi},fitlab)); set(ax,'FontName','Arial','FontSize',9,'XTick',x);
    ax=nexttile(t,gi+2); hold(ax,'on'); box(ax,'off');
    for si=1:numel(subs)
        fr=Fits(Fits.Subject==subs(si),:); if isempty(fr); continue; end; yf=one_curve(fr,xd,kind); plot(ax,xd,yf,'-','Color',[.78 .78 .78],'LineWidth',.75);
    end
    plot(ax,xd,yg,'-','Color',cfg.cols(gi,:),'LineWidth',3);
    xlabel(ax,'Sequence position'); ylabel(ax,'Continuation-value threshold'); title(ax,sprintf('%s\nparticipant %s fits + mean fit',cfg.group_short{gi},fitlab)); set(ax,'FontName','Arial','FontSize',9,'XTick',x);
end
saveas(f,[cfg.outdir filesep outfile '.png']); print(f,[cfg.outdir filesep outfile '.tif'],'-dtiff','-r600');
end

function make_point_fig(Fits,cfg)
f=figure('Color','w','Name','Winner-model slope and curvature point spreads'); t=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
metrics={'LinSlopeStd','QuadAdvBIC'}; ylabs={'Standardised linear slope','Quadratic advantage in BIC'}; tits={'Linear slope','Quadratic vs linear'};
for pi=1:2
    ax=nexttile(t,pi); hold(ax,'on'); box(ax,'off');
    for gi=1:numel(cfg.groups)
        vals=Fits.(metrics{pi})(string(Fits.BestFitModel)==string(cfg.groups{gi})); vals=vals(isfinite(vals)); rng(gi+10*pi); xj=gi+(rand(size(vals))-.5)*.18;
        plot(ax,xj,vals,'o','Color',[.25 .25 .25],'MarkerFaceColor',[.75 .75 .75],'MarkerSize',4);
        plot(ax,[gi-.25 gi+.25],[mean(vals) mean(vals)],'-','Color',cfg.cols(gi,:),'LineWidth',3);
    end
    plot(ax,[.5 2.5],[0 0],'k:'); xlim(ax,[.5 2.5]); set(ax,'XTick',1:2,'XTickLabel',cfg.group_short,'FontName','Arial','FontSize',9); xtickangle(ax,25); ylabel(ax,ylabs{pi}); title(ax,tits{pi});
end
saveas(f,[cfg.outdir filesep 'trust2_threshold_winner_absolute_pointspread_v3.png']); print(f,[cfg.outdir filesep 'trust2_threshold_winner_absolute_pointspread_v3.tif'],'-dtiff','-r600');
end

function y=group_curve(Fits,group,x,kind)
fr=Fits(string(Fits.BestFitModel)==string(group),:);
if strcmp(kind,'linear')
    b0=mean(fr.LinIntercept,'omitnan'); b1=mean(fr.LinSlope,'omitnan'); y=b0+b1*x;
else
    b0=mean(fr.QuadIntercept,'omitnan'); b1=mean(fr.QuadSlope,'omitnan'); b2=mean(fr.QuadTerm,'omitnan'); y=b0+b1*x+b2*x.^2;
end
end

function y=one_curve(fr,x,kind)
if strcmp(kind,'linear'); y=fr.LinIntercept+fr.LinSlope*x; else; y=fr.QuadIntercept+fr.QuadSlope*x+fr.QuadTerm*x.^2; end
end

function print_summary(Summary,cfg)
fprintf('\n====================================================================\n');
fprintf('WINNER-MODEL ABSOLUTE-POSITION THRESHOLD RESULTS\n');
fprintf('====================================================================\n');
fprintf('Only each participant''s winning model is analysed.\n');
fprintf('Quadratic advantage = BIC_linear - BIC_quadratic. Positive favours quadratic.\n\n');
for gi=1:numel(cfg.groups)
    group=string(cfg.groups{gi}); fprintf('%s\n%s\n',cfg.group_labels{gi},repmat('-',1,numel(cfg.group_labels{gi})));
    pr(Summary,group,'Standardised linear slope vs zero','  Standardised linear slope');
    pr(Summary,group,'Raw linear slope vs zero','  Raw linear slope');
    pr(Summary,group,'Linear R2','  Linear R2');
    pr(Summary,group,'Quadratic R2','  Quadratic R2');
    pr(Summary,group,'Quadratic R2 advantage vs zero','  Quadratic R2 advantage');
    pr(Summary,group,'Quadratic BIC advantage vs zero','  Quadratic BIC advantage');
    fprintf('\n');
end
fprintf('Between-group comparison: cost-to-sample winners minus biased-prior winners\n');
fprintf('--------------------------------------------------------------------------\n');
rows=Summary(string(Summary.Subset)=='Cost-to-sample minus biased-prior winners',:);
for i=1:height(rows)
    r=rows(i,:); fprintf('  %s: mean difference = %.4f, t(%g) = %.3f, p = %.4f, N = %d\n',r.Measure{1},r.MeanValue,r.DF,r.Tstat,r.P,r.N);
end
fprintf('====================================================================\n\n');
end

function pr(Summary,group,measure,label)
r=Summary(string(Summary.Subset)==group & string(Summary.Measure)==measure,:);
if isempty(r); fprintf('%s: not found\n',label); return; end
if contains(measure,'R2') && ~contains(measure,'advantage','IgnoreCase',true)
    fprintf('%s: M = %.4f, SD = %.4f, N = %d\n',label,r.MeanValue,r.SDValue,r.N);
elseif contains(measure,'BIC')
    fprintf('%s: M = %.4f, SD = %.4f, t(%g) = %.3f, p = %.4f, N = %d; quadratic better = %g, linear better = %g\n',label,r.MeanValue,r.SDValue,r.DF,r.Tstat,r.P,r.N,r.NQuadraticBetter,r.NLinearBetter);
else
    fprintf('%s: M = %.4f, SD = %.4f, t(%g) = %.3f, p = %.4f, N = %d\n',label,r.MeanValue,r.SDValue,r.DF,r.Tstat,r.P,r.N);
end
end

function [p,t,df,m,sd,n]=ttest_safe(vals)
vals=vals(isfinite(vals)); n=numel(vals); m=mean(vals); sd=std(vals);
if n>=2 && sd>0; [~,p,~,stats]=ttest(vals); t=stats.tstat; df=stats.df; else; p=NaN; t=NaN; df=NaN; end
end

function [p,t,df,md,n]=ttest2_safe(v1,v2)
v1=v1(isfinite(v1)); v2=v2(isfinite(v2)); md=mean(v1)-mean(v2); n=numel(v1)+numel(v2);
if numel(v1)>=2 && numel(v2)>=2 && (std(v1)>0 || std(v2)>0); [~,p,~,stats]=ttest2(v1,v2); t=stats.tstat; df=stats.df; else; p=NaN; t=NaN; df=NaN; end
end
