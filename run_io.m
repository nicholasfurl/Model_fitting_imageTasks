%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Generate_params = run_io(Generate_params);

for sub = 1:Generate_params.num_subs;

    disp(...
        sprintf('computing performance, ideal observer subject %d' ...
        , sub ...
        ) );

    for sequence = 1:Generate_params.num_seqs;

        clear sub_data;

        prior.mu =  nanmean(log(Generate_params.ratings(:,sub)));
        prior.sig = nanvar(log(Generate_params.ratings(:,sub)));
        prior.kappa = 2;
        prior.nu = 1;

        sampleSeries = log(Generate_params.seq_vals(sequence,:,sub));

        [choiceStop, choiceCont, difVal] = ...
            computeSecretary(prior,sampleSeries);

        samples(sequence,sub) = find(difVal<0,1,'first');

        %rank of chosen option
        dataList = tiedrank(squeeze(Generate_params.seq_vals(sequence,:,sub))');    %ranks of sequence values
        ranks(sequence,sub) = dataList(samples(sequence,sub));

    end;    %sequence loop

end;    %sub loop

%add new io field to output struct
num_existing_models = size(Generate_params.model,2);
Generate_params.model(num_existing_models+1).name = 'Optimal';
Generate_params.model(num_existing_models+1).num_samples_est = samples;
Generate_params.model(num_existing_models+1).ranks_est = ranks;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [choiceStop, choiceCont, difVal] = computeSecretary(priorProb, sampleSeries)

N = length(sampleSeries);
Cs = 0;

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
    
    [expectedStop, expectedCont] = rnkBackWardInduction(sampleSeries, ts, priorProb, N, x, Cs);    
    
    [rnkv, rnki] = sort(sampleSeries(1:ts), 'descend');
    z = find(rnki == ts);
            
    difVal(ts) = expectedCont(ts) - expectedStop(ts);
        
    choiceCont(ts) = expectedCont(ts);
    choiceStop(ts) = expectedStop(ts);
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [expectedStop, expectedCont, expectedUtility] = rnkBackWardInduction(sampleSeries, ts, priorProb, N, x, Cs)

Nx = length(x);

payoff = sort(sampleSeries,'descend');

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
rnki = z;

ties = 0;
if length(unique(sampleSeries(1:ts))) < ts
    ties = 1;
end

mxv = ts;
if mxv > maxPayRank
    mxv = maxPayRank;
end

rnkv = [Inf*ones(1,1); rnkvl(1:mxv)'; -Inf*ones(20, 1)];

[postProb] = normInvChi(priorProb, data);
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
            zi = length(utStop) - 1;
        end

        utStop = utStop(zi+1)*ones(Nx, 1);
        
    end
    
    utCont = utCont - Cs;
    
    utility(:, ti)      = max([utStop utCont], [], 2);
    expectedUtility(ti) = px'*utility(:,ti);
    
    expectedStop(ti)    = px'*utStop;
    expectedCont(ti)    = px'*utCont;
                 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function prob_y = posteriorPredictive(y, postProb)

tvar = (1 + postProb.kappa)*postProb.sig/postProb.kappa;

sy = (y - postProb.mu)./sqrt(tvar);

prob_y = tpdf(sy, postProb.nu);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


