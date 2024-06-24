function [choiceStop, choiceCont, difVal] = analyzeSecretaryPR(Generate_params,sampleSeries);

%2024. Was analyseSecretaryNick_2021. Updated to analyseSecretaryPR
%2/Feb/2024. Updated BV to zero below threshold values. Cleaned up a bit in
%prep for GitHub / sharing. Cleaned up sampleSeries / list.allVals. Got rid
%of minValue stuff.

%2021:updated for use with Param_rover_v1.m
%Written assuming it operating on exactly onse sequence at a time


N = Generate_params.seq_length;
Cs = Generate_params.model(Generate_params.current_model).Cs;      %Will already be set to zero unless Cs model


%Assign params. Hopefully everything about model is now pre-specified
%(I would update mean and variance directly from Generate_params 
prior.mu    = Generate_params.PriorMean + Generate_params.model(Generate_params.current_model).BP; %prior mean offset is zero unless biased prior model
prior.sig   = Generate_params.PriorVar + Generate_params.model(Generate_params.current_model).BPV;                        %Would a biased variance model be a distinct model?
if prior.sig < 1; prior.sig = 1; end;   %It can happen randomly that a subject has a low variance and subtracting the bias gives a negative variance. Here, variance is set to minimal possible value.
prior.kappa = Generate_params.model(Generate_params.current_model).kappa;   %prior mean update parameter
prior.nu    = Generate_params.model(Generate_params.current_model).nu;

%BV
if Generate_params.model(Generate_params.current_model).identifier == 4;   %If identifier is BV

    sampleSeries( find(sampleSeries <= Generate_params.model(Generate_params.current_model).BVmid)) = 0;

end;

[choiceStop, choiceCont, difVal, currentRnk] = computeSecretary(Generate_params, sampleSeries, prior, N, Cs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%










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

    [expectedStop, expectedCont] = rnkBackWardInduction(sampleSeries, ts, priorProb, N, x, Cs,  Generate_params);

    [rnkv, rnki] = sort(sampleSeries(1:ts), 'descend');
    z = find(rnki == ts);
        
    difVal(ts) = expectedCont(ts) - expectedStop(ts);
    
    choiceCont(ts) = expectedCont(ts);
    choiceStop(ts) = expectedStop(ts);
    
    currentRnk(ts) = z;
    
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [expectedStop, expectedCont, expectedUtility] = rnkBackWardInduction(sampleSeries, ts, priorProb, listLength, x, Cs, Generate_params)

N = listLength;
Nx = length(x);

%BR
if Generate_params.model(Generate_params.current_model).identifier == 5;

    %     rewards = ...
    %         (Generate_params.BVrange(2) - Generate_params.BVrange(1)) ./ ...
    %         (1+exp(-Generate_params.model(Generate_params.current_model).BRslope*(sampleSeries-Generate_params.model(Generate_params.current_model).BRmid))); %do logistic transform
    %
    %     payoff = sort(rewards,'descend');
    %     payoff = (payoff-0)/(Generate_params.BVrange(2)-0);  %normalises bin values to 0 to 1 scale (for some reason, presumably so I can compare models with different reward functions by placing them on same scale, although I'm not ure scale matters. Will keep for now)

    sampleSeries( find(sampleSeries <= Generate_params.model(Generate_params.current_model).BRmid)) = 0;

end;

payoff = sort(sampleSeries,'descend');   %sort the sample values
% payoff = (temp-1)/(nbins-1);  %normalises bin values to 0 to 1 scale (for some reason, presumable so I can compare models with different reward functions by placing them on same scale, although I'm not ure scale matters. Will keep for now)

maxPayRank = numel(payoff);
temp = [payoff zeros(1, 1000)];
payoff = temp;
% 
% maxPayRank = numel(payoff);
% payoff = [payoff zeros(1, 20)];

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

postProb.mu ...
    = postProb.mu ...
    + Generate_params.model(Generate_params.current_model).optimism; %...Then add constant to the posterior mean (will be zero if not optimism model)

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
            %             fprintf('accessing utStop at %d value x %.2f\n', zi, x);
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
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








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
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function prob_y = posteriorPredictive(y, postProb)

tvar = (1 + postProb.kappa)*postProb.sig/postProb.kappa;

sy = (y - postProb.mu)./sqrt(tvar);

prob_y = tpdf(sy, postProb.nu);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






