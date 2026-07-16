function [choiceStop, choiceCont, difVal, currentRnk, winnings] = analyzeSecretary(dataPrior, list, params, distOptions,minValue)

if list.flip == -1
    sampleSeries = -(dataList - mean(dataList)) + mean(dataList);
else
    %     sampleSeries = dataList;
    sampleSeries = list.vals;
end

N = list.length;

prior = dataPrior;

%%% if not using ranks
%%% Cs = params(1)*prior.mu;

Cs = params(1);

%%% First time probLarger goes under 0.5, subject should stop
% probLarger = forwardInduction(sampleSeries, prior);

[choiceStop, choiceCont, difVal, currentRnk] = computeSecretary(params, sampleSeries, prior, N, list, Cs, distOptions,minValue);

if list.optimize == 1
    z = find(difVal < 0);
    [~, rnki] = sort(sampleSeries, 'descend');
    rnkValue = find(rnki == z(1));

    winnings = (rnkValue == 1)*5 + (rnkValue == 2)*2 + (rnkValue == 3)*1;
else
    winnings = 0;
    rnkValue = -1*ones(length(list.vals), 1);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [choiceStop, choiceCont, difVal, currentRnk] = computeSecretary(params, sampleSeries, priorProb, N, list, Cs, distOptions,minValue)

sdevs = 8;
dx = 2*sdevs*sqrt(priorProb.sig)/100;
x = ((priorProb.mu - sdevs*sqrt(priorProb.sig)) + dx : dx : ...
     (priorProb.mu + sdevs*sqrt(priorProb.sig)))';
 
Nchoices = length(list.vals);

if list.optimize == 1
    Nconsider = length(list.allVals);
else
    Nconsider = length(sampleSeries);
    if Nconsider > N
        Nconsider = N;
    end
end

difVal = zeros(1, Nconsider);
choiceCont = zeros(1, Nconsider);
choiceStop = zeros(1, Nconsider);
currentRnk = zeros(1, Nconsider);

for ts = 1 : Nconsider
    
    [expectedStop, expectedCont] = rnkBackWardInduction(sampleSeries, ts, priorProb, N, x, Cs, distOptions,minValue);    
%     [expectedStop, expectedCont] = backWardInduction(sampleSeries, ts, priorProb, x, Cs);    
    
    [rnkv, rnki] = sort(sampleSeries(1:ts), 'descend');
    z = find(rnki == ts);
            
    difVal(ts) = expectedCont(ts) - expectedStop(ts);
        
    choiceCont(ts) = expectedCont(ts);
    choiceStop(ts) = expectedStop(ts);
    
    currentRnk(ts) = z;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [expectedStop, expectedCont, expectedUtility] = backWardInduction(sampleSeries, ts, priorProb, x, Cs)

N = length(sampleSeries);

data.n  = ts;

% if ts > 0
    data.sig = var(sampleSeries(1:ts));
    data.mu = mean(sampleSeries(1:ts));
% else
%     data.sig = priorProb.sig;
%     data.mu  = priorProb.mu;
% end

utStop  = x;
    
utCont = zeros(N, 1);
utility = zeros(length(x), N);

if ts == 0
    ts = 1;
end

for ti = N : -1 : ts
    
    expData = data;
    expData.n = ti;
    
    [postProb] = normInvChi(priorProb, expData);
    
    px = posteriorPredictive(x, postProb);
    
    px = px/sum(px);
       
    if ti == N
        utCont(ti) = -Inf;
    else
        utCont(ti) = sum(px.*utility(:, ti+1)) - Cs;
    end
                        
    if ti == ts
        utility(:, ti)   = ones(size(utStop, 1), 1)*max([sampleSeries(ts) utCont(ti)]);        
        expectedStop(ti) = sampleSeries(ts);
    else
        utility(:, ti)   = max([utStop ones(size(utStop, 1), 1)*utCont(ti)], [], 2);
        expectedStop(ti) = sum(px.*utStop);
    end
    
    expectedUtility(ti) = sum(px.*utility(:,ti));
    expectedCont(ti)    = utCont(ti);
                 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [expectedStop, expectedCont, expectedUtility] = rnkBackWardInduction(sampleSeries, ts, priorProb, ...
    listLength, x, Cs, distOptions,minValue)


N = listLength;
Nx = length(x);

% payoff = sort(sampleSeries,'descend')';
% payoff = [N:-1:1];
% payoff = (payoff-1)/(N-1);




% % %bins
% temp = sort(sampleSeries,'descend')';
% [dummy,payoff] = histc(temp, [minValue(1:end-1) Inf]);
% nbins = size(minValue,2)-1;
% payoff = (payoff-1)/(nbins-1);

% %normalised rating value
% payoff = sort(sampleSeries,'descend'); %assign actual values to payoff
% payoff = (payoff-0)/(minValue(end) - 0);    %normalise seq values between zero and 1 relative to maximally rated face




% payoff(find(payoff~=8))=0.0000000000000000000000000000000001;
% payoff(find(payoff==8))=100000000000000000000000000000;
%     
% %bound values between zero and 1
% if numel(minValue) > 2;
% payoff = ((payoff-0)/((numel(minValue)-1)-0));
% end;
% payoff = payoff.^40;

payoff = [.12 .08 .04]; 


maxPayRank = numel(payoff);
payoff = [payoff zeros(1, 20)];



data.n  = ts;

if distOptions == 0
    data.sig = var(sampleSeries(1:ts));
    data.mu = mean(sampleSeries(1:ts));
    
else
    data.mu = mean(sampleSeries(1:ts));
    data.sig = data.mu;
end
               
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
