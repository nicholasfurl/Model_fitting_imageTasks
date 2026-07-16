%seq length 8. Beta levels up to 26.
%out_PR_120models5modelTypes4paramLevels6betaLevels10subs5seqs8opts_hybrid_newPrior_20232312.mat
% out_PR_400models5modelTypes8paramLevels10betaLevels10subs5seqs8opts_hybrid_newPrior_20232712
%This one does beta [1 10 20] and 428 phase1. But the BV 1 to 100 bound is fixed.
out_PR_60models5modelTypes4paramLevels3betaLevels10subs10seqs8opts_hybrid_20240301

% %seq length 12. Beta levels up to 26. Low values of BV produce a few crazy
% %fits, and small N / param levels. 
% out_PR_120models5modelTypes4paramLevels6betaLevels10subs5seqs12opts_hybrid_newPrior_20232412.mat
% %makes the above obsolete. The same but works up to beta = 61
% out_PR_140models5modelTypes4paramLevels7betaLevels10subs5seqs12opts_hybrid_newPrior_20232412.mat
%This is really good, except one stray point makes opt .76. Can I fix just that?
% out_PR_200models5modelTypes5paramLevels8betaLevels10subs5seqs12opts_hybrid_newPrior_20232512
out_PR_220models5modelTypes4paramLevels11betaLevels10subs5seqs12opts_hybrid_20233112    %bv bound not quite right [-100 100] but else perfect

%seq length 10. This one does beta [1 10 20] and 428 phase1. But the BV 1 to 100 bound is fixed.
out_PR_60models5modelTypes4paramLevels3betaLevels10subs10seqs10opts_hybrid_20240301