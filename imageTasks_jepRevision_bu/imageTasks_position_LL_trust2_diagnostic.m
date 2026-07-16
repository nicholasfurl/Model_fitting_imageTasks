clear; clc;

outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\outputs';
fit_file = [outpath filesep 'out_imageTask_trust_2_COCSBMP2_20250103.mat'];
pos_file = [outpath filesep 'position_ll_trust2_out' filesep 'trust2_position_likelihood_results.mat'];

S = load(fit_file, 'Generate_params');
R = load(pos_file);

GP = S.Generate_params;

model_indices = [2 1 3]; % Cost to sample, Cut off, Biased prior
model_names = {'Cost to sample','Cut off','Biased prior'};

% Whole-sequence participant-level winner, using saved NLL.
% Equal k across these models, so this is also the BIC winner.
NLL = [ ...
    GP.model(model_indices(1)).ll(:), ...
    GP.model(model_indices(2)).ll(:), ...
    GP.model(model_indices(3)).ll(:) ...
    ];

[~, whole_seq_winner] = min(NLL, [], 2);

% Decision-reached mask from the position-wise likelihood output.
% seq x position x participant
reached = isfinite(R.all_nll_decision(:,:,:,1));

num_seqs = size(reached,1);
seq_length = size(reached,2);
num_subs = size(reached,3);
num_models = numel(model_names);

% Observed stopping positions: seq x participant.
obs_samples = GP.num_samples;

% ------------------------------------------------------------
% Position-wise composition table
% ------------------------------------------------------------
N_decisions = NaN(seq_length,1);
N_participants = NaN(seq_length,1);
Stop_rate = NaN(seq_length,1);
Mean_value = NaN(seq_length,1);
Prop_best_so_far = NaN(seq_length,1);

Ndec_by_winner = NaN(seq_length,num_models);
Prop_dec_by_winner = NaN(seq_length,num_models);
Nsubs_by_winner = NaN(seq_length,num_models);

for pos = 1:seq_length

    pos_reached = squeeze(reached(:,pos,:)); % seq x sub

    N_decisions(pos) = sum(pos_reached(:));
    N_participants(pos) = sum(any(pos_reached,1));

    % Stop rate among decisions reached at this position.
    stop_here = (obs_samples == pos);
    Stop_rate(pos) = sum(stop_here(pos_reached)) ./ N_decisions(pos);

    % Mean value among reached trials at this position.
    vals_here = squeeze(GP.seq_vals(:,pos,:));
    Mean_value(pos) = mean(vals_here(pos_reached), 'omitnan');

    % Is current item best so far?
    best_so_far = false(num_seqs, num_subs);
    for seq = 1:num_seqs
        for sub = 1:num_subs
            best_so_far(seq,sub) = GP.seq_vals(seq,pos,sub) == max(GP.seq_vals(seq,1:pos,sub));
        end
    end
    Prop_best_so_far(pos) = mean(best_so_far(pos_reached), 'omitnan');

    % Participant composition by whole-sequence winning model.
    for g = 1:num_models
        these_subs = find(whole_seq_winner == g);
        Ndec_by_winner(pos,g) = sum(sum(pos_reached(:,these_subs)));
        Prop_dec_by_winner(pos,g) = Ndec_by_winner(pos,g) ./ N_decisions(pos);
        Nsubs_by_winner(pos,g) = sum(any(pos_reached(:,these_subs),1));
    end
end

Tpos = table( ...
    (1:seq_length)', ...
    N_decisions, ...
    N_participants, ...
    Stop_rate, ...
    Mean_value, ...
    Prop_best_so_far, ...
    Ndec_by_winner(:,1), Prop_dec_by_winner(:,1), ...
    Ndec_by_winner(:,2), Prop_dec_by_winner(:,2), ...
    Ndec_by_winner(:,3), Prop_dec_by_winner(:,3), ...
    'VariableNames', { ...
    'Position','N_decisions','N_participants','Stop_rate','Mean_value','Prop_best_so_far', ...
    'N_decisions_Cs_winners','Prop_decisions_Cs_winners', ...
    'N_decisions_Cutoff_winners','Prop_decisions_Cutoff_winners', ...
    'N_decisions_BP_winners','Prop_decisions_BP_winners'} ...
    );

disp(Tpos);

writetable(Tpos, [outpath filesep 'position_ll_trust2_out' filesep 'trust2_position_composition_by_position.csv']);

% ------------------------------------------------------------
% Compare positions where Cs versus BP was best in the position-wise plot
% Use positions 2-4 for Cs and 5-7 for late BP. Exclude position 8 because terminal.
% ------------------------------------------------------------
phase_names = {'Cs-best positions 2-4','Late BP-best positions 5-7'};
phase_positions = {2:4, 5:7};

for ph = 1:numel(phase_positions)

    positions = phase_positions{ph};

    phase_reached = false(num_seqs, num_subs);
    total_decisions_by_sub = zeros(num_subs,1);

    for pos = positions
        pos_reached = squeeze(reached(:,pos,:));
        phase_reached = phase_reached | pos_reached;
        total_decisions_by_sub = total_decisions_by_sub + sum(pos_reached,1)';
    end

    fprintf('\n%s\n', phase_names{ph});
    fprintf('  Total decisions = %d\n', sum(total_decisions_by_sub));
    fprintf('  Participants contributing at least one decision = %d\n', sum(total_decisions_by_sub > 0));

    for g = 1:num_models
        these_subs = whole_seq_winner == g;
        n_dec = sum(total_decisions_by_sub(these_subs));
        n_sub = sum(total_decisions_by_sub(these_subs) > 0);
        fprintf('  %-15s whole-sequence winners: %3d decisions, %2d participants, %.1f%% of decisions\n', ...
            model_names{g}, n_dec, n_sub, 100*n_dec/sum(total_decisions_by_sub));
    end
end

% ------------------------------------------------------------
% Optional: stratified position-wise fit by whole-sequence winner group
% This asks whether BP wins late only because BP-winner participants dominate late positions.
% ------------------------------------------------------------
fprintf('\nBest position-wise model within each whole-sequence winner group\n');

for g = 1:num_models

    these_subs = find(whole_seq_winner == g);
    fprintf('\nParticipants whose whole-sequence winner was %s (n = %d)\n', model_names{g}, numel(these_subs));

    mean_nll_group = NaN(seq_length,num_models);

    for pos = 1:seq_length
        for mi = 1:num_models
            vals = R.all_nll_decision(:,pos,these_subs,mi);
            vals = vals(isfinite(vals));
            if ~isempty(vals)
                mean_nll_group(pos,mi) = mean(vals);
            end
        end
    end

    [~, best_group] = min(mean_nll_group, [], 2);

    for pos = 1:seq_length
        fprintf('  Position %d: %s\n', pos, model_names{best_group(pos)});
    end
end