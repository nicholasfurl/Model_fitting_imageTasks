
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = fit_models_trust_2_v2;

%big trust study model fits. Needs imageTask_run_models.m to work. Needs
%imageTask_figures_CP_revisions_v2 to view results in figures.

tic

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\FMINSEARCHBND'))

Generate_params.do_models = [1 2 6];    %model identifiers: 1 cutoff, 2 Cs, 3 obsolete placeholder, 4 BV, 5 BR, 6 BPM, 7 Opt, 8 BPV
Generate_params.comment = 'out_imageTask_trust_2_COCSBMP_';     %The filename will already fill in basic parameters so only use special info for this.

subjects = 1:64;    %big trust sub nums

disp('Getting subject data ...');
num_subs_found = 0;
for subject = subjects;

    %sub_fail 1*1, female 1*1, mean_ratings 1*num_stim, seq_vals seqs*opts, output seqs*1
    [sub_fail female mean_ratings seq_vals output] = get_sub_data_gorilla(subject);

    if sub_fail == 0;
        continue;
    end;
    num_subs_found = num_subs_found + 1;

    %Get ranks
    clear seq_ranks ranks;
    seq_ranks = tiedrank(seq_vals')';
    for i=1:size(seq_ranks,1);
        ranks(i,1) = seq_ranks(i,output(i,1));
    end;    %loop through sequences to get rank for each

    Generate_params.ratings(:,num_subs_found) = mean_ratings';
    Generate_params.seq_vals(:,:,num_subs_found) = seq_vals;

    Generate_params.num_samples(:,num_subs_found) = output;
    Generate_params.ranks(:,num_subs_found) = ranks;

end;    %ends subject loop


Generate_params = imageTask_run_models(Generate_params);

Generate_params = run_io(Generate_params);

save([Generate_params.outpath filesep Generate_params.outname], 'Generate_params');

disp('audi5000')

toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%









%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sub_fail rated_female mean_ratings seq_vals output] = get_sub_data_gorilla(subject);

%col1: participant numbers who rated male faces
%col2: ratings of  faces, sorted by filename (as in big_trust_key_fnames.xlsx) then by participant number
%col3: filenames of faces
[dummy1 dummy2 data{1}] = xlsread('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\trust_study_2_big\all_big_trust_ratings_noaverage.xlsx');
%     %col1: list of rated male filenames, sorted same as in data{1}
%     %col2: list of rated female filenames, sorted same as in data{1}
%     [dummy1 dummy2 key_ratings] = xlsread('C:\matlab_files\fiance\big_trust\big_trust_key_fnames.xlsx');
%col1: participant numbers for male sequences (rows)
%col2: numbers of draws for male sequences (rows), sorted by chronological sequence order
%col3: participant numbers for female sequences (rows)
%col4: numbers of draws for female sequences (rows), sorted by chronological sequence order
[draws{1} dummy dummy] = xlsread('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\trust_study_2_big\all_sequences_big_trust.xlsx');
%col1: 1 for male faces and 2 for female faces
%col2: event number, we don't need it
%col3: participant number
%cols4-11: filenames for sequences positions 1-8
[dummy dummy seq_fnames{1}] = xlsread('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\trust_study_2_big\all_big_trust_sequence_fnames.xlsx');

these_sub_ids = unique( cell2mat(seq_fnames{1}(:,3)) );
sub_id = subject;

%ready the sequence filenames for search
this_sub_indices = find( cell2mat(seq_fnames{1}(:,3)) == these_sub_ids(sub_id) );   %get indices rows of sequence fname data for this one subject
this_sub_data = seq_fnames{1}(this_sub_indices,4:11);   %get rows for this sub using those indices
this_sub_domain = cell2mat(seq_fnames{1}(this_sub_indices(1),1)); %Is it male (1) or female (2) - just check first row, all rows should be same for one subject
this_sub_num_seqs = numel(this_sub_indices);    %Should always be 40 sequences
num_seq_pos = size(this_sub_data,2);    %should also be 8
%******Returned variable*******
if this_sub_domain == 2;
    rated_female = 1;
else
    rated_female = 0;
end;

%ready the ratings data for search
this_sub_ratings_indices = find(cell2mat(data{1}(:,1))==these_sub_ids(sub_id)); %get indices for this one sub into the list of ratings in the appropriate column for male of female faces (domain)
this_sub_ratings = cell2mat(data{1}(this_sub_ratings_indices,2)); %Get the ratings that go with this participant's indices from that correct col

%Now loop through the filenames for this subject and average the
%ratings for each one. For some reason, some have two and some have
%three (That I know of so far)
this_sub_rated_files = data{1}(this_sub_ratings_indices,3);
this_sub_rated_fnames = unique( this_sub_rated_files );
for file = 1:size(this_sub_rated_fnames,1);
    %******Returned variable*******
    mean_ratings(file) = mean(this_sub_ratings(find( strcmp( this_sub_rated_files, this_sub_rated_fnames{file} ) )));
end;    %loop through files

%     mean_ratings = this_sub_ratings;    %I did this averaging in Excel when making the data files so we can just assign this directly to the output vector mean_ratings

%get this sub's numbers of draws for each sequence, look in column appropriate for this subject's face sex
%**************returned variable**************
output = draws{1}(find(draws{1}(:,1)==these_sub_ids(sub_id)),2);    %Should return the 5/6 sequence indices; subids in draws file better match up with subids in sequience lists (they should)

%Now do search; replace sequence filenames with their corresponding ratings
for seq_num = 1:this_sub_num_seqs;
    for seq_pos = 1:num_seq_pos;

        filename_to_find = seq_fnames{1}(this_sub_indices(seq_num),seq_pos+3); %What is this filename?
        idx = find(strcmp(this_sub_rated_fnames, filename_to_find));  %find the index corresponding to the sequence position's filename
        this_seq_pos_rating = mean_ratings(idx);    %get rating corresponding to the index for this filename
        %************returned variable**********
        seq_vals(seq_num,seq_pos) = this_seq_pos_rating;   %populate temporary matrix for this subject to be attached to seq_data cell array when this subject is finished

    end;    %loop through sequence positions
end;    %Loop through sequences

%************returned variable**********
sub_fail = 1;   %1 is success for a variable called, fail. Go figure;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
