
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = fit_models_food_2_v2;

tic

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\FMINSEARCHBND'))

Generate_params.do_models = [1 2 6];    %model identifiers: 1 cutoff, 2 Cs, 3 obsolete placeholder, 4 BV, 5 BR, 6 BPM, 7 Opt, 8 BPV
Generate_params.comment = 'out_imageTask_food_2_COCSBMP_';     %The filename will already fill in basic parameters so only use special info for this.

    disp('Getting subject data ...');
    num_subs_found = 0;
    
    [mean_ratings_all seq_vals_all output_all] = get_sub_data;
    
    subjects = 1:size(mean_ratings_all,2);
    
    for subject = subjects;
        
        mean_ratings = mean_ratings_all(:,subject);
        seq_vals = seq_vals_all(:,:,subject);
        output = output_all(:,subject);
        
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
        
    end;    %subject loop
    

Generate_params = imageTask_run_models(Generate_params);

Generate_params = run_io(Generate_params);

save([Generate_params.outpath filesep Generate_params.outname], 'Generate_params');

disp('audi5000')

toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mean_ratings seq_vals output] = get_sub_data;

%In this version, we'll extract all the subject data here before we embark
%on the subject loop in the main body. That means another subject loop
%inside of here, unfortunately.

%sub num then data (rating) for holidays then foods then female faces than male faces
temp_r = xlsread('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\food_study_2_openday\data\all_attractiveness_ratings_noheader_openday.xlsx');
%There are some nans hanging on the end of domains with less than the max num of subjects. Get rid of them nice are early (here)
data = temp_r(~isnan(temp_r(:,3)),3:4);
%sub num then data (draws) for holidays then foods then female faces than male faces
[temp_d dummy dummy] = xlsread('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\food_study_2_openday\data\all_sequences_openday.xlsx');
draws = temp_d(~isnan(temp_d(:,3)),3:4);
%These cols are 1:domain code (1=holidays, 2=food, 3=female, 4=male), 2:event num, 3: sub_id, 4-11: the 8 filenames for this sequence
[dummy dummy temp_f] = xlsread('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\food_study_2_openday\data\all_domains_openday_sequence_fnames.xlsx');
seq_fnames = temp_f(find(cell2mat(temp_f(:,1)) == 2),:);
%filename index key (cols: holiday, food, male, female) 90 filenames key to ratings
[dummy1 dummy2 key_ratings] = xlsread('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\food_study_2_openday\data\domains_key_ratings_fnames.xlsx');
key_this_domain = key_ratings(:,2);

num_seq_pos = 8;
these_sub_ids = unique( cell2mat(seq_fnames(:,3)) );

for sub_id = 1:numel(these_sub_ids);
    
    %We're going to replace seq filenames with ratings. Ready the sequence filenames
    this_sub_indices = find( cell2mat(seq_fnames(:,3)) == these_sub_ids(sub_id) );  %index into one of the subs
    this_sub_data = seq_fnames(this_sub_indices,4:11);  %filenames for sequences for this sub
    this_sub_num_seqs = numel(this_sub_indices);    %Should always be 6 sequences
    if exist('key_this_domain_f');  %if rating filenames were divided into male and female earlier
        sexes_this_subject = female_faces(this_sub_indices,1);
        if sexes_this_subject(1) == 1;
            key_this_domain = key_this_domain_f;
        else
            key_this_domain = key_this_domain_m;
        end;
        
    end;    %check if i need to assign sex specific face key
    
    %We're going to replace seq filenames with ratings. Ready the ratings
    this_sub_ratings_indices = find(data(:,1)==these_sub_ids(sub_id)); %This should find all the ratings for this subject in the same domain as the sequences. Should be 90
    this_sub_ratings = data(this_sub_ratings_indices,2);   %ratings themselves for this subject, should be 90. Should be sorted by key
    
    mean_ratings(:,sub_id) = this_sub_ratings;  %first output of this function
    
    %Get number of samples for this subject's sequence (Will use in sequence loop to get ranks for each sequence)
    clear this_sub_draws;
    this_sub_draws = draws(find(draws(:,1)==these_sub_ids(sub_id)),2);    %Should return the 5/6 sequence indices; subids in draws file better match up with subids in sequience lists (they should)
    
    %next function output
    output(:,sub_id) = this_sub_draws;
    
    for seq_num = 1:this_sub_num_seqs;
        for seq_pos = 1:num_seq_pos;
            
            filename_to_find = seq_fnames(this_sub_indices(seq_num),seq_pos+3); %What is this filename?
            idx = find(strcmp(key_this_domain, filename_to_find));  %find the index corresponding to the sequence position's filename
            this_seq_pos_rating = this_sub_ratings(idx);    %get rating corresponding to the index for this filename
            
            %next main output
            seq_vals(seq_num,seq_pos,sub_id) = this_seq_pos_rating;
            
        end;    %loop through sequence positions
    end;    %loop through sequences
end;   %loop through subs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

