
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = fit_models_face_1_v2;

%av face study model fits. Needs imageTask_run_models.m nd run_io.m to work. Needs
%imageTask_figures_CP_revisions_v2.m to view results in figures.

tic

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\FMINSEARCHBND'))

Generate_params.do_models = [1 2 6];    %model identifiers: 1 cutoff, 2 Cs, 3 obsolete placeholder, 4 BV, 5 BR, 6 BPM, 7 Opt, 8 BPV
Generate_params.comment = 'out_imageTask_face_1_COCSBMP_';     %The filename will already fill in basic parameters so only use special info for this.

subjects = 1:20;

disp('Getting subject data ...');
num_subs_found = 0;
for subject = subjects;

    %sub_fail 1*1, female 1*1, mean_ratings num_stim*1, seq_vals seqs*opts, output seqs*1
    %Note that ratings data are scaled 1 to 100 by this function.
    [sub_fail female mean_ratings seq_vals output] = get_sub_data_matlab(subject);

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

end;    %subjects


Generate_params = imageTask_run_models(Generate_params);

Generate_params = run_io(Generate_params);

save([Generate_params.outpath filesep Generate_params.outname], 'Generate_params');

disp('audi5000')

toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sub_fail rated_female mean_ratings seq_vals draws] = get_sub_data_matlab(subject);

%sub_fail indicates whether full data could be extracted from this
%subject or whether subject should be skipped in outer loop that
%calls get_sub_data. Can be used to decide whether to increment
%num_subs_found if you outout 1 for success and 0 for fail;

data_folder = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\face_study_1_av\data';
rating_files = dir( [data_folder filesep sprintf('av_ratings_sub%02d*.mat',subject)] );
seq_files = dir( [data_folder filesep sprintf('aav_sequence_sub%02d*.mat',subject)] );

%In case I need to break
sub_fail = 0;
rated_female = [];
mean_ratings = [];
seq_vals = [];
output = [];
draws = [];


num_rating_files = [];
if ~isempty(rating_files);  %is there a subject?
    num_rating_files = size( rating_files, 1); %how many ratings (there's usually three)?
else
    disp(sprintf('there is no subject %d',subject));
    return;   %skip to next sub if this sub has no files
end;

%rated_female is 1 if female face, 0 if male
if ~isempty( strfind(rating_files(1).name,'sex01') );
    rated_female = 1;
elseif ~isempty( strfind(rating_files(1).name,'sex00') );
    rated_female = 0;
end;

%get and store ratings data
for file = 1:num_rating_files;

    clear output;   %just in case
    load( [data_folder filesep rating_files(file).name],'output'); %columns: trial_num, filename, face_num, rating
    output = sortrows(output,3); %sort output by face_num

    %correct cogent codes
    ratings( find(cell2mat(output(:,4)) > 70),file) = cell2mat( output(find(cell2mat(output(:,4)) > 70),4) ) - 75;  %if number pad
    ratings( find(cell2mat(output(:,4)) < 70),file) = cell2mat( output(find(cell2mat(output(:,4)) < 70),4) ) - 27;  %if top of keyboard numbers

    %NaN other weird values
    if numel(find( ratings(:,file) < 1 ))>0 | numel(find( ratings(:,file) > 9 ))>0
        ratings( find( ratings(:,file) < 1 ), file ) = NaN;    %in case subject presses wrong key, log will choke later
        ratings( find( ratings(:,file) > 9 ), file ) = NaN;    %in case subject presses wrong key
    end;    %if any residual weird keypresses

end;    %loop through ratings files

mean_ratings = nanmean( ratings' )';    %raw ratings
%new to this version
%scale them to be between 1 and 100 (so all studies will be on the same scale)
% min_ratings = min(mean_ratings);   %what is new min after trandform?
% max_ratings = max(mean_ratings);   %what is new max after transform?
% mean_ratings = (mean_ratings - min_ratings)/(max_ratings - min_ratings);
old_min = 1;
old_max = 9;
new_min = 1;
new_max = 100;
mean_ratings = ((new_max-new_min)/(old_max-old_min))*(mean_ratings - old_max)+100;

%now load sequence data for this subject
load( [data_folder filesep seq_files.name], 'rows_data','output' );  %get the list of face numbers in the sequences

%Get numbers of draws
draws = cell2mat(output( find( cell2mat(output(:,6))==29 ), 3 ));  %draws

%need the list of values for each facenumber
seq_vals = zeros(size(rows_data));
for i=1:numel(rows_data);
    seq_vals(i) = mean_ratings( rows_data(i) );     %raw values
end;

sub_fail = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

