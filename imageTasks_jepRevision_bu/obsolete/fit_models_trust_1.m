
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = fit_models_trust_1_v2

%small trust study model fits. Needs imageTask_run_models.m to work. Needs
%imageTask_figures_CP_revisions_v2 to view results in figures.

tic

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\FMINSEARCHBND'))

Generate_params.do_models = [1 2 7 4];    %model identifiers: 1 cutoff, 2 Cs, 3 obsolete placeholder, 4 BV, 5 BR, 6 BPM, 7 Opt, 8 BPV
Generate_params.comment = 'out_imageTask_trust_1_';     %The filename will already fill in basic parameters so only use special info for this.

subjects = [101:116 118 119];    %small trust sub nums

disp('Getting subject data ...');
num_subs_found = 0;
for subject = subjects;

    %sub_fail 1*1, female 1*1, mean_ratings 1*num_stim, seq_vals seqs*opts, output seqs*1
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

    Generate_params.ratings(:,num_subs_found) = mean_ratings;
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
function [sub_fail rated_female mean_ratings seq_vals draws] = get_sub_data_matlab(subject);

%sub_fail indicates whether full data could be extracted from this
%subject or whether subject should be skipped in outer loop that
%calls get_sub_data. Can be used to decide whether to increment
%num_subs_found if you outout 1 for success and 0 for fail;

data_folder = 'C:\matlab_files\fiance\trustworthiness';
rating_files = dir( [data_folder filesep sprintf('av_ratings_sub%03d*.mat',subject)] );
seq_files = dir( [data_folder filesep sprintf('av_sequence_sub%03d*.mat',subject)] );

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

%For some reason, some participants have ratings in match and small trust have 425 entries
%and others have 426. If 425, I'm going to add a NaN for number 426 so they
%can be put together into one matrix. But the face numbers for these two types of
%participant don't match the filename lists (so face 100 in a 425 ratinmg
%participant is not face 100 in a 426 rating participant). So these
%matrices should not be averaged or used to plot distributions unless they
%are redone so they are sorted by filename instead of filenumber. It
%should be ok to use the filenumbers as lookup tables within the same
%participants as is done in the lines above though.
if numel(mean_ratings) == 425;
    disp(sprintf('Subject file %s has 425 not 426 ratings',[data_folder filesep rating_files(1).name]));
    mean_ratings = [mean_ratings; NaN];
end;

sub_fail = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%










%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

        list.flip = 0;
        list.vals = log(Generate_params.seq_vals(sequence,:,sub));
        list.length = size(list.vals,2);
        list.optimize = 0;
        params = 0; %Cs
        [choiceStop, choiceCont, difVal] = ...
            analyzeSecretaryNick3_test(prior,list,0,0,0);

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

fprintf('');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
