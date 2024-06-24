
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = new_fit_code_make_plots_2022_v3;

%_2022_v3. Adds ideal observer sampling rate to sampling plots. You should
%use _2022_v2 if you have an estimated datafile that hasn't had optimal /
%ideal observer estimated.

%_2022_v2 I'm trying to change the order of the bars / models in plots and
%pave the way for inserting ideal observer. I think this is likely to make
%a complete mess of the existing code but so it goes.

%New fit code make plots_2022. I'm seeing if I can't excise the model
%fitting code (can post that separately) and pare down to just the plotting
%for paper figures. Maybe I can even incorporate multiple datasets into one
%programme. We'll see.

%new_fit_code_av_2022: Creates results for av study model fitting. 2022
%revamps it slightly to make plots more figure-worthy.

%v2: I tried to introduce log_or_not functionality

% new_fit_code_big_trust: I am now modifying my sweep code to fit
% parameters to human participants data with the biggest dataset that we
% have. Big trust.

%v3_sweep isn't an improvemnt on v2 buit starts a new branch. V2 (and its
%successors if any) will continue to run preconfigured models that simulate
%over versus undersampling. The v3_sweep branch will sweep across parameter
%values for each model and plot configured versus estimated [parameters and
%performance).

%v3 note: I have a bad habit of using identifier and indicator
%interchangeably.

%v2: I changed the biased values and biased rewards models to be single
%parameter (Just threshold with a fixed high slope to resemble a sharp
%threshold). Neither all draws nor last draw seem to vary both parameters
%as configured but mainly varies only slope, with some exception. So number
%of models changes

%v2: Also added to v1 some more visualisatioin tools: scatterplots to compare estimated anbd configured
%performance subject by subject and modifications to parameter visuation figure

%V1: originally implemented slope and threshold versions of BV and BR.


addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\plotSpread'));
addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\klabhub-bayesFactor-3d1e8a5'));


use_file_for_plots = 1; %Set the above to zero and this to 1 and it'll read in a file you specify (See filename_for_plots variable below) and make plots of whatever analyses are in the Generate_params structure in that file;
make_plots = 1;         %if 1, plots the results
all_draws_set = 1;          %You can toggle how the ll is computed here for all models at once if you want or go below and set different values for different models manually in structure
log_or_not = 0; %I'm changing things so all simulated data is logged at point of simulation (==1) or not
%1: cutoff 2: Cs 3: dummy (formerly IO in v2) 4: BV 5: BR 6: BPM 7: Opt 8: BPV
%(I keep model 3 as a legacy for IO because analyseSecertaryNick_2021
%looks for identifiers 4 and 5 for BV and BR and needs that for v2. Also it keeps the same color scheme as v2)
do_models = [3 1 2 7 4 5];    %These are now what v2 called model identifiers - Applies at the moment to both make_model_data and check_params;
do_models = [1];    %These are now what v2 called model identifiers - Applies at the moment to both make_model_data and check_params;
model_bar_order = [2 5 3 4 1];    %These are the indices into do_models placed in their nerw order
comment = 'test';    %The filename will already fill in basic parameters so only use special info for this.
outpath = 'C:\matlab_files\fiance\parameter_recovery\outputs';
%Unfortunately still needs to be typed in manually
filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_av_20212906.mat';   %av
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_matchNoLog_20210407.mat';   %matchmaker
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_sTrustNoLog_20211007.mat'; %small trust
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_bigtrustNoLog_20211007.mat';    %big trust
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_facesonlineSubsLog0_20212807.mat';    %faces online
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_facesOpendaySubsLog0_20212707.mat';    %faces open day
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_foodsonlineSubsLog0_20213107.mat';    %foods open day
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_foodsOpendaySubsLog0_20210208.mat';    %foods open day
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_holsOnlineSubsLog0_20210308.mat';    %hols open day
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_holsOpendaySubsLog0_20210408.mat';    %hols open day

% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_HybridPrior2SubsLog0vals0_20211208.mat';   
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_HybridPrior2SubsLog0vals0_20211208_save.mat';  
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_HybridPriorSubsLog0vals1_20211108.mat';  
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridPrior_betaFix_Log0vals0_20222311.mat';
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridPrior_betaFix_Log0vals1_20222511.mat';
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_HybridPrior2SubsLog0vals0_20223003.mat';    
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_HybridPrior2SubsLog0vals1_20223003.mat';    
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridPrior_betaFix_Log0vals0_20220112.mat'; %rerun prior/ratings for sanity check
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridPrior_betaFix_Log0vals1_20220112.mat'; %rerun prior/ratings for sanity check

%new fits to PRICES in the other hybrid conditions
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridpriorDeBugged2_Log0vals0_20231701.mat';
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridbaselineDeBugged2_Log0vals0_20231201.mat';
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridrewardDeBugged2_Log0vals0_20231201.mat';
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridtimingDeBugged2_Log0vals0_20231501.mat';
%filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridSquaresDeBugged2_Log0vals0_20231901.mat';
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\hybrid_combined_objective_v2.mat';

%full and pilot2 (bugged)
%filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridPilot2DeBugged2_Log0vals0_20231101.mat';
%filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridFullDeBugged2_Log0vals0_20231101.mat';
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridFull_Log0vals1_20230401.mat';
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridPilot2_Log0vals1_20230601.mat';

%filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridFull_fromRawData_Log0vals1_20230303.mat';
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridFull_fromRawData_Log0vals0_20230303.mat';

%uses new code which analyses data directly from raw data instead of Sahira's processes data, debugged for sub and obj values
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridStudy2_fromRawData_Log0vals1_20230503.mat';
%filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridStudy2_fromRawData_Log0vals0_20230503.mat';
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridStudy5_fromRawData_Log0vals1_20230803.mat';   %study 5 is ratings condition
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridStudy5_fromRawData_Log0vals0_20230803.mat';   %study 5 is ratings condition
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridStudy2_fromRawData_Log0vals1_20231003.mat';   %study 2 is full pilot condition
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridStudy2_fromRawData_Log0vals0_20231003.mat';   %study 2 is full pilot condition
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridStudy1_fromRawData_Log0vals0_20231203.mat';   %study 2 is full pilot condition
%filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridStudy3_fromRawData_Log0vals0_20231203.mat';   %study 2 is full pilot condition
% filename_for_plots = ['C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\outputs\out_NEWnoIO_ll1pay1vals020230307.mat']; %Unfortunately still needs to be typed in manually
% filename_for_plots = ['C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\outputs\out_NEWnoIO_ll1pay1vals120230307.mat']; %Unfortunately still needs to be typed in manually


%fMRI
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridfMRItest_Log0vals1_20230601.mat';

%NEW N=151 study
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_test_jscriptNEW_Pay1vals0_20231306.mat'


%These correspond to identifiers (not configured implementations like in v2) in the v3_sweep version
model_names = {'Cut off' 'Cs' 'IO' 'BV' 'BR' 'BPM' 'Opt' 'BPV' }; %IO is a placeholder, don't implement
num_model_identifiers = size(model_names,2);
subjects = 1:64;    %big trust sub nums
% subjects = 1;    %big trust sub nums
IC = 2; %1 if AIC, 2 if BIC
nbins_psi = 6;                  %how many bins in which to divide option values when constructed psychometric curves (the idea is to use as many as possible, such that you avoid missing data in too many sequence position*option value bin combinations and get smooth sensible curves. For some studies, where there are 9 ratings levels, then the 9 bins means every raw level is used.
analyze_value_positions = 0;    %Create plots with psychometric curves, their thresholds (model fits) and their correlations (nbins_psi hardwired at function call)
BayesThresh = 3;    %What Bayes factor threshold to be used when painting sig lines on plots? (3 is substantial and 10 is strong)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%PLOT!%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if make_plots == 1;

    %Or open a file where these things were computed on previous run?
    if use_file_for_plots == 1;
        
    %Use data already in
        load(filename_for_plots,'Generate_params');
        
    end;    %Plot data structure from a file?
    
    %Add some stuff to the structure that will be needed for plotting and
    %further analysis
    Generate_params.IC = IC;    %AIC (0) or BIC (1) correction?
    Generate_params.nbins_psi = nbins_psi;
    Generate_params.analyze_value_positions = analyze_value_positions;
    Generate_params.model_bar_order = model_bar_order;
    Generate_params.BayesThresh = BayesThresh;
    
    %run the main function call
    plot_data(Generate_params);
    
end;    %Do plots or not?


%Just to be safe
% save([Generate_params.outpath filesep Generate_params.outname],'Generate_params');

disp('audi5000')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function analyze_value_position_functions(value_data,choice_trial,plot_cmap,binEdges_psi,legend_labels,param_to_fit,two_params,model_bar_order, BayesThresh);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Look at proportion choice, position and value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param_to_fit = [0 param_to_fit];    %the first model in this function is subject so add it to this list as zeroth model so it can be indexed (e.g., in colormaps)

%value_data was log(raw_seq_subs), contains ratings data in sequences and is seq*position*sub
nbins = size(binEdges_psi,2)-1;
num_subs = size(value_data,3);
num_positions = size(value_data,2);
num_seqs = size(value_data,1);
num_models = size(choice_trial,3);
serial_r = value_data;  %only for ratings plots, nothing else

f_a = 0.1; %face alpha
sw = 0.5;  %ppoint spread width
font_size = 12;

%ok. this time, let's be less efficient but more organised. I want to bin
%things right up front before anything so there is a ratings dataset
%(value_data) and a binned dataset (value_bins)
for sub = 1:num_subs;
    binEdges = binEdges_psi(sub,:);
    [dummy,value_bins(:,:,sub)] = histc(value_data(:,:,sub), [binEdges(1:end-1) Inf]);
end;

%nan mask has zeros for view decisions and 1s for take decisions and NaNs
%when no face was seen because of elapsed decision. Can use to mask other
%arrays later ...
nan_mask = NaN(num_seqs,num_positions,num_subs,num_models);
for model=1:num_models;
    for sub=1:num_subs;
        for seq=1:num_seqs;
            nan_mask(seq,1:round(choice_trial(sub,seq,model)),sub,model) = 0;
            nan_mask(seq,round(choice_trial(sub,seq,model)),sub,model) = 1;
        end;
    end;
end;

%now we have two seq*position*subject arrays, one ratings, one bins, now
%make supersubject seq*position versions by concatenating subjects. Effectively, each
%new subject justs adds new sequences so its a long list of sequences
value_data_sub = [];
value_bins_sub = [];
choice_trial_sub = [];

for sub=1:num_subs;
    value_data_sub = [value_data_sub; value_data(:,:,sub)];
    value_bins_sub = [value_bins_sub; value_bins(:,:,sub)]; %bins are still subject specific
    choice_trial_sub = [choice_trial_sub; squeeze(choice_trial(sub,:,:))];
end;

nan_mask_sub = [];
for model=1:num_models;
    temp = [];
    for sub=1:num_subs;
        temp = [temp; squeeze(nan_mask(:,:,sub,model))];
    end;
    nan_mask_sub(:,:,model) = temp;
end;

%yes, it's yet another subject loop. I'm being modular. This one prepares
%the average ratings * serial position data. It also computes the proportion choices *serial position data.
%It also computes proportion subject predicted * serial position data
%This one needed whether using a super subject or fitting all subjects, it's a separate analysis
model_predicted_choices = NaN(num_subs,num_positions,num_models-1); %for proportion correctly predicted subject choices
position_choices = NaN(num_subs,num_positions,num_models);   %for proportion responses serial positions
position_function = zeros(num_positions,num_subs,num_models);   %for average ratings as function of serial position
position_it = zeros(num_positions,num_subs,num_models);         %for average ratings as function of serial position
for sub=1:num_subs;
    
    this_subject_ratings = squeeze(serial_r(:,:,sub));  %only for plotting ratings by serial position
    
    for position=1:num_positions; %loop through the positions
        for model=1:num_models;
            
            sub_choices_this_position = nan_mask(:,position,sub,1);
            model_choices_this_position = nan_mask(:,position,sub,model);
            % %             %computes proportion responses for each position
            position_choices(sub,position,model) = sum( choice_trial(sub,:,model) == position )/size(choice_trial,2);
            
            %find average attractiveness of the choices in each position
            this_subject_choices = squeeze(choice_trial(sub,:,model));
            indices_into_choices = find(this_subject_choices==position);
            if ~isempty(indices_into_choices);
                for i=1:size(indices_into_choices,2);
                    position_function(position,sub,model) = position_function(position,sub,model)+this_subject_ratings(indices_into_choices(i),position);
                    position_it(position,sub,model) = position_it(position,sub,model)+1;
                end;    %loop through values for this position
            end;    %is there a valid position value?
            
        end;    %model
    end;        %position
end;            %subject

%individual subs
position_data_indiv = NaN(nbins,num_positions,num_subs,num_models);            %for value function analyses as function of serial position
ave_rating_per_bin_indiv = NaN(nbins,num_positions,num_subs,num_models);       %use this for curve fitting later

for sub=1:num_subs;
    
    for position=1:num_positions; %loop through the positions
        for model=1:num_models;
            
            this_subject_bins = value_bins(:,:,sub);
            temp2 = squeeze(nan_mask(:,:,sub,model));
            this_subject_bins(isnan(temp2(:)))=NaN;
            
            for val_bin = 1:nbins;
                
                %find bins at this position and,if any, check what are the CHOICES and RATINGS for that bin/position
                trials_with_bins_in_this_position = [];
                trials_with_bins_in_this_position = find( this_subject_bins(:,position) == val_bin );   %on which sequences did a value in this bin occur in this position?
                num_trial_with_vals = numel(trials_with_bins_in_this_position);                     %how many sequences have this value in this position?
                position_data_indiv(val_bin,position,sub,model) = sum(choice_trial(sub,trials_with_bins_in_this_position,model)==position)/ num_trial_with_vals ; %Now I need the number of CHOICES for this positon and bin
                ave_rating_per_bin_indiv(val_bin,position,sub,model) = nanmean(value_data(trials_with_bins_in_this_position,position,sub));   %need this for regression later (no sense of model here)
                
            end;    %value bin
        end;    %model
    end;        %position
end;            %subject


position_data = position_data_indiv;
ave_rating_per_bin = ave_rating_per_bin_indiv;

%loop again, this time through positions and fit averages over subjects
for model=1:num_models;
    for position=1:num_positions;
        
        %computes value slopes for each position, and model
        this_position_no_subs = nanmean(squeeze( position_data(:,position,:,model) ),2);   %returns bin values for a subject in a position
        
        y_this_position = this_position_no_subs(~isnan(this_position_no_subs));
        x_this_position = [1:nbins];
        x_this_position = x_this_position(~isnan(this_position_no_subs))';
        clear f_position;
        if numel(x_this_position)<3 | position==num_positions | sum(this_position_no_subs) == 0;    %if there are too many nans and not enough datapoints, if its that last position with the flat line or all ones, or if no response was ever made
            
            b1(position,model) = NaN;
            b2(position,model) = NaN;
        else
            %             f_position=fit(x_this_position,y_this_position,'1./(1+exp(-p1*(x-p2)))','StartPoint',[1 5],'Lower',[0 1],'Upper',[Inf 8]);
            %             temp_coef = coeffvalues(f_position);
            %             b1(position,model) = temp_coef(1);  %if only slope and mid are free
            %             b2(position,model) = temp_coef(2); %if only slope and mid are free
            if two_params == 1;
                %             Two params free
                f_position=fit(x_this_position,y_this_position,'1./(1+exp(-p1*(x-p2)))','StartPoint',[1 5],'Lower',[0 1],'Upper',[Inf 8]);
                temp_coef = coeffvalues(f_position);
                b1(position,model) = temp_coef(1);  %if only slope and mid are free
                b2(position,model) = temp_coef(2); %if only slope and mid are free
                
            else
                
                % %             %Three params free
                f_position=fit(x_this_position,y_this_position,'p1./(1+exp(-p3*(x-p4)))','StartPoint',[1 1 5],'Lower',[0 0 1],'Upper',[1 Inf 8]);
                temp_coef = coeffvalues(f_position);
                b1(position,model) = temp_coef(2);  %if only slope and mid are free
                b2(position,model) = temp_coef(3); %if only slope and mid are free
            end;
            
        end;    %check is there enough data to do a fit
        
    end;    %loop through positions
end;    %loop through models

b_ci = zeros(size(b2));
b = b2;

%%%%%%%new part: correlation position data for each model with subjects
r_graph = zeros(1,num_models);
r_ci_graph = zeros(1,num_models);
for model = 1:num_models;
    
    if model ~=1;
        
        for sub=1:num_subs;
            
            clear this_subject_data this_model_data this_subject_data_rs this_model_data_rs
            
            %extract_data
            this_subject_data = squeeze(position_data(:,:,sub,1));
            this_model_data = squeeze(position_data(:,:,sub,model));
            
            %reshape data
            this_subject_data_rs = reshape(this_subject_data,prod(size(this_subject_data)),1);
            this_model_data_rs = reshape(this_model_data,prod(size(this_model_data)),1);
            
            %correlate them
            [temp1 temp2] = corrcoef(this_subject_data,this_model_data,'rows','complete');
            r(sub,model-1) = temp1(2,1);
            p(sub,model-1) = temp2(2,1);
            sub_nums(sub,model-1) = sub;
            mod_nums(sub,model-1) = model-1;
            
        end;    %loop through subs
        
    end;    %only consider models other than subjects
    
end;    %models

r_graph = [0 nanmean(r,1)];
r_ci_graph = [0 1.96*(nanstd(r)/sqrt(size(r,1)))];


%average proportion responses, over subjects
mean_position_choices = squeeze(mean(position_choices,1));
ci_position_choices = squeeze(1.96*(std(position_choices,1,1)/sqrt(size(position_choices,1))));
%average ratings as function of serial position
clear ave_ratings ave ci;
ave_ratings = position_function./position_it;

%serial postion plots: average rating, proportion correct and value sensitivity slopes
h3 = figure; set(gcf,'Color',[1 1 1]);  %For serial position/PSE plots/correlation plots
h4 = figure; set(gcf,'Color',[1 1 1]);  %For psychometric function plots
for model = 1:size(choice_trial,3);
    
    markersize = 3;
    
    %average rating as function of serial positon
    legend_locs = [0.5:-0.05:(0.5 - (0.05*5))];
    %
    %     %proportion choices
    %     figure(h3); subplot( 2,2,3); hold on;
    %     sph = shadedErrorBar(1:size(mean_position_choices,1),mean_position_choices(:,model),ci_position_choices(:,model),{'MarkerFaceColor',plot_cmap(param_to_fit(model)+1,:),'MarkerEdgeColor',plot_cmap(param_to_fit(model)+1,:),'Marker','o','MarkerSize',markersize,'LineStyle','-'},1); hold on;
    %     set(sph.mainLine,'Color',plot_cmap(param_to_fit(model)+1,:));
    %     set(sph.patch,'FaceColor',plot_cmap(param_to_fit(model)+1,:));
    %     set(sph.edge(1),'Color',plot_cmap(param_to_fit(model)+1,:));
    %     set(sph.edge(2),'Color',plot_cmap(param_to_fit(model)+1,:));
    %     %         text(3,legend_locs(model),legend_names{model},'Color',plot_cmap(param_to_fit(model)+1,:),'FontSize',12,'FontName','Arial');
    %     box off;
    %     %axis square;
    %     set(gca,'FontSize',12,'FontName','Arial','xtick',[1:size(b,1)],'ytick',[0.1:0.1:0.8],'Ylim',[0 0.5],'Xlim',[1 size(b,1)],'LineWidth',2);
    %     xlabel('Position in Sequence'); ylabel('Proportion Choices');
    %
    %psychometric function parameters
    figure(h3); subplot( 1,2,1 ); hold on;
    
    %Need to re-order lines according to model_bar_order
%     this_model = model_bar_order(model);
    
    %These are plots of thresholds over sequence positions. They are not
    %ordered but just overlaid so it makes no difference whether you
    %correct for model order or not. But it matters for the legend.
    legend_positions = [3:-.5:0];
    errorbar(1:size(b,1),b(:,model),b_ci(:,model),'Color',plot_cmap(param_to_fit(model)+1,:),'MarkerFaceColor',plot_cmap(param_to_fit(model)+1,:),'MarkerEdgeColor',plot_cmap(param_to_fit(model)+1,:),'Marker','o','MarkerSize',markersize,'LineStyle','-','LineWidth',1); hold on;
    if model == 1;
        text(2,legend_positions(model),'Participants', 'Color',plot_cmap(1,:), 'FontSize',12,'FontName','Arial');
    else
        text(2,legend_positions(model),legend_labels{model_bar_order(model-1)}, 'Color',plot_cmap( param_to_fit(model_bar_order(model-1)+1)+1,:), 'FontSize',12,'FontName','Arial');
    end;
    box off;
    set(gca,'FontSize',12,'FontName','Arial','xtick',[1:size(b,1)],'Xlim',[1 size(b,1)],'YLim',[0 num_positions],'LineWidth',2);
    xlabel('Position in sequence'); ylabel('Choice threshold');
    
    if model ~=1;   %no subjects
        
        %model correlations with proportion choice
        figure(h3); subplot( 1,2,2 ); hold on;
        %         legend_positions = [1.1:-.05:0];
        
        corrected_model_nums = model_bar_order(model-1);
        
        handles = plotSpread(r(:,corrected_model_nums), ...
            'xValues',model,'distributionColors',plot_cmap(param_to_fit(model_bar_order(model-1)+1)+1,:),'distributionMarkers','.', 'spreadWidth', sw);
        
        bar(model,r_graph(corrected_model_nums+1), ...
            'FaceColor',plot_cmap(param_to_fit(model_bar_order(model-1)+1)+1,:),'FaceAlpha',f_a,'EdgeColor',[0 0 0] );
        
        % %         text(1.5,legend_positions(model),legend_labels{param_to_fit(model)}, 'Color',plot_cmap(param_to_fit(model)+1,:), 'FontSize',12,'FontName','Arial');
        %          text(1.5,legend_positions(model),legend_labels{model-1}, 'Color',plot_cmap(param_to_fit(model)+1,:), 'FontSize',12,'FontName','Arial');
        %
        set(gca,'FontSize',12,'FontName','Arial', 'XTick',[],'xticklabel',{[]},'YTick',[(floor(min(min(r))/20)*20):.2:1],'yticklabel',{[(floor(min(min(r))/20)*20):.2:1]},'LineWidth',2);
        ylabel('Model-participant correlation');
        xlim([1 numel(r_graph)+0.5]);
        ylim([(floor(min(min(r))/.2)*.2) 1.3]);
        
        x_axis_test_offset = 0.025;
        this_offset = (floor(min(min(r))/.2)*.2) - x_axis_test_offset*diff(ylim);
        text( model, this_offset ...
            ,sprintf('%s',legend_labels{corrected_model_nums}) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',12 ...
            ,'Rotation',25 ...
            ,'HorizontalAlignment','right' ...
            );
        
    end;    %If not a subject
    
    
    
    %value psychometric functions (in a different figure with different colormap), with lines for each position
    figure(h4);
    
    %reorder subplots to match the model order this prgram has been using
    
    if model == 1;
        new_model_position=1;
    else
        new_model_position = find(model_bar_order == model-1)+1;
    end;
    
%     new_model_position = model;
    
    subplot(1,num_models,new_model_position);
    pm_line_colors = cool(size(position_data,2)+1);
    
    for position_line = 1:size(position_data,2)-1;
        
        if numel(size(position_data))==4;
            h = plot( nanmean(squeeze(position_data(:,position_line,:,model)),2) ); hold on;
        else
            h = plot( position_data(:,position_line,model) ); hold on;
        end;
        axis square;
        set(h,'Marker','o','MarkerSize',6,'MarkerEdgeColor',pm_line_colors(position_line,:),'MarkerFaceColor',pm_line_colors(position_line,:),'Color',pm_line_colors(position_line,:),'LineStyle','-','LineWidth',2);
        set(gca,'FontSize',12,'FontName','Arial','xtick',[1:size(position_data,1)],'xlim',[0.5 size(position_data,1)+0.5],'ylim',[0 1.1],'ytick',[0:0.2:1],'LineWidth',2);
        xlabel('Option value bin'); ylabel('Proportion Choices'); box off;
        
    end;    %position lines
    
    if model == num_models;
        legend('Position 1','Position 2','Position 3','Position 4','Position 5','Position 6','Position 7','Position 8','Position 9','Position 10','Position 11','Position 12');
    end;
    
end;    %loop through models


%the model loop is done but we need to add the sig tests using the accumulated model data
%stole and modified code from BIC plot
%note: this num_models variable counts participants as a model
if num_models-1 ~= 1 & num_subs > 1;
    
    %run and plot ttests
    figure(h3); subplot( 1,2,2 );
    pairs = nchoosek(1:num_models-1,2);
    num_pairs = size(pairs,1);
    [a In] = sort(diff(pairs')','descend');  %lengths of connecting lines
    line_pair_order = pairs(In,:);    %move longest connections to top
    
    %Where to put top line?
    y_inc = .055;
    ystart = max(max(r)) + y_inc*num_pairs + 2*y_inc;
    line_y_values = ystart:-y_inc:0;
    
    fprintf(' ');
    
    for pair = 1:num_pairs;
        %
        %         %run ttest this pair (with Fisher r to z)
        %         [h r_pvals(pair) ci stats] = ttest(atanh(r(:,line_pair_order(pair,1))), atanh(r(:,line_pair_order(pair,2))));
        
        %remove NaNs (if any), as it seems bf.ttest chokes on them.
        %NaNs may arise from missing data in the psychometric
        %functions due to small numbers of trials per participant.
        %            temp =  atanh(r(:,line_pair_order(pair,1))) - atanh(r(:,line_pair_order(pair,2)));
        
        r_corrected = r(:,model_bar_order);
        
        temp =  r_corrected(:,line_pair_order(pair,1)) - r_corrected(:,line_pair_order(pair,2));
        these_data = temp(find(~isnan(temp)));
        
        %get Bayes factor too
        [bf10(pair),r_pvals(pair),ci,stats] = ...
            bf.ttest( these_data  );
        
        %plot result
        %             subplot(2,4,6); hold on;
        %         set(gca,'Ylim',[0 ystart]);
        ylim([(floor(min(min(r))/.2)*.2) ystart]);
        
        %         %correct line_pair_orders for the corrected model x axis positions
        %         x_position_1 = find( model_bar_order == line_pair_order(pair,1))+1;
        %         x_position_2 = find( model_bar_order == line_pair_order(pair,2))+1;
        
        x_position_1 = line_pair_order(pair,1)+1;
        x_position_2 = line_pair_order(pair,2)+1;
        
        if r_pvals(pair) < 0.05/size(pairs,1);
            plot([x_position_1 x_position_2],...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',4,'Color',[0 0 0]);
        end;
        
        if bf10(pair) < (1/BayesThresh);
            plot([x_position_1 x_position_2],...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',2,'Color',[1 0 1]);
        end;
        if bf10(pair) > BayesThresh;
            plot([x_position_1 x_position_2],...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',1,'Color',[0 1 0]);
        end;
        %
        %             plot([line_pair_order(pair,1)+1 line_pair_order(pair,2)+1],...
        %                 [line_y_values(pair) line_y_values(pair)],'LineWidth',2,'Color',[0 0 0]);
        
        
    end;    %loop through ttest pairs
    
end;    %Only compute ttests if there is at least one pair of models
























%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = plot_data(Generate_params);

%set up plot appearance
%For now I'll try to match model identifiers to colors. Which means this
%colormap needs to scale to the total possible number of models, not the
%number of models
plot_cmap = hsv(8+1);  %models + subjects
f_a = 0.1; %face alpha
sw = 1;  %ppoint spread width
graph_font = 12;
x_axis_test_offset = .07;   %What percentage of the y axis range should x labels be shifted below the x axis?

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%SAMPLES AND RANKS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Plot samples for participants and different models
h10 = figure('NumberTitle', 'off', 'Name',['parameters ' Generate_params.outname]);
set(gcf,'Color',[1 1 1]);

%Need to accumulate these for analyze_value_position_functions below.
all_choice_trial(:,:,1) = Generate_params.num_samples'; %Should be subs*seqs*models. num_samples here (human draws) is seqs*subs
model_strs = {};
for this_bar = 1:Generate_params.num_models+2; 

    %     for perf_measure = 1:2;   %samples or ranks
    %     for perf_measure = 1;   %samples

%So now I've changed it so +1 is optimal / io and +2 is participanmts then
%the models follow after that.

    %         if perf_measure == 1;   %If samples
    subplot(2,3,1); hold on; %Samples plot
    y_string = 'Samples to decision';
    
if this_bar == 1;   %if optimality 
   
        optimal_model_num =  Generate_params.num_models + 1;    %ideal observer was estimated like ther others, it's a later add on so not included in num_models
        these_data = nanmean(Generate_params.model(optimal_model_num).num_samples_est)';
        plot_color = [.5 .5 .5];
        model_label = 'Optimal';

elseif this_bar == 2;   %If participants

        these_data = nanmean(Generate_params.num_samples)';
        plot_color = [1 0 0];
        model_label = 'Participants';

    else;   %if model
        
        corrected_model_num = Generate_params.model_bar_order(this_bar-2);
        
        these_data = nanmean(Generate_params.model(corrected_model_num).num_samples_est)';
        plot_color = plot_cmap(Generate_params.model(corrected_model_num).identifier+1,:);
        model_label = Generate_params.model(corrected_model_num).name;
        
%         these_data = nanmean(Generate_params.model(this_bar-1).num_samples_est)';
%         plot_color = plot_cmap(Generate_params.model(this_bar-1).identifier+1,:);
%         model_label = Generate_params.model(this_bar-1).name;
    end;    %partricipants or model?
    
    %accumulate data for computation of pairwise tests and the output to file later on
    samples_accum(:,this_bar) = these_data;
    
    %         else;   %If ranks
    % %             subplot(2,2,2); hold on; %Samples plot
    % %             y_string = 'Rank of chosen option';
    % %             if this_bar == 1;   %If participants
    % %                 these_data = nanmean(Generate_params.ranks)';
    % %                 plot_color = [1 0 0];
    % %                 model_label = 'Participants';
    % %             else;   %if model
    % %                 these_data = nanmean(Generate_params.model(this_bar-1).ranks_est)';
    % %                 plot_color = plot_cmap(Generate_params.model(this_bar-1).identifier+1,:);
    % %                 model_label = Generate_params.model(this_bar-1).name;
    % %             end;    %partricipants or model?
    %
    %         end;    %samples or ranks?
    
    %average over sequences (rows) but keep sub data (cols) for scatter points
    handles = plotSpread(these_data ...
        ,'xValues',this_bar ...
        ,'distributionColors',plot_color ...
        ,'distributionMarkers','.' ...
        , 'spreadWidth', sw ...
        );
    
    bar(this_bar,nanmean(these_data) ...
        ,'FaceColor',plot_color ...
        ,'FaceAlpha',f_a ...
        ,'EdgeColor',[0 0 0] ...
        );
    
    set(gca ...
        ,'XTick',[] ...
        ,'fontSize',graph_font ...
        ,'FontName','Arial',...
        'XLim',[0 Generate_params.num_models+3] ...
        ,'YLim',[0 Generate_params.seq_length] ...
        ,'YTick',[0:2:Generate_params.seq_length] ...
        ,'LineWidth',2 ...
        );
    ylabel(y_string);
    
    this_offset = -x_axis_test_offset*diff(ylim);
    text( this_bar, this_offset ...
        ,sprintf('%s',model_label) ...
        ,'Fontname','Arial' ...
        ,'Fontsize',graph_font ...
        ,'Rotation',45 ...
        ,'HorizontalAlignment','right' ...
        );
    
    %     end;    %switch between samples and ranks
    
    %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %     %%%LOG LIKELIHOOD ANALYSIS
    %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %Still inside model (this_bar) loop
    if this_bar ~= 1 & this_bar ~= 2;
        %
        %         %%%%%%%%%%%%%%%%%%%%
        %         %%%%Plot of raw ll's
        %         subplot(2,4,2);
        %
        %         %accumulate ll's for output to file later
        ll_accum(:,this_bar-2) = Generate_params.model(this_bar-2).ll;
        %
        %         handles = plotSpread(Generate_params.model(this_bar-1).ll ...
        %             ,'xValues',this_bar ...
        %             ,'distributionColors',plot_color ...
        %             ,'distributionMarkers','.' ...
        %             , 'spreadWidth', sw ...
        %             );
        %
        %         bar(this_bar,nanmean(Generate_params.model(this_bar-1).ll) ...
        %             ,'FaceColor',plot_color ...
        %             ,'FaceAlpha',f_a ...
        %             ,'EdgeColor',[0 0 0] ...
        %             );
        %
        %         set(gca ...
        %             ,'XTick',[] ...
        %             ,'fontSize',graph_font ...
        %             ,'FontName','Arial',...
        %             'XLim',[1 Generate_params.num_models+2] ...
        %             ,'LineWidth',2 ...
        %             );
        %         %                     ,'YLim',[0 Generate_params.seq_length]...0
        %         ylabel('Log-likelihood');
        %
        %         this_offset = -x_axis_test_offset*diff(ylim);
        %         text( this_bar, this_offset ...
        %             ,sprintf('%s',model_label) ...
        %             ,'Fontname','Arial' ...
        %             ,'Fontsize',graph_font ...
        %             ,'Rotation',25 ...
        %             ,'HorizontalAlignment','right' ...
        %             );
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        %Plot of AIC/BIC (Not so relevant if they all have two parameters though)
        subplot(2,3,2);
        
        
        
        %Model IC
        no_params = numel( Generate_params.model(this_bar-1).this_models_free_parameters ) + 1; %+1 for beta
       
        lla = Generate_params.model(corrected_model_num).ll;
%         lla = Generate_params.model(this_bar-1).ll;

        if Generate_params.IC == 1; %If AIC (per participant)
            IC_pps = 2*no_params + 2*lla;
            %             IC_sum = nansum(IC_pps);
            %             IC_sum = 2*no_params(param_to_fit(model)) + 2*nansum(lla);
            a_label = 'AIC';
            %             IC_ylims = [800 1350];
        elseif Generate_params.IC == 2; %If BIC (per participant)
            IC_pps = no_params*log(Generate_params.num_seqs) + 2*lla;
            %             IC_sum = nansum(IC_pps);
            %             IC_sum = no_params(param_to_fit(model))*log(numel(lla)*28) + 2*nansum(lla);
            a_label = 'Bayesian information criterion';
            %             IC_ylims = [750 1250];
        end;
        
        handles = plotSpread(IC_pps ...
            , 'xValues',this_bar-1 ...
            ,'distributionColors',plot_color ...
            ,'distributionMarkers','.' ...
            , 'spreadWidth', sw ...
            );
        
        bar(this_bar-1,nanmean(IC_pps), ...
            'FaceColor',plot_color,'FaceAlpha',f_a,'EdgeColor',[0 0 0] );
        
        %35 seems like a common max BIC value for the data in the figures
        %I need to specify some estimate of this now to put the x axis
        %labels in the right place (based on ylim) even though I don;t have
        %easy access to the max value of all BICs until after this model
        %loop finishes.
        set(gca ...
            ,'XTick',[] ...
            ,'fontSize',graph_font ...
            ,'FontName','Arial' ...
            ,'XLim',[1 Generate_params.num_models+2] ...
            ,'YLim',[0 45] ...
            ,'LineWidth',2 ...
            );
        ylabel(a_label);
        
        %I'm going to wait until the ttest bars are in place and I know the max y
        %lim before adding x axis labels with test (so I can rotate).
        
        %We need to accumulate these data over models in this loop to do the
        %next step more easily
        IC_pps_all_models(:,this_bar-2) = IC_pps';
        %These we need to accumulate so they can be passed into
        %analyze_value_position_functions below
        all_choice_trial(:,:,this_bar-1) = Generate_params.model(this_bar-2).num_samples_est';
        model_strs{this_bar-2} = Generate_params.model(this_bar-2).name;
        
    end;    %If not participants (this bar ~=1)
    
end;    %loop through models





%Add pairwise test results to SAMPLES PLOT
if Generate_params.num_models ~= 1 & Generate_params.num_subs > 1;
    
    %Paint ttest / Bayes factor results onto samples graph (This one only
    %needs participants compared against each of the models
    %Where to put top line?
    subplot(2,3,1);
    
%     num_pairs = numel(Generate_params.do_models_identifiers);
    num_pairs = size(samples_accum,2);
    y_inc = .5;
    ystart = max(max(samples_accum)) + y_inc*num_pairs +y_inc;
%     line_y_values = ystart:-y_inc:0;
    line_y_values = (0:y_inc:ystart) + (max(max(samples_accum)) + y_inc);

%     for pair = 1:num_pairs;
    for pair = num_pairs:-1:1;

        [bf10(pair),samples_pvals(pair),ci,stats] = ...
                        bf.ttest( samples_accum(:,2) - samples_accum(:,pair) );

%             bf.ttest( samples_accum(:,1) - samples_accum(:,num_pairs + 2 - pair) );
        
        set(gca,'Ylim',[0 ystart]);
        
        %distance on plot
%         distance_on_plot = [1 num_pairs + 2 - pair];
        distance_on_plot = [2 pair];
        
        if samples_pvals(pair) < 0.05/num_pairs;
            plot(distance_on_plot,...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',4,'Color',[0 0 0]);
        end;
        
        if bf10(pair) < (1/Generate_params.BayesThresh);
            plot(distance_on_plot,...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',2,'Color',[1 0 1]);
        end;
        if bf10(pair) > Generate_params.BayesThresh;
            plot(distance_on_plot,...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',1,'Color',[0 1 0]);
        end;
        
        
    end;   %pairs: Loop through comparisons between participants and models
    
    
    
    
    
    %run and plot ttests and Bayesfactor results on the plots above for BIC
    %graph (This one needs comparisons among all models)
    subplot(2,3,2);
    pairs = nchoosek(1:Generate_params.num_models,2);
    num_pairs = size(pairs,1);
    [a In] = sort(diff(pairs')','descend');  %lengths of connecting lines
    line_pair_order = pairs(In,:);    %move longest connections to top
    
    %Where to put top line?
    max_y = max(max(IC_pps_all_models));
    y_inc = .065*max_y;
    ystart = max_y + y_inc*num_pairs;
    line_y_values = ystart:-y_inc:0;
    
    for pair = 1:num_pairs;
        %
        %         %run ttest this pair
        %         [h IC_pp_pvals(pair) ci stats] = ttest(IC_pps_all_models(:,line_pair_order(pair,1)), IC_pps_all_models(:,line_pair_order(pair,2)));
        %
        %get Bayes factor too
        [bf10(pair),IC_pp_pvals(pair),ci,stats] = bf.ttest( IC_pps_all_models(:,line_pair_order(pair,1)) - IC_pps_all_models(:,line_pair_order(pair,2)) );
        
        %plot result
        %             subplot(2,4,6); hold on;
        
        yticks = linspace(0, ceil(max_y/20)*20,5);
        set(gca,'Ylim',[0 ystart],'YTick',yticks);
        
        %         if bf10(pair) < (1/10);
        %             pair_color = [1 0 1];   %magenta
        %         elseif IC_pp_pvals(pair) < 0.05/size(pairs,1) & bf10(pair) > 10;
        %             pair_color = [0 .5 0];
        %         elseif IC_pp_pvals(pair) < 0.05/size(pairs,1) & bf10(pair) < 10;
        %             pair_color = [0 0 0];
        %         elseif IC_pp_pvals(pair) > 0.05/size(pairs,1) & bf10(pair) > 10;
        %              pair_color = [0 1 0];
        %         else
        %             pair_color = [1 1 1];
        %         end;
        
        if IC_pp_pvals(pair) < 0.05/size(pairs,1);
            plot([line_pair_order(pair,1)+1 line_pair_order(pair,2)+1],...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',4,'Color',[0 0 0]);
        end;
        
        if bf10(pair) < (1/Generate_params.BayesThresh);
            plot([line_pair_order(pair,1)+1 line_pair_order(pair,2)+1],...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',2,'Color',[1 0 1]);
        end;
        if bf10(pair) > Generate_params.BayesThresh;
            plot([line_pair_order(pair,1)+1 line_pair_order(pair,2)+1],...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',1,'Color',[0 1 0]);
        end;
        
    end;    %loop through ttest pairs
    
end;    %Only compute ttests if there is at least one pair of models

%Now put in x axis labels that full y lim is known
for this_bar = 1:Generate_params.num_models;
    
    corrected_model_num = Generate_params.model_bar_order(this_bar);
    
    this_offset = -(x_axis_test_offset)*diff(ylim);
    text( this_bar+1, this_offset ...
        ,sprintf('%s',Generate_params.model(corrected_model_num).name) ...
        ,'Fontname','Arial' ...
        ,'Fontsize',graph_font ...
        ,'Rotation',45 ...
        ,'HorizontalAlignment','right' ...
        );
    
end;    %end redundant loop through models






%%%%%%%%%%%%%%%%%%%%%%
%Plot of numbers of winning subs for each model

subplot(2,3,3); hold on; box off;

%winning models
[a pps_indices] = min(IC_pps_all_models');

for model = 1:Generate_params.num_models;
    
    %The frequencies themselves are already corrected for model order
    %(because the data they're based on is) but need to correct the colors
    %and the x axis labels.
    corrected_model_num = Generate_params.model_bar_order(model);
    
    
    bar(model,numel(find(pps_indices==model)), ...
        'FaceColor', plot_cmap(Generate_params.model(corrected_model_num).identifier+1,:),'FaceAlpha',f_a,'EdgeColor',[0 0 0] );
    
    %get frequency of most common model so I can set the Y axis limits (and
    %thereby correctly judge distance of x axis labels from axis)
    %     max_freq = max(histc(pps_indices,Generate_params.do_models_identifiers));
    max_freq = max(histcounts(pps_indices));
    
    set(gca ...
        ,'XTick',[] ...
        ,'fontSize',graph_font ...
        ,'FontName','Arial'...
        ,'XLim',[0 Generate_params.num_models+1] ...
        ,'YLim',[0 max_freq] ...
        ,'LineWidth',2 ...
        );
    ylabel('Frequency');
    
    this_offset = -x_axis_test_offset*diff(ylim);
    text( model, this_offset ...
        ,sprintf('%s',Generate_params.model( corrected_model_num ).name ) ...
        ,'Fontname','Arial' ...
        ,'Fontsize',graph_font ...
        ,'Rotation',45 ...
        ,'HorizontalAlignment','right' ...
        );
    
end;    %models


%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Write out the data matrices used for plots so they can be analysed statistically in JASP
%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf(' ');

%big matrix
%cols 1-6: samples participants & 5 models; cols 7-11: ll 5 models; cols 12-16: BIC 5 models
out = [samples_accum ll_accum IC_pps_all_models];

writematrix(out,'C:\matlab_files\fiance\parameter_recovery\outputs\models_av_out01.csv');


%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%INDIVIDUAL PARTICIPANT DATA SCATTERPPLOTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Scatterplots of participant-model subject by subject relationships (sampling)
h11 = figure('NumberTitle', 'off', 'Name',['parameters ' Generate_params.outname]);
set(gcf,'Color',[1 1 1]);

%For scatterplot subplots
num_rows = floor(sqrt(numel(Generate_params.do_models_identifiers )) );
num_cols = ceil(numel(Generate_params.do_models_identifiers)/num_rows);

for identifier = 1:numel(Generate_params.do_models_identifiers);
    
    %     subplot(num_rows, num_cols, identifier); hold on; box off;
%     subplot( 1, numel(Generate_params.do_models_identifiers), identifier); hold on; box off;
    subplot( 1, numel(Generate_params.do_models_identifiers), find(Generate_params.model_bar_order == identifier)); hold on; box off;
    
    scatter( ...
        nanmean(Generate_params.num_samples)' ...
        , nanmean(Generate_params.model(identifier).num_samples_est)' ...
        , 38 ... %marker size apparently
        , 'MarkerEdgeColor', plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:) ...
        ,'MarkerEdgeAlpha', .3 ...
        ,'LineWidth',2 ...
        );
    %             , 'MarkerFaceColor', plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:) ...
    
    
    %regression line
    [b,bint,r,rint,stats] = regress( ...
        nanmean(Generate_params.model(identifier).num_samples_est)' ...
        , [ones(Generate_params.num_subs,1) nanmean(Generate_params.num_samples)'] ...
        );
    %     x_vals = [min(nanmean(Generate_params.num_samples)') max(nanmean(Generate_params.num_samples)')];
    x_vals = [0 Generate_params.seq_length];
    
    y_hat = b(1) + b(2)*x_vals;
    
    %Put Rsquared on plot
    text( 0.5, Generate_params.seq_length + .5 ...
        , sprintf('R squared = %0.2f',stats(1)) ...
        , 'Fontname','Arial' ...
        , 'Fontsize',graph_font ...
        , 'FontWeight','normal' ...
        );
    
    
    %Plot a diagonal of inequality too
    plot( ...
        [0 Generate_params.seq_length] ...
        , [0 Generate_params.seq_length] ...
        , 'Color', [.5 .5 .5] ...
        );
    
    %Plot regression line
    plot( x_vals, y_hat ...
        , 'Color', plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:) ...
        , 'LineWidth',2 ...
        );
    
    set(gca ...
        , 'Fontname','Arial' ...
        , 'Fontsize',graph_font ...
        , 'FontWeight','normal' ...
        , 'YTick',[0:2:Generate_params.seq_length] ...
        , 'XTick',[0:2:Generate_params.seq_length] ...
        , 'LineWidth',2 ...
        );
    
    ylim([0 Generate_params.seq_length+1]);
    xlim([0 Generate_params.seq_length]);
    ylabel('Model sampling');
    xlabel('Participant sampling');
    
    title( ...
        sprintf('%s',Generate_params.model(identifier ).name) ...
        , 'Fontname','Arial' ...
        , 'Fontsize',graph_font ...
        , 'FontWeight','normal' ...
        );
    
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Analysis of thresholds!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if Generate_params.analyze_value_positions == 1;
    
%     analyze_value_position_functions(raw_seqs_subs,all_choice_trial,plot_cmap,binEdges_psi,model_strs,param_to_fit,two_params);
    
    nbins_psi = Generate_params.nbins_psi;
    binEdges_psi(1:Generate_params.num_subs,:) = ...
        repmat(...
        linspace(...
        Generate_params.rating_bounds(1) ...
        ,Generate_params.rating_bounds(2) ...
        ,nbins_psi+1 ...
        ), ...
        numel(1:Generate_params.num_subs),1 ...
        );
    
    analyze_value_position_functions(...
        Generate_params.seq_vals ...
        ,all_choice_trial ...
        ,plot_cmap ...
        ,binEdges_psi ...
        ,model_strs ...
        ,Generate_params.do_models_identifiers ...
        ,1 ...
        , Generate_params.model_bar_order ...
        ,Generate_params.BayesThresh ...
        );
    
end;    %make threshold by serial position plot



