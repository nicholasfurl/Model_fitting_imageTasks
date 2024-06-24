
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = imageTasks_figures_CP_revisions;

%This is the right one to use. *v2 is for runs w/o BR. I ended up
%reincluding BR again so back to the original version.

%Based on hybrid_figures_CP_revisions_v4.m, developed to make figures for
%the communications psychology submission version of the hybrid paper. One
%major change is that there are no OV models in this paper.

%adapted from hybrid_subj_vals_model_comparison_NEW.

warning('off','all');

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\plotSpread'));
addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\klabhub-bayesFactor-3d1e8a5'));

%Order of bars. These are the indices in the original structs placed in the
%order you want them in the new structs
%1 CO, 2 Cs, 3 Opt, 4 BV, 5 BR, 6 optimal
%Orders will be duplicated across objective and subjective values, if both exist
%Numbers mean where I want a certain model index to appear in new array. So
%putting a 6 first mean the first model in the input struct will be optimal
%zero means participants.
%bar_order = [6 0 1:5];
bar_order = [6 0 1 2 4 5 3]; 
% bar_order = [5 0 1:4];


%Master control switch! Which figure do I want to make! 
%It's possible paper figure nums may change later but in THIS code:
figure_num = 3;

%output file name decoder:
%study1: av faces 1, 2: full pilot, 3: baseline, 4: full, 5: ratings phase, 6: squares 7: timing 8:payoff
%pay1: continuous reward (rewarded by option value of choice), pay2: 5/3/1 ratio for top three ranks (for stars). pay3: monetary ratio for top three ranks
%val0: objective values, val1: subjective values
outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\outputs';
file_paths = {...
    [outpath filesep 'out_imageTask_face_1_newerBVBR_20240502.mat']
    [outpath filesep 'out_imageTask_face_2_newerBVBR_20240502.mat']
    [outpath filesep 'out_imageTask_face_3_newerBVBR_20240502.mat']
    [outpath filesep 'out_imageTask_matchmaker_newerBVBR_20240502.mat']
    [outpath filesep 'out_imageTask_trust_1_newerBVBR_20240502.mat']
    [outpath filesep 'out_imageTask_trust_2_newerBVBR_20240602.mat']
    [outpath filesep 'out_imageTask_food_1_newerBVBR_20240602.mat']
    [outpath filesep 'out_imageTask_food_2_newerBVBR_20240602.mat']
    [outpath filesep 'out_imageTask_holiday_1_newerBVBR_20240602.mat']
    [outpath filesep 'out_imageTask_holiday_2_newerBVBR_20240702.mat']
    };

%7 Feb 2024
%     [outpath filesep 'out_imageTask_face_1_20240302.mat']
%     [outpath filesep 'out_imageTask_face_2_20240302.mat']
%     [outpath filesep 'out_imageTask_face_3_20240402.mat']
%     [outpath filesep 'out_imageTask_matchmaker_20240402.mat']
%     [outpath filesep 'out_imageTask_trust_1_20240402.mat']
%     [outpath filesep 'out_imageTask_trust_2_20240402.mat']
%     [outpath filesep 'out_imageTask_food_1_20240402.mat']
%     [outpath filesep 'out_imageTask_food_2_20240402.mat']
%     [outpath filesep 'out_imageTask_holiday_1_20240402.mat']
%     [outpath filesep 'out_imageTask_holiday_2_20240502.mat']

% 5 Feb 2024
%     [outpath filesep 'out_new_ll1_imageTask_matlab_study01_20231510.mat']
%     [outpath filesep 'out_imageTask_gorilla_study05_20231410.mat']
%     [outpath filesep 'out_imageTask_gorilla_study08_20231510.mat']
%     [outpath filesep 'out_new_ll1_imageTask_matlab_study02_20231610.mat']
%     [outpath filesep 'out_new_ll1_imageTask_matlab_study03_20231810.mat']
%     [outpath filesep 'out_new_ll1_bigtrustLog_20231910.mat']
%     [outpath filesep 'out_new_ll1_foodsonlineSubsLog0_20213107.mat']
%     [outpath filesep 'out_new_ll1_foodsOpendaySubsLog0_20210208.mat']
%     [outpath filesep 'out_new_ll1_holsOnlineSubsLog0_20210308.mat']
%     [outpath filesep 'out_new_ll1_holsOpendaySubsLog0_20210408.mat']

%     [outpath filesep 'out_new_ll1_av_20212906.mat']

study_names = {...
    'Faces study 1'
    'Faces study 2'
    'Faces study 3'
    'Matchmaker study'
    'Trust study 1'
    'Trust study 2'
    'Foods study 1'
    'Foods study 2'
    'Holidays study 1'
    'Holidays study 2'
    };
study_nums = [1 2 3 4 5 6 7 8 9 10];

%w/o any IO in model comparison
if figure_num == 1;
    
    data{1}{1} = load(file_paths{1});
    data{2}{1} = load(file_paths{2});
    data{3}{1} = load(file_paths{3});
    
elseif figure_num == 2;
    
    data{1}{1} = load(file_paths{4});
    data{2}{1} = load(file_paths{5});
    data{3}{1} = load(file_paths{6});
    
    
elseif figure_num == 3;
    
    data{1}{1} = load(file_paths{7});
    data{2}{1} = load(file_paths{8});
    data{3}{1} = load(file_paths{9});
    data{4}{1} = load(file_paths{10});
    
end;    %which fig?

%Some figures involve more than one study in each figure (right now,
%setting each study as a column)
num_studies = size(data,2);

for study = 1:num_studies;
    
%     %SV & OV? Or just OV?
%     num_datasets = size(data{study},2);
   
%     for dataset = 1:num_datasets;
        
        %Takes list of file paths and then concatenates the data for the models.
        %Different from combineModels, which treats the same models from different files as
        %though they were different.
        combined_data{1} = reformatData(data{study}{1});
        
%     end;    %loop through datasets
    
    %This bit makes a new structure, but now the models in each input struct
    %are treated as separate models in the output struct, even if they have
    %the same names (e.g., so SV Cs and OV Cs are added as separate models)
    New_struct.study(study) = combineModels(combined_data, bar_order);
    
end;    %loop through studies

%update it with handy info that is not study-specific (top-level info)
New_struct.IC = 2; %1=AIC, 2=BIC (They should all be one parameter models anyway)
New_struct.figure = figure_num;
New_struct.bar_order = bar_order;
%New_struct.num_panels = num_panels;
New_struct.num_studies = num_studies;

%Make this figure
plot_data(New_struct);

disp('audi5000');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function heatmap_struct = add_sig_to_behav_plot(h1,this_bar_num,data_accum, BayesThresh, labels)

figure(h1)
subplot(2,1,1); hold on;

start_bar = this_bar_num - size(data_accum,2) -2;

%Horizontal "Significance" connector lines
pairs = nchoosek(1:size(data_accum,2),2);   %get pairs of conditions
pairs= sortrows(pairs(:,[2 1]),'descend');
num_pairs = size(pairs,1);
line_pair_order = pairs;


max_y = max(cellfun(@max,data_accum));
y_inc = .03*max_y;
ystart = max_y + y_inc*num_pairs;
line_y_values = ystart:-y_inc:0;

%initalise matrix for heatmap
heatmap_struct.heatmap_mat = NaN(size(data_accum,2));

for pair = 1:num_pairs;
    
    %trad t-test
    [h tpvals(pair) ci stats] =   ...      
        ttest2( data_accum{line_pair_order(pair,1)}, data_accum{line_pair_order(pair,2)} );

    %Bayes Factors
    [bf10(pair),pvals(pair),ci,stats] = ...
        bf.ttest2( data_accum{line_pair_order(pair,1)}, data_accum{line_pair_order(pair,2)} );

     if tpvals(pair) < .05/num_pairs;
        
         plot([line_pair_order(pair,1)+start_bar line_pair_order(pair,2)+start_bar],...
            [line_y_values(pair) line_y_values(pair)],'LineWidth',1.5,'Color',[0 1 0]);
        
    end;
    
%     BayesThresh = 10;
    if bf10(pair) < (1/BayesThresh);
        plot([line_pair_order(pair,1)+start_bar line_pair_order(pair,2)+start_bar],...
            [line_y_values(pair) line_y_values(pair)],'LineWidth',1.5,'Color',[1 0 1]);
    end;
    if bf10(pair) > BayesThresh;
        
        plot([line_pair_order(pair,1)+start_bar line_pair_order(pair,2)+start_bar],...
            [line_y_values(pair) line_y_values(pair)],'LineWidth',.5,'Color',[0 1 0]);
 
    end;  %"significant"?
    
        %Not very elegent, but I added this later and don't want to disturn
        %existing code. I'm computing Cohen's d from the pairs and then
        %assembling them into matrix to be made into heatmap, to respond to
        %reviewer request for effect sizes.
        N1 = numel(data_accum{line_pair_order(pair,1)});
        N2 = numel(data_accum{line_pair_order(pair,2)});
        heatmap_struct.heatmap_mat(line_pair_order(pair,1),line_pair_order(pair,2)) = ...
            stats.tstat*sqrt((N1+N2)/(N1*N2));
        heatmap_struct.labels = labels;
        
end;    %loop through pairs for Bayesian pairwise tests

fprintf('');
%%%%%%%%%%%%%%%%add_sig_to_behav_plot%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = make_behaviour_plot(data);

f_a = 0.1; %face alpha
sw = 1;  %point spread width
graph_font = 9;
x_axis_test_offset = .03;   %What percentage of the y axis range should x labels be shifted below the x axis?
x_rot = 45;

BayesThresh = 3;

h1 = figure('Color',[1 1 1],'Position',[ 395.4000  101.0000  590.4000  667.2000]);
subplot(2,1,1); hold on

num_studies = size(data,2);

this_bar_num = 0;
study_num = 1;  %used to space the bars and organise sig connectors. Make a space when new study num
study_counter = 1;  %count studies with the same study number
num_heatmaps = 1;
for study = 1:num_studies;
 
    y_string = 'Samples to decision';
    these_data = nanmean(data{study}.Generate_params.num_samples)';
    plot_color = [0 0 0];
    study_label = data{study}.Generate_params.study_name;
    
    %Make faint guidlines for sig connector lines
    plot([this_bar_num this_bar_num],[nanmean(these_data) 8000],'Color',[.9 .9 .9], 'LineWidth',.25);

    %average over sequences (rows) but keep sub data (cols) for scatter points
    handles = plotSpread(these_data ...
        ,'xValues',this_bar_num ...
        ,'distributionColors',plot_color ...
        ,'distributionMarkers','.' ...
        , 'spreadWidth', sw ...
        );
    
    bar(this_bar_num,nanmean(these_data) ...
        ,'FaceColor',plot_color ...
        ,'FaceAlpha',f_a ...
        ,'EdgeColor',[0 0 0] ...
        );
    
    set(gca ...
        ,'XTick',[] ...
        ,'fontSize',graph_font ...
        ,'FontName','Arial'...
        ,'YTick',[0:2: 10] ...
        ,'YLim', [0 10] ...    %8 is the largest sequence length out of the studies
);
    ylabel(y_string);


    
    this_offset = -x_axis_test_offset*diff(ylim);
    text( this_bar_num, this_offset ...
        ,sprintf('%s',study_label) ...
        ,'Color', [0 0 0] ...
        ,'Fontname','Arial' ...
        ,'Fontsize',graph_font ...
        ,'Rotation',x_rot ...
        ,'HorizontalAlignment','right' ...
        );
    
    %Run pairwise analysis whenever the next study_num differs from the
    %current one or if its the last stay_num.
    
    study_num = data{study}.Generate_params.study_num;
    data_accum{study_num}{study_counter} = these_data;   %Need for significance bars, below ..
    data_labels_accum{study_num}{study_counter} = study_label;
    
    if (study == num_studies) | ...
            ((data{study+1}.Generate_params.study_num ~= study_num));
        
        %setup for plotting next study nums after a space
        this_bar_num = this_bar_num + 2;    %add extra space on X axis
        study_counter = 1;  %reset study counter
        
        if (size(data_accum{study_num},2) > 1);    %If the next one is the same as this one, carry on
            
            heatmap_struct(num_heatmaps) = add_sig_to_behav_plot(h1,this_bar_num,data_accum{study_num}, BayesThresh, data_labels_accum{study_num});
            num_heatmaps = num_heatmaps+1;
        end;
        
    else;  %conditions in which we trigger pairwise analysis
        
        this_bar_num = this_bar_num + 1;
        study_counter = study_counter + 1;
        
    end
  
end;    %loop through conditions / studies

%Now that you've finished plotting the bars, plot the heatmaps in subplot below bars

% figure(h1);
subplot(2,1,2);

for map = 1:numel(heatmap_struct);
    
    if numel(heatmap_struct(map).heatmap_mat) > 2*2;    %If there's more than one pair (two conditions) and a proper matrix to plot (i.e., Study 1)
        
%         h_hm = figure;
        %     subplot(2,1,2);
        hm_handle = heatmap( ...
            heatmap_struct(map).labels ...
            ,heatmap_struct(map).labels ...
            ,heatmap_struct(map).heatmap_mat ...
            );
        hm_handle.CellLabelFormat = '%2.2f';
        
    else;   %Otherwise, if only two conditions; intended for Study 3 sequence length comparison
        
        disp( ...
            sprintf( ...
            'Cohens d for %s versus %s is %2.2f',  ...
            heatmap_struct(map).labels{1} ...
            , heatmap_struct(map).labels{2} ...
            , heatmap_struct(map).heatmap_mat(2,1) ...
            ));
        
    end;    %Study 1 or Study 3 heatmap?
    
end;    %loop through saved heatmaps
%%%%%%%%%%%%%%%%%%%make_behaviour_plot%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%plot_samples_by_sequence%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_samples_by_sequence(plot_details)

figure(plot_details.fig);
subplot(1,plot_details.cols,plot_details.study);

num_seqs = size(plot_details.these_data,1);
Xs = [1:num_seqs];
N = size(plot_details.these_data,2);
means = mean(plot_details.these_data');
SE = std(plot_details.these_data')/sqrt(N);
CI95 = tinv([0.025 0.975], N-1);                    % Calculate 95% Probability Intervals Of t-Distribution
yCI95 = bsxfun(@times, SE, CI95(:));

plot(means,'Marker','o','MarkerSize',8,'MarkerFaceColor',[0 0 0],'Color',[0 0 0]);
hold on;
for seq=1:num_seqs;
    plot( ...
        [Xs(seq) Xs(seq)],...
        [means(seq) - yCI95(2,seq) means(seq) + yCI95(2,seq)], ...
        'Color',[0 0 0] ...
        ); 
end;
xlim([0 num_seqs+1]);
ylim([0 plot_details.seq_length]);
box off;
%axis square;
ylabel('Samples to decision');
xlabel('Sequence number');
xticks(Xs);
%%%%%%%%%%%%%%%%%%%plot_samples_by_sequence%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%
%%%%Taken from new_fit_code_hybrid_prior_subs_v2.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = plot_data(Generate_params);

%set up plot appearance
%For now I'll try to match model identifiers to colors. Which means this
%colormap needs to scale to the total possible number of models, not the
%number of models
% num_models_per_study = numel(Generate_params.bar_order);
% num_models_per_study = numel(Generate_params.study(1).model);
BayesThresh = 3;

%io first, participants next, theoertical fitted models next
temp = hsv(9);
plot_details.plot_cmap = [0 0 0; .75 .75 .75; temp(2:end,:)];  %supports up to eight model identifiers (not models per se) plus optimal (identifier 0) plus participants. Probably will use seven identifiers plus optimal plus participants
% plot_cmap = hsv(num_models_per_study+1);  %models + subjects
plot_details.f_a = 0.1; %face alpha
plot_details.sw = 1;  %ppoint spread width
plot_details.graph_font = 9;
plot_details.x_axis_test_offset = .05;   %What percentage of the y axis range should x labels be shifted below the x axis?
plot_details.x_rot = 65;
plot_details.num_panels = 3; %This means samples, parameters, BIC, frequencies

%Should allow easier subplot panel rearrangement by controling these
%studies row-wise
% rows = Generate_params.num_studies;  %subplots
% cols = Generate_params.num_panels; %subplots
% subplot_num = "panel_num+((study-1)*cols)" ;    %Will need to use eval, as study and panel_num don't exist yet
%studies colwise
plot_details.panel_nums = [1 2 3];   %fixes the subplot locations of the samples, BIC and model win freq (indices 1,2,3 into panel_nums)
plot_details.rows = plot_details.num_panels;  %subplots
plot_details.cols = Generate_params.num_studies; %subplots
plot_details.subplot_num = "((plot_details.panel_num-1)*plot_details.cols)+study" ;    %Will need to use eval, as study and panel_num don't exist yet

%Samples, BIC & frequency
h10 = figure('Color',[1 1 1],'Name','samples');

%ranks and parameter values (for supplementary information)
h11 = figure('Color',[1 1 1],'Name','ranks');

%samples by sequence number
h_sequences = figure('Color',[1 1 1],'Name','samples by sequence');

%BIC effect size heatmaps
h_BIChm = figure('Color',[1 1 1],'Name','BIC effect sizes');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%SAMPLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for study = 1:Generate_params.num_studies;
    
    %create figure handles for samples scatterplots
    h_scatter(study) = figure('Color',[1 1 1],'Name',sprintf('Study %d',study));
    
    plot_details.num_models_per_study = numel(Generate_params.study(study).model);
    plot_details.study = study;
    
    clear IC_pps_all_models samples_accum;
    human_is_index = [];    %Used for plotting significance connector lines on samples plot, because I need to know which bar is humans for significance comparison
    it_model_comparison = 1;
    
    for this_bar = 1:plot_details.num_models_per_study;
        
        plot_details.this_bar = this_bar;
        identifier = Generate_params.study(study).model(this_bar).identifier;
        
        %plot samples on one figure
        plot_details.fig = h10;
        plot_details.y_string = 'Samples to decision';
        plot_details.these_data = Generate_params.study(study).model(this_bar).num_samples_est;
        %keep data for use later on when plotting significance connector lines
        samples_accum(:,this_bar) = plot_details.these_data;
        labels_accum{this_bar} = Generate_params.study(study).model(this_bar).name;  %Need later for outputting effect sizes (Cohen's d)
        plot_a_bar(plot_details,Generate_params);
        
        %plot ranks on the supplementary figure
        plot_details.fig = h11;
        plot_details.y_string = 'Rank of chosen price';
        plot_details.these_data = Generate_params.study(study).model(this_bar).ranks;
        plot_a_bar(plot_details,Generate_params);
        
        %         %keep data for use later on when plotting significance connector lines
        if Generate_params.study(study).model(this_bar).identifier == 1;    %if participants
            
            %Will need this later to identify model data/locations
            human_is_index = this_bar;
            plot_details.human_is_index = human_is_index;
            
            %Can use this if/then to make plot of samples by sequence position
            plot_details.fig = h_sequences;
            plot_details.these_data = Generate_params.study(study).samples_by_sequence;
            plot_details.seq_length = Generate_params.study(study).seq_length;
            plot_samples_by_sequence(plot_details);
            
        end;
        
        

        %After this skip filter, we do plots that only involve models and
        %not participants like BIC and parameter plots
        if Generate_params.study(study).model(this_bar).skip == 0; %If a theoretical model and not participants or IO
            
            %%%%%%%%%%%%%%%%%%%%%%%%
            %AIC/BIC plots

            %Compute AIC ior BIC from ll and number of params
            no_params = numel( Generate_params.study(study).model(this_bar).this_models_free_parameters ) + 1; %+1 for beta
            lla = Generate_params.study(study).model(this_bar).ll;
            if Generate_params.IC == 1; %If AIC (per participant)
                IC_pps = 2*no_params + 2*lla;
                a_label = 'AIC';
            elseif Generate_params.IC == 2; %If BIC (per participant)
                sum_draws = Generate_params.study(study).num_samples.*Generate_params.study(study).num_seqs;
                IC_pps = no_params.*sum_draws + 2*lla;
%                 IC_pps = no_params*log(Generate_params.study(study).num_seqs) + 2*lla;
                a_label = 'Bayesian information criterion';
            end;
            
            %plot AIC/BIC data
            IC_pps_all_models(:,it_model_comparison) = IC_pps'; %We need to accumulate these data over models in this loop to do the next step more easily
            IC_labels_all_models(:,it_model_comparison) = Generate_params.study(study).model(this_bar).name; %Accumulate model names too so we can use them as labels on the BIC effect size heat map
            plot_details.these_data = IC_pps;
            plot_details.fig = h10;
            plot_details.panel_num = plot_details.panel_nums(2);
            plot_details.it_model_comparison = it_model_comparison;
            plot_details.plot_color = plot_details.plot_cmap(identifier,:);
            plot_details.y_string = a_label;
            plot_a_model(plot_details,Generate_params);
            

            %plot first parameter values (the cut off, Cs, etc.)
            plot_details.these_data = Generate_params.study(study).model(this_bar).estimated_params(:,1);
            plot_details.fig = h11;
            plot_details.panel_num = plot_details.panel_nums(2);
            plot_details.y_string = 'First parameter';
            plot_a_model(plot_details,Generate_params);
            
            %plot second parameter values (beta)
            plot_details.these_data = Generate_params.study(study).model(this_bar).estimated_params(:,2);
            plot_details.fig = h11;
            plot_details.panel_num = plot_details.panel_nums(3);
            plot_details.y_string = 'Second parameter (beta)';
            plot_a_model(plot_details,Generate_params);
            
            %Plot scatterplot for this model samples against human samples
            plot_details.these_data = Generate_params.study(study).model(this_bar).num_samples_est;
            plot_details.fig = h_scatter(study);
            plot_a_scatterplot(plot_details,Generate_params);
            
            %prepare for next iteration
            it_model_comparison = it_model_comparison + 1;
            
        end;    %If a theoretical model (i.e., skip == 0)
        
    end;    %loop through models
    
    
    
    
    
    
    
    %%%%%%%%%This is the finish of the loops that add bars to the samples
    %%%%%%%%%and BIC plots. During this loop, we would have accumulated
    %%%%%%%%%data from those bars into matrixs that, below, we will use in
    %%%%%%%%%new loops to add pairwise test bars to plots and make model
    %%%%%%%%%"win" frequency plots
    
    
    %significance lines
%     if num_models_per_study ~= 1;
        
        %%%%%%%%%%%%%%%%%%
        %Samples significance lines
        
        %Connecting just
        %%%participants with all models,
        %%%but no need for models with each other
        
        %return to samples subplot
        figure(h10);
        plot_details.panel_num = plot_details.panel_nums(1);  %Panels nums can be changed colwise versus rowwise in subplot
        subplot(plot_details.rows,plot_details.cols,eval(plot_details.subplot_num));
        
        %get the pairs connecting participants to models and set up their y axis locations
        num_pairs = size(samples_accum,2);  %This means the number of pairs is just the number of cols in the samples matrix (minus the first)
        y_inc = .45;
        
        %OK, seems weird, but I'm fixing the maximum height of the Y axis
        %to 14 samples, which is the largest sequence length across all the
        %studies (Seq study 2) so that all bars are on the same scale and
        %can be compared across studies. But I'll still number the X ticks
        %differently depending on sequence length.
        %         ystart = max(max(samples_accum(:,1:human_is_index))) + y_inc*human_is_index +y_inc;
        %         ystart = Generate_params.study(study).seq_length + y_inc*human_is_index +y_inc;
%         ystart = 14.5 + y_inc*human_is_index +y_inc;
        ystart = 9 + y_inc*3;

        %     line_y_values = ystart:-y_inc:0;
        %line_y_values = (0:y_inc:ystart) + (max(max(samples_accum)) + 6*y_inc);
        line_y_values = (0:y_inc:ystart) + (Generate_params.study(study).seq_length + 4*y_inc);

        
        %loop through pairs, get "significance" of each, plot them
        %         for pair = num_pairs:-1:1;
        heatmap_mat = NaN(human_is_index);
        for pair = human_is_index-1:-1:1;
            
            
            [bf10(pair),samples_pvals(pair),ci,stats] = ...
                bf.ttest( samples_accum(:,human_is_index) - samples_accum(:,pair) );
            
            %             bf.ttest( samples_accum(:,1) - samples_accum(:,num_pairs + 2 - pair) );
            
            %distance on plot
            %         distance_on_plot = [1 num_pairs + 2 - pair];
            distance_on_plot = [human_is_index pair];
            
            %             if samples_pvals(pair) < 0.05/num_pairs;
            %                 plot(distance_on_plot,...
            %                     [line_y_values(pair) line_y_values(pair)],'LineWidth',.5,'Color',[0 0 0]);
            %             end;
            
            if bf10(pair) < (1/BayesThresh);
                plot(distance_on_plot,...
                    [line_y_values(pair) line_y_values(pair)],'LineWidth',1,'Color',[0 0 0]);   %black lines mean equivalent means
            end;
            if bf10(pair) > BayesThresh;
                plot(distance_on_plot,...
                    [line_y_values(pair) line_y_values(pair)],'LineWidth',2.5,'Color',[0.5 0.5 0.5]);   %grey lines mean significant differences
            end;
            
            %Not very elegent, but I added this later and don't want to disturn
        %existing code. I'm computing Cohen's d from the pairs and then
        %assembling them into matrix to be made into heatmap, to respond to
        %reviewer request for effect sizes.
%         N1 = numel(samples_accum{line_pair_order(pair,1)});
%         N2 = numel(samples_accum{line_pair_order(pair,2)});
        %r_rm = corr(samples_accum(:,human_is_index),samples_accum(:,pair));
        s_diff = nanstd(samples_accum(:,human_is_index) - samples_accum(:,pair));
%         m_diff = (mean(samples_accum(:,human_is_index)) - mean(samples_accum(:,pair)));
        m_diff = nanmean(samples_accum(:,human_is_index) - samples_accum(:,pair));
        cohens_d = m_diff/s_diff;
%         heatmap_struct.heatmap_mat(human_is_index,pair) = ...
%             m_diff/ ...
%             (s_diff/sqrt(2*(1-r_rm)));
        disp(sprintf('Study %d Cohens d effect size for %s versus %s is %2.2f', study, labels_accum{human_is_index}, labels_accum{pair},cohens_d));
            
            
        end;   %pairs: Loop through comparisons between participants and models
        %Set Y lim to 14 + space for sig bars but y ticks to count the real
        %sequene length.
        set(gca,'Ylim',[0 ystart],'YTick',[0:2:Generate_params.study(study).seq_length]); box off;
        
        
                
        
        
        
        
        
        
        %%%%%%%%%%%%%%%%%%
        %BIC significance lines
        %run and plot ttests on IC averages
        
        %return to BIC subplot
        plot_details.panel_num = plot_details.panel_nums(2);  %Panels nums can be changed colwise versus rowwise in subplot
        subplot(plot_details.rows,plot_details.cols,eval(plot_details.subplot_num));
        
        pairs = nchoosek(1:size(IC_pps_all_models,2),2);
        pairs= sortrows(pairs(:,[2 1]),'descend');
        num_pairs = size(pairs,1);
        %         [a In] = sort(diff(pairs')','descend');  %lengths of connecting lines
        %         line_pair_order = pairs(In,:);    %move longest connections to top
        line_pair_order = pairs;    %move longest connections to top
        
        
        %         %Where to put top line?
        %         y_inc = 2;
        %         ystart = max(max(IC_pps_all_models)) + y_inc*num_pairs;
        %         line_y_values = ystart:-y_inc:0;
        %Where to put top line?
        max_y = max(max(IC_pps_all_models));
        y_inc = .065*max_y;
        ystart = max_y + y_inc*num_pairs;
        line_y_values = ystart:-y_inc:0;
        
        %to store and display effect sizes (Cohen's d) for pairwise comparisons
        heatmap_mat = NaN(size(IC_pps_all_models,2));
        
        for pair = 1:num_pairs;
            
            %run ttest this pair
%             [h IC_pp_pvals(pair) ci stats] = ttest(IC_pps_all_models(:,line_pair_order(pair,1)), IC_pps_all_models(:,line_pair_order(pair,2)));
% 
%              if IC_pp_pvals(pair) < 0.05/size(pairs,1);  %multiple comparison corrected
        %
        %
        %                 %%find identifier of the rightmost model in the pair
        %                 rightmost_num = line_pair_order(pair,2);    %This is an index into IC_pps_all_models which excludes IO and Human
        %                 rightmost_id = Generate_params.study(study).model(line_pair_order(pair,2)+human_is_index).identifier;
        %                 rightmost_color = plot_cmap(rightmost_id,:);    %color associated with identifer of rightmost model in pairwise comparison
        %
        %
        %                 plot([line_pair_order(pair,1) line_pair_order(pair,2)],...
        %                     [line_y_values(pair) line_y_values(pair)],'LineWidth',.5,'Color',rightmost_color);
        %                 %                                 [line_y_values(pair) line_y_values(pair)],'LineWidth',.5,'Color',[0 0 0]);
        %
        %             end;    %Do line on plot?;
            
            
            
            %%%%%Bayes Factors
            diffs = IC_pps_all_models(:,line_pair_order(pair,1)) ...
                - IC_pps_all_models(:,line_pair_order(pair,2));
            [bf10(pair),IC_pp_pvals(pair),ci,stats] = ...
                bf.ttest( diffs );
            
            yticks = linspace(0, ceil(max_y/20)*20,5);
            set(gca,'Ylim',[0 ystart],'YTick',yticks);
            
            if bf10(pair) < (1/BayesThresh);
                plot([line_pair_order(pair,1) line_pair_order(pair,2)],...
                    [line_y_values(pair) line_y_values(pair)],'LineWidth',1.0,'Color',[0 0 0]);
            end;
            if bf10(pair) > BayesThresh;
                
                %colour sig line according to which model has the smaller BIC, if significant
                %(code used to pick out rightmost but is repurposed, so
                %rightmost means winning model now)
                if sum(diffs) < 0; pair_index = 1; else; pair_index = 2; end;
                rightmost_num = line_pair_order(pair,pair_index);    %This is an index into IC_pps_all_models, which excludes IO and Human
%                 rightmost_id = Generate_params.study(study).model(line_pair_order(pair,2)+human_is_index).identifier;
                rightmost_id = Generate_params.study(study).model(rightmost_num+human_is_index).identifier;
                rightmost_color = plot_details.plot_cmap(rightmost_id,:);    %color associated with identifer of winning in pairwise comparison
                
                plot([line_pair_order(pair,1) line_pair_order(pair,2)],...
                    [line_y_values(pair) line_y_values(pair)],'LineWidth',.5,'Color',rightmost_color);
                %                     [line_y_values(pair) line_y_values(pair)],'LineWidth',.5,'Color',[0 1 0]); %plot "significant" connector lines in a color like green
            end;  %"significant"?
            
            %Compute paired Cohen's d for all possible pairs, significant
            %or not, and store them in square matrix for displaying heatmap
%             if line_pair_order(pair,1) == 7 & line_pair_order(pair,2) == 5;
%                 fprintf('');
%             end;
            if sum(diffs) == 0;  %A couple times BR and BV seem to make identical predictions
                heatmap_mat(line_pair_order(pair,1), line_pair_order(pair,2)) = 0;
            else;
                heatmap_mat(line_pair_order(pair,1), line_pair_order(pair,2)) = ...
                    nanmean(diffs)/nanstd(diffs);
            end;
   
        end;    %loop through pairs for Bayesian pairwise tests

    %This works better here, despite the inelegent extra loop, because the
    %y lim has been fixed for certain by this point so I can safetly make
    %the distance of the text from the X axis be dependent on the Y limit
    %(e.g., the x axis labels won't end up with variable diustances as the
    %plot evolves over model iterations).
    for model = 1:size(IC_pps_all_models,2)
        
        this_offset = -plot_details.x_axis_test_offset*diff(ylim);
        text( model, this_offset ...
            ,sprintf('%s',Generate_params.study(study).model(model+human_is_index).name) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',plot_details.graph_font ...
            ,'Rotation',plot_details.x_rot ...
            ,'HorizontalAlignment','right' ...
            );
        
    end;    %loop through models to add x tick labels to BIC plot
    box off;
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%
    %Plot of numbers of winning subs for each model
    
    %     subplot(Generate_params.num_panels,Generate_params.num_studies,subplot_indices(4,study));
    %     subplot(2,3,6);
    plot_details.panel_num = plot_details.panel_nums(3);  %Panels nums can be changed colwise versus rowwise in subplot
    subplot(plot_details.rows,plot_details.cols,eval(plot_details.subplot_num));
    hold on; box off;
    
    %winning models
    [a pps_indices] = min(IC_pps_all_models');
    
    model_it = 0;
    for model = 1:plot_details.num_models_per_study-human_is_index;
        
        
        bar(model,numel(find(pps_indices==model)), ...
            'FaceColor', plot_details.plot_cmap(Generate_params.study(study).model(model+human_is_index).identifier,:),'FaceAlpha',plot_details.f_a+.2,'EdgeColor',[0 0 0] );
        %             'FaceColor', plot_cmap(Generate_params.study(study).model(model).identifier,:),'FaceAlpha',f_a,'EdgeColor',[0 0 0] );
        
        set(gca ...
            ,'XTick',[] ...
            ,'fontSize',plot_details.graph_font ...
            ,'FontName','Arial',...
            'XLim',[0 plot_details.num_models_per_study-(human_is_index-1)] ...
            );
        ylabel('Frequency');
        
    end;    %models
    
    %Like above, it's embarrassing inelegent to repeat a loop just
    %completed, but I need the max frequency and therefore the ylim to be
    %finalised or I'll end up with different diustances from the x axis here
    for model = 1:plot_details.num_models_per_study-human_is_index;
        
        this_offset = -plot_details.x_axis_test_offset*diff(ylim);
        text( model, this_offset ...
            ,sprintf('%s',Generate_params.study(study).model(model+human_is_index).name) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',plot_details.graph_font ...
            ,'Rotation',plot_details.x_rot ...
            ,'HorizontalAlignment','right' ...
            );
        
    end;
    
    %Make new figure for heatmaps and display them
    fprintf('');
    figure(h_BIChm);
    subplot(1,plot_details.cols,study);
    
    hm_handle = heatmap( ...
        IC_labels_all_models, ...
        IC_labels_all_models, ...
        heatmap_mat ...
        );
    hm_handle.CellLabelFormat = '%2.2f';
    
    
end;    %loop through studies
%%%%%%%%%%%%plot_data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_a_scatterplot(plot_details,Generate_params);

%Adapted from new_fit_code_make_plots_2022_v3.m

study = plot_details.study;
num_studies = size(Generate_params.study,2);
these_data = plot_details.these_data;
participant_data = Generate_params.study(study).num_samples;
num_participants = numel(participant_data);
seq_length = Generate_params.study(study).seq_length;
current_model = plot_details.it_model_comparison;
num_models = size(Generate_params.study(study).model,2)-plot_details.human_is_index;
% y_string = sprintf('%s sampling',Generate_params.study(study).model(plot_details.this_bar).name);
y_string = sprintf('%s sampling',Generate_params.study(study).model(plot_details.this_bar).name);
graph_font = plot_details.graph_font;

num_cols = floor(sqrt(num_models) );
num_rows = ceil(num_models/num_cols);
% 
% num_cols = num_studies;
% num_rows = num_models;

figure(plot_details.fig)
% subplot(num_rows,num_cols,study+(current_model-1)*num_studies);
subplot(num_rows,num_cols,plot_details.it_model_comparison);
hold on;
axis square;

scatter( ...
    participant_data ...
    , these_data ...
    , 36 ... %marker size apparently
    , 'MarkerEdgeColor', plot_details.plot_color ...
    ,'MarkerEdgeAlpha', .3 ...
    ,'LineWidth',1 ...
    );

%regression line
[b,bint,r,rint,stats] = regress( ...
   these_data ...
    , [ones(num_participants,1) participant_data] ...
    );
x_vals = [0 seq_length];

y_hat = b(1) + b(2)*x_vals;

%Put Rsquared on plot
text( 0.5, seq_length +.5 ...
    , sprintf('R squared = %0.2f',stats(1)) ...
    , 'Fontname','Arial' ...
    , 'Fontsize',graph_font ...
    , 'FontWeight','normal' ...
    );


%Plot a diagonal of inequality too
plot( ...
    [0 seq_length] ...
    , [0 seq_length] ...
    , 'Color', [.5 .5 .5] ...
    );


%Plot regression line
plot( x_vals, y_hat ...
    , 'Color', plot_details.plot_color ...
    , 'LineWidth',.5 ...
    );

set(gca ...
    , 'Fontname','Arial' ...
    , 'Fontsize',graph_font ...
    , 'FontWeight','normal' ...
    , 'YTick',[0:2:seq_length] ...
    , 'XTick',[0:2:seq_length] ...
    , 'LineWidth',2 ...
    );

ylim([0 seq_length+1]);
xlim([0 seq_length]);
ylabel(y_string);
xlabel('Human sampling');

% title( ...
%     sprintf('%s',Generate_params.study(study).model(plot_details.this_bar).name) ...
%     , 'Fontname','Arial' ...
%     , 'Fontsize',plot_details.graph_font ...
%     , 'FontWeight','normal' ...
%     );

box off;

%%%%%%%%%%%%%%%plot_a_scatterplot%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Use to plot samples and ranks inside loop through models, without repeating code
function plot_a_model(plot_details, Generate_params);

study = plot_details.study;

figure(plot_details.fig)
subplot(plot_details.rows,plot_details.cols,eval(plot_details.subplot_num));

%If the BIC plot, then add faint grey guidelines
if strcmp(plot_details.fig.Name,'samples') & plot_details.panel_num == 2;
    plot([plot_details.it_model_comparison plot_details.it_model_comparison],[nanmean(plot_details.these_data) 8000],'Color',[.9 .9 .9], 'LineWidth',.25);
end;


handles = plotSpread(plot_details.these_data ...
    , 'xValues',plot_details.it_model_comparison...
    ,'distributionColors',plot_details.plot_color ...
    ,'distributionMarkers','.' ...
    , 'spreadWidth', plot_details.sw ...
    );

bar(plot_details.it_model_comparison,nanmean(plot_details.these_data), ...
    'FaceColor',plot_details.plot_color,'FaceAlpha',plot_details.f_a,'EdgeColor',[0 0 0] );

set(gca ...
    ,'XTick',[] ...
    ,'fontSize',plot_details.graph_font ...
    ,'FontName','Arial',...
    'XLim',[0 plot_details.num_models_per_study-(plot_details.human_is_index-1)] ...
    );
ylabel(plot_details.y_string);
%%%%%%%%%%%%%%%%%plot_a_model%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Use to plot samples and ranks inside loop through models, without repeating code
function plot_a_bar(plot_details, Generate_params);

        study = plot_details.study;
        this_bar = plot_details.this_bar;
        these_data = plot_details.these_data;

        figure(plot_details.fig)
        plot_details.panel_num = plot_details.panel_nums(1);  %Panels nums can be changed colwise versus rowwise in subplot
        subplot(plot_details.rows,plot_details.cols,eval(plot_details.subplot_num)); hold on;
        
        panel_num = plot_details.panel_nums(1);  %Panels nums can be changed colwise versus rowwise in subplot
        subplot(plot_details.rows,plot_details.cols,eval(plot_details.subplot_num)); hold on; 
        identifier = Generate_params.study(study).model(this_bar).identifier;
        
        plot_color = plot_details.plot_cmap(identifier,:); %+2 skips the first color for participants and the second color for optimal
        model_label = Generate_params.study(study).model(this_bar).name;
        
        
        
        %average over sequences (rows) but keep sub data (cols) for scatter points
        handles = plotSpread(these_data ...
            ,'xValues',this_bar ...
            ,'distributionColors',plot_color ...
            ,'distributionMarkers','.' ...
            , 'spreadWidth', plot_details.sw ...
            );
        
        bar(this_bar,nanmean(these_data) ...
            ,'FaceColor',plot_color ...
            ,'FaceAlpha',plot_details.f_a ...
            ,'EdgeColor',[0 0 0] ...
            );
        
        set(gca ...
            ,'XTick',[] ...
            ,'fontSize',plot_details.graph_font ...
            ,'FontName','Arial',...
            'XLim',[0 plot_details.num_models_per_study+2] ...
            ,'YTick',[0:2:Generate_params.study(study).seq_length] ...
            ,'YLim',[0,Generate_params.study(study).seq_length]);
            ylabel(plot_details.y_string);
        
        this_offset = -plot_details.x_axis_test_offset*diff(ylim);
        text( this_bar, this_offset ...
            ,sprintf('%s',model_label) ...
            ,'Color', [0 0 0] ...
            ,'Fontname','Arial' ...
            ,'Fontsize',plot_details.graph_font ...
            ,'Rotation',plot_details.x_rot ...
            ,'HorizontalAlignment','right' ...
            );
 %%%%%%%%%%%%plot_a_bar%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%












%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function New_struct = combineModels(New_struct_SV, New_struct_OV);
function New_struct = combineModels(data, bar_order);

%I'm making it so identifier 1 is participants, iderntifier 2 is ideal
%observer and the rest of the identifiers (which in original struct were integers 1
%or greater) are now added a two to make room for participants and ideal
%observer. The identifiers in hybrid_figures* are used to access color maps

%This takes the subjective values struct and the objective values struct and
%makes a combined struct from them, with some formastting to prepare it for
%figure-worthy plotting

%bar_order gives the new model type order for the bars. SV and OV model
%types like Cs for example will be placed together in position defined (for both value type) by
%bar_orde. Each index in bar order refers to number in bar_order refers to
%the position in the new struct and the number at that index refers to the
%position in the old array. If a zero appears in an index position, then
%that means put the participant samples data there instead of a model.

%Basically, for the intended procedure, the idea is to move optimal from the
%sixth position to the first and insert participants after optimal and then
%have the theoretical models follow.

%subjective and objective value models? Or only objective values models?
num_datasets = size(data,2);

%initialise based on first dataset. These things are the same whether the model is SV or OV
New_struct.num_samples = data{1}.num_samples; %Both SV and OV should have the same data for participant samples, only models should differ
New_struct.samples_by_sequence = data{1}.samples_by_sequence;
New_struct.seq_length = data{1}.seq_length;
New_struct.num_seqs = data{1}.num_seqs;
New_struct.num_models = size(data{1}.model,2);

%concatenate models
% for model = 1:New_struct.num_models;
% if num_datasets == 2;
%     model_name_suffix = {'SV','OV'};
% else
    model_name_suffix = {''};
% end;
it_bar_order = 1;
for model = 1:numel(bar_order);
    
    if bar_order(model) == 0;
        
        %Put participants' samples data in free model location
        temp(it_bar_order).num_samples_est = data{1}.num_samples;
        temp(it_bar_order).ranks = data{1}.ranks;
        temp(it_bar_order).identifier = 1;
        temp(it_bar_order).name = "Human";
        temp(it_bar_order).skip = 1;    %1 will tell BIC and frequency plotters to ignore participants
        
        %set up for next model entry in next loop iteration
        it_bar_order = it_bar_order + 1;
        
    else
        
        for dataset = 1:num_datasets;
            
            %Assign sample data for this model and dataset
            temp(it_bar_order).this_models_free_parameters = data{dataset}.model(bar_order(model)).this_models_free_parameters;
            temp(it_bar_order).estimated_params = data{dataset}.model(bar_order(model)).estimated_params;
            temp(it_bar_order).ll = data{dataset}.model(bar_order(model)).ll;
            temp(it_bar_order).num_samples_est = data{dataset}.model(bar_order(model)).num_samples_est;
            temp(it_bar_order).ranks = data{dataset}.model(bar_order(model)).ranks;
            
            %Let's do some reformatting of labels to facilitate plotting later
            if bar_order(model) == 6;   %If optimality model
                
                temp(it_bar_order).identifier = 2;  %Humans are 1, ideal observer is 2
                temp(it_bar_order).name = "IO";
                temp(it_bar_order).skip = 1;    %1 will tell BIC and frequency plotters to ignore IO
                
            elseif bar_order(model) == 1;   %If cut-off model
                
                temp(it_bar_order).identifier =  data{dataset}.model(bar_order(model)).identifier + 2; %make two spaces for humans and io
                temp(it_bar_order).name = "CO";    %shorten name and add suffix
                temp(it_bar_order).skip = 0;    %0 will tell BIC and frequency plotters to plot this as a theoretical model
                
            elseif bar_order(model) == 2;   %If Cs model (only want to slightly modify name to conform to others' format)
                
                temp(it_bar_order).identifier =  data{dataset}.model(bar_order(model)).identifier + 2; %make two spaces for humans and io
                temp(it_bar_order).name = "CS";    %shorten name and add suffix
                temp(it_bar_order).skip = 0;    %0 will tell BIC and frequency plotters to plot this as a theoretical model
                
            elseif bar_order(model) == 3;   %If Optimism model (only want to slightly modify name so readers don't confuse Opt with optimal)
                
                temp(it_bar_order).identifier =  data{dataset}.model(bar_order(model)).identifier + 2; %make two spaces for humans and io
                temp(it_bar_order).name = "O";    %shorten name and add suffix
                temp(it_bar_order).skip = 0;    %0 will tell BIC and frequency plotters to plot this as a theoretical model
                
            else;   %If any other model (i.e., BV, BR), then use the literal model string name without a change
                
                temp(it_bar_order).identifier =  data{dataset}.model(bar_order(model)).identifier + 2; %make two spaces for humans and io
                temp(it_bar_order).name = data{dataset}.model(bar_order(model)).name;    %shorten name and add suffix
                temp(it_bar_order).skip = 0;    %0 will tell BIC and frequency plotters to plot this as a theoretical model
                
            end;
            
            temp(it_bar_order).old_name = data{dataset}.model(bar_order(model)).name;
            
            %set up for next model entry in next loop iteration
            it_bar_order = it_bar_order + 1;
            
        end;    %loop sv. ov datasets
        
    end; %check if participants spot in bar order
    
end;    %loop through bar order indices (locations in new struct)

New_struct.model = temp;

fprintf('');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%











%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function New_data_struct = reformatData(data);

%Makes a simplified struct from the data that's easier to combine between
%subjective anbd objective values in the next step

%I've just copied this straight from combine_hybrid_objective_v3 then modified it to fit new context

num_models = size(data.Generate_params.model,2);

New_data_struct.seq_length = data.Generate_params.seq_length;    %assumes all studies use same seq length (num options per seq)
New_data_struct.num_samples = [];
New_data_struct.ranks = [];
New_data_struct.num_seqs = [];

% for j=1:size(data,2);   %datasets
%
% disp(sprintf('dataset: %d subjects', size(data.Generate_params.num_samples,2)));

%participants
New_data_struct.num_samples = ...
    [New_data_struct.num_samples; ...
    nanmean(data.Generate_params.num_samples)' ...
    ];

New_data_struct.ranks = ...
    [New_data_struct.ranks; ...
    nanmean(data.Generate_params.ranks)' ...
    ];

%participants - I need to get every participant's sequence length
%so I can compute BIC
New_data_struct.num_seqs = ...
    [New_data_struct.num_seqs; ...
    sum(data.Generate_params.num_samples>0)' ...
    ];

%I've already saved the mean number of samples for each subject above to
%New_data_struct.num_samples for use making bar plot, but a reviewer asked
%for samples by sequence number so I'll need to add a field for samples
%without averaging too
%So this should be num_seqs*num_subs
New_data_struct.samples_by_sequence = data.Generate_params.num_samples;

% end;    %datasets

disp(sprintf('TOTAL: %d participants', numel(New_data_struct.num_samples)));



for model=1:num_models; %models, minus 1 if optimal (io) is last one
    
    %copy some stuff over that's common to all studies (assumed)
    New_data_struct.model(model).identifier = data.Generate_params.model(model).identifier;
    New_data_struct.model(model).name = data.Generate_params.model(model).name;
    New_data_struct.model(model).this_models_free_parameters = data.Generate_params.model(model).this_models_free_parameters;
    %initialise some stuff to accumulate over studies
    New_data_struct.model(model).estimated_params = [];
    New_data_struct.model(model).ll = [];
    New_data_struct.model(model).num_samples_est = [];
    New_data_struct.model(model).ranks= [];
    
    %models
    New_data_struct.model(model).estimated_params = ...
        [New_data_struct.model(model).estimated_params; ...
        data.Generate_params.model(model).estimated_params ...
        ];
    
    New_data_struct.model(model).ll = ...
        [New_data_struct.model(model).ll; ...
        data.Generate_params.model(model).ll ...
        ];
    
    New_data_struct.model(model).num_samples_est = ...
        [New_data_struct.model(model).num_samples_est; ...
        nanmean(data.Generate_params.model(model).num_samples_est)' ...
        ];
    
   New_data_struct.model(model).ranks = ...
        [New_data_struct.model(model).ranks; ...
        nanmean(data.Generate_params.model(model).ranks_est)' ...
        ];
    
    %     end;    %datasets
    
end;    %models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








