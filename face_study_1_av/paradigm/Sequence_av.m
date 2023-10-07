function Sequence_av(sub, female);

%to run go to matlab command line and type:
%Sequence(subject_number, female);
%for subject_number, type the subject_number (e.g., 3)
%for female put a 1 if female faces, 0 for male faces

%if you don't enter arguments, it will default to sub 99 and female == 1
if ~exist('sub');
    sub=99;
end;

if ~exist('female');
    female = 1;
end;

addpath(genpath('C:\Cogent2000v1.32'));
config_display ( 0, 3, [0 0 0], [1 1 1], 'Arial', 100, 4); %0 is partial, 1 is full screen
config_keyboard(100,5,'nonexclusive');

curr_path = pwd;

%prepare experiment start
start_cogent;

num_seqs = 28;
num_choices = 8;

%keep this (1) and comment out (2) for normal operation
%doc xlsread if problems
if female == 1;
    [dummy data] = xlsread('all_females.xlsx');
else
    [dummy data] = xlsread('all_males.xlsx');
end;

%need to figure out some stuff for the small pics of refused options
%how much space are we working with?
scr_width = [-450:1:450];    %roughly, in pixels
scr_width_size = numel(scr_width);
pic_space = scr_width_size/num_choices; %how much space available for each pic?
%image_dims = [350 453];
%new_sizes = image_dims/5;   %can scale their size, 5 should be less than pic space
locs = round(scr_width(1):pic_space:scr_width(end)+1);

%instructions
settextstyle('Arial', 30); %big text
line_start = 100; line_increment = 80;
preparestring('You will see a number of sequences. Treat each sequence as it would be the last one.',1,0, line_start);
preparestring('Press key "2" to choose the most attractive face in the sequence.',1,0, line_start-line_increment);
preparestring('The last face will be assigned as choice if no choice is made be then.',1,0, line_start-line_increment*2);
preparestring('Press space bar to continue.',1,0,line_start-line_increment*3);

%if you put this here. then instructions will stay on the screen until a
%button is pressed, then the buffer will be cleared
drawpict( 1 ); %sends buffer contents to screen
[k,t,n]=waitkeydown(inf);
clearpict(1);


preparestring(['Each sequence has 8 options'],1,0,line_start-line_increment);
drawpict( 1 ); % sends buffer contents to screen
[k,t,n]=waitkeydown(inf);
clearpict(1);

%get ransomly selected images
% rows_data = randi(size(data,1), [num_seqs*num_choices 1]);
%randi samples with replacement so use randperm instead
rows_data = randperm(size(data,1),num_seqs*num_choices);
rows_data = reshape(rows_data,num_seqs,num_choices);

rows_data_it = 1; %we are making an iterator for rows_data and it starts at one

for trial=1:num_seqs;
    
    %clear chosen_image this_sequence;
    
    %so will iterate trial=1,2,3,4,5.. until it reaches the total number of
    %sequences
    
    
    preparestring('NEXT SEQUENCE',1,0,line_start-line_increment);
    
    %if you put this here. then instructions will stay on the screen until a
    %button is pressed, then the buffer will be cleared
    drawpict( 1 ); %sends buffer contents to screen
    [k,t,n]=waitkeydown(inf);
    clearpict(1);
    
    
    % initialise stuff
    blank_switch = 0; % switch to show blank image is off
    used_faces = [];
    
    for option=1:num_choices;
        
        % put main pic into buffer
        if blank_switch == 0
            picture = data{rows_data(trial,option),1}; % find filename
        else
            picture = 'grey_box.jpg';
            
        end
        
        %put main pic into buffer
       %     picture = data{rows_data( trial,option),1};    %find filename
        loadpict(picture,1,0,50);
        %loadpict(picture,1,0,50,image_dims(1),image_dims(2));
        %put option number into buffer
        preparestring(sprintf('Number of left options: %d',num_choices-option),1,0,325);
        %put used pics into buffer
        preparestring(sprintf('Refused options: %d',option-1),1,-400,-220);
        for used=1:size(used_faces,2);  %loop through used faces and load them into buffer
            %             loadpict(used_faces{used},1,-400,-300,88,113);
            %loadpict([ curr_path '/small/' used_faces{used}],1,locs(used),-300,new_sizes(1),new_sizes(2));
        try
            loadpict([ curr_path '/small/' used_faces{used}],1,locs(used),-300 );
        catch
            disp('SMALL FILE NOT FOUND');
        end;

        end;
        
        %send all the stuff to screen, wait for response
        drawpict(1);
        [k,t,n]=waitkeydown(inf);
        clearpict(1);
        
        % get response, update stuff
        if blank_switch == 1; % keep them from choosing the grey box
            response = 28;
        elseif blank_switch == 0 & option == num_choices; % if last option and no choice, force them to take last option
            response = 29;
        else
            response = k(numel(k)); %they might have hit more than one button, pull out the last one
        end; 
        
        
        
        %get response, update stuff
        %response = k(numel(k)); %they might have hit more than one button, pull out the last one
        output{rows_data_it,1} = rows_data_it;  %col 1 trial number
        output{rows_data_it,2}  = trial;       %col 2 sequence number
        output{rows_data_it,3} = option;       %col 3 option number
        output{rows_data_it,4} = picture;      %col 4 face filename
        output{rows_data_it,5} = rows_data( trial,option);  %col 5 face number
        output{rows_data_it,6} = response;       %col 6: saves the last button pressed
        output{rows_data_it,7} = 0;              %col 7: reward value
        
        %prepare for next trial
        rows_data_it = rows_data_it + 1; %we increase this iterator by one everytime an image is shown, so that we can move on to the next on the list
        
        if response  ==  29;    %if they press keep (note == tests for equality and = assigns a value!!)
           blank_switch = blank_switch + 1;
           chosen_image = picture; 
          % break;
           %wait(1000)
        end;    %if they press anything else, this iteration will end by itself and the next one should start.
        
        %if face not chosen, store it in refused face cell array
        used_faces{option} = picture;
        
    end;    %ends loop through options
    
    %before progressing to next trial/sequence, present chosen image
    preparestring(['HERE IS YOUR NEW DATE! How rewarding is your choice?'],1,0,line_start+100);
    preparestring('Use a scale from 1 (not rewarding) to 9 (most rewarding)',1,0, line_start-line_increment+100);
    loadpict(chosen_image,1,0,-50);
    drawpict( 1 ); % sends buffer contents to screen
    [k,t,n]=waitkeydown(inf);
    clearpict(1);
    
    % save reward rating
    output{rows_data_it-1,7} = k(numel(k)); %col 7: reward value
    
    xlswrite(sprintf('av_sequence_sub%02d_sex%02d.xlsx',sub,female),output);
    save(sprintf('av_sequence_sub%02d_sex%02d.mat',sub,female));
end;    %ends loop through sequences


stop_cogent;



    




