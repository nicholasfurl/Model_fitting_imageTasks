function phase1_ratings_av(sub, female, session);

%for example, if you want to run subject 5, who is male and its his third
%session then type on the matlabb command line >>phase1_ratings_nf(5,0,3)
%and hit enter

if ~exist('sub');
    sub=99;
end;

if ~exist('female');
    female = 1;
end;

if ~exist('session');
    session = 99;
end;


%change the path to wherever cogent is
addpath (genpath('C:\Cogent2000v1.32'));
%intialise (required for cogent)
config_display ( 0, 5, [0 0 0], [1 1 1], 'Arial', 100, 4); %0 is partial, 1 is full screen 
config_keyboard(100,5,'nonexclusive');

start_cogent;

% %keep this (1) and comment out (2) for normal operation
% %doc xlsread if problems
% [dummy data] = xlsread('files.xlsx');

if female == 1;
    [dummy data] = xlsread('all_females.xlsx');
else
    [dummy data] = xlsread('all_males.xlsx');
end;

n_trials = size(data,1);    %figure out the number of trials, based on input file
rng('shuffle'); %seed random number generator
order = randperm(n_trials); %determine a random order


%Put the instructions into a screen buffer
settextstyle('Arial', 30); %big text
line_start = 100; line_increment = 80;
preparestring('Please rate the attractiveness of each face on a scale from 1 to 9',1,0, line_start);
preparestring('Press space bar to continue',1,0,line_start-line_increment);

%sends the instructions in buffer to the screen, waits for space bar
drawpict( 1 ); %sends buffer contents to screen 
waitkeydown(inf, 71); %71 is key code for space bar
clearpict(1); %clears the buffer for next time

for i=1:n_trials; %loop through all the trials
    %clears key presses from memory
    clearkeys;
    
    %assign a random stimulus from list
%     rating = data{order(i),2};
    rating = 'rate from 1 (Very unattractive) to 9 (Highly attractive)?';
    picture = data{order(i),1}; %keep picture to be shown in this trial
    loadpict(picture,1);
  
    preparestring(rating, 1, 0, 300);
  
    
    drawpict( 1 ); 
    [k,t,n] = waitkeydown(inf);
    responses(i) = k(numel(k)); %saves the last key pressed to the output variable
    
    clearpict(1);
    
    %record the stuff that happened on this trial in one variable...
    output{i,1} = i;    %col1: which trial is it?
    output{i,2} = picture;  %col 2: filename of the picture
    output{i,3} = order(i);  %col 3, the number of the picture
    output{i,4} = responses(i); %col4: the response
    
    %write what you recorded to an excel file for analysis
    xlswrite(sprintf('av_ratings_sub%02d_sex%02d_sess%02d.xlsx',sub,female,session),output);
    save(sprintf('av_ratings_sub%02d_sex%02d_sess%02d.mat',sub,female,session));
    

end;

stop_cogent;
