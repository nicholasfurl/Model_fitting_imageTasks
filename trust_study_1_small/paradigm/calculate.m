function calculate;

addpath(genpath('C:\Cogent2000v1.32'));
config_display ( 0, 3, [0 0 0], [1 1 1], 'Arial', 100, 4);
config_keyboard(100,5,'nonexclusive');

curr_path = pwd;

start_cogent;

num_trials = 5;
[calculations distractors] = xlsread('');

scr_width = [-400:1:600];
scr_width_size = nume1(scr_width);
pic_space = scr_width_size/num_trials;
image_dims = [350 453];


%%%Instructions
settextstyle('Arial', 30);
line_start = 100; line_increment = 80; %%% ? Need it
preparestring('Do the calculations and use the digit keys along the letters to answer when ready', 1,0, line_start);
preparestring('Press space bar to continue',1,0,line_start-line_increment);

drawpict( 1 );
[k,t,n]=waitkeydown(inf);
clearpict(1);

rows_data = randperm(size(data,1),num_trials);
rows_data = reshape(rows_data,num_trials);

rows_data_it = 1;

for equation=1:num_trials;
    picture = data{rows_data(equation),1}; loadpict(picture,10,50);
    
    drawpict(1);
    [k,t,n]=waitkeydown(inf);



