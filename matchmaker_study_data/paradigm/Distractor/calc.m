function calc;

addpath(genpath('C:\Cogent2000v1.32'));
config_display ( 1, 3, [0 0 0], [1 1 1], 'Arial', 100, 4);
config_keyboard(100,5,'nonexclusive');

curr_path = pwd;

start_cogent;


[dummy data] = xlsread('all_calc.xlsx')

ncalc = size(data,1)

scr_width = [-450:1:450];

%%%Instructions
settextstyle('Arial', 30);
line_start = 100; line_increment = 80; %%% ? Need it
preparestring('Try and solve as many calculations as possible in max 2 min.', 1,0, line_start);
preparestring('Use the digit keys along the keyboard to answer.',1,0,line_start-line_increment);
preparestring('Press space bar to begin.',1,0,line_start-line_increment*2);

drawpict( 1 );
[k,t,n]=waitkeydown(inf);
clearpict(1);

rows_calculations_it = 1;
correct = [0 0 0 0 0 0 0 0 0 0];
tStart = tic;


for i=1:ncalc;
   
    picture = data{i,1}; loadpict(picture,1,0,50);
    preparestring(sprintf('Left: %d',ncalc-i),1,0,line_start*2);
    if i > 0;
    preparestring(sprintf('Time left updated: %d sec',120-round(toc(tStart))),1,0,line_start+50);
    end;
    
    drawpict(1);
    [k,t,n]=waitkeydown(inf);
    clearpict(1);
    
    response = k(numel(k)); 
    
    if response == dummy(i);
        correct(i) = 1;
    else
        correct(i) = 0;
    end;
    
  
        
    tElapsed = toc(tStart)
   
   if tElapsed > 120;  
       

       preparestring(sprintf('Correct answers percentage: %d %',round(sum(correct)/size(correct,2)*100)),1,0,line_start-line_increment);
       drawpict( 1 );
       wait(1000);
       clearpict(1);
       break;
   elseif tElapsed < 120 && i == ncalc;
        
       preparestring(sprintf('Correct answers percentage: %d %',round(sum(correct)/size(correct,2)*100)),1,0,line_start-line_increment);
       drawpict( 1 );
       wait(1000);
       clearpict(1);
       break;
   %else
      % rows_calculations_it = rows_calculations_it + 1;
       
   end;
   
   

end;

stop_cogent;