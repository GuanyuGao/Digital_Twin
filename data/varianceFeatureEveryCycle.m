%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variance model features %
%%%%%%%%%%%%%%%%%%%%%%%%%%%
function varianceFeatureEveryCycle(whole_batch)

% addpath('D:\Matlab\toolbox\jsonlab-master');
% three batches have 140 battery in total.
battery_num = 140;

fid = fopen('data.json', 'w+');
% whole_battery: the dataset of combining the three batches.

% get the value of Qdlin in first cycle of every battery, 
% jion them and get a 1000 * 140 matrix 
for i = 1:battery_num
    
    % set the second cycle as the standard value
    Q_2 = whole_batch(i).cycles(2).Qdlin; 
    cycle_life = whole_batch(i).cycle_life - 1;
    
    for j = 3:cycle_life
        Q_j = whole_batch(i).cycles(j).Qdlin;
        
        delta_Q = Q_j - Q_2;
        var = abs(sum((delta_Q - mean(delta_Q)).^2)/1000);
        log_var = log10(var);
        
        data = jsonencode({[j;log_var]});
        fprintf(fid, '%s',data);
    end
end

fclose(fid);







