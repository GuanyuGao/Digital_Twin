%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variance model features %
%%%%%%%%%%%%%%%%%%%%%%%%%%%
function varianceFeatureEveryCycle(whole_batch)

% addpath('D:\Matlab\toolbox\jsonlab-master');
% three batches have 140 battery in total.
battery_num = 100;
start = 1;

fid = fopen('data1.json', 'w+');
% whole_battery: the dataset of combining the three batches.

% get the value of Qdlin in first cycle of every battery, 
% jion them and get a 1000 * 140 matrix 
% get [cycle_life log_var] couple
% for i = 1:battery_num
%     
%     % set the second cycle as the standard value
%     Q_2 = whole_batch(i).cycles(2).Qdlin; 
%     cycle_life = whole_batch(i).cycle_life - 1;
%     
%     for j = 3:cycle_life
%         Q_j = whole_batch(i).cycles(j).Qdlin;
%         
%         delta_Q = Q_j - Q_2;
%         var = abs(sum((delta_Q - mean(delta_Q)).^2)/1000);
%         log_var = log10(var);
%         
%         data = jsonencode({[j;log_var]});
%         fprintf(fid, '%s',data);
%     end
% end
% 
% fclose(fid);

log_vars = [];
cycle_lifes = [];
for i = start:battery_num
    Q_2 = whole_batch(i).cycles(2).Qdlin;
    cycle_life = whole_batch(i).cycle_life - 1;
    if isnan(cycle_life)
        continue;
    end
    for j = 3:cycle_life - 1
        
        Q_j = whole_batch(i).cycles(j).Qdlin;
        delta_Q = Q_j - Q_2;
        var = abs(sum((delta_Q - mean(delta_Q)).^2)/1000);
        log_var = log10(var);
        
        log_vars = [log_vars log_var];
        cycle_lifes = [cycle_lifes cycle_life - j];
    end
    
end
        
json_log_var = jsonencode(log_vars);
json_cycle_life = jsonencode(cycle_lifes);
fprintf(fid, '%s', json_log_var);

fprintf(fid,'\r\n');
fprintf(fid, '%s', json_cycle_life)


fclose(fid);

