%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variance model features %
%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [log_vars, cycle_lifes] = varianceFeatureEveryCycle(whole_batch)

%     fid = fopen('ALLdata.json', 'w+');
% 
%     log_vars = [];
%     log_miniums = [];
%     log_skewnesses = [];
%     log_kurtosises = [];
%     cycle_lifes = [];
% 
%     % get the battery info specified by battery id 
%     battery_id = 1;
%     % set the second cycle of the battery as the benchmark
%     Q_2 = batch(battery_id).cycles(2).Qdlin;
%     cycle_life = batch(battery_id).cycle_life - 1;
%     % calculate the some features by every cycles
%     for j = 3:cycle_life
% 
%             Q_j = batch(battery_id).cycles(j).Qdlin;
%             delta_Q = Q_j - Q_2;
%             var = abs(sum((delta_Q - mean(delta_Q)).^2) / 1000);
%             log_var = log10(var');
% 
%             % Minimum
%             minium = abs(min(delta_Q));
%             log_minium = log10(minium');
% 
%             % Skewness
%             numerator_s = sum((delta_Q-mean(delta_Q)).^3) / 1000;
%             denominator_s = sqrt(sum((delta_Q-mean(delta_Q)).^2)).^3;
%             skewness = abs(numerator_s./denominator_s);
%             log_skewness = log10(skewness');
% 
%             % Kurtosis
%             numerator_k = sum((delta_Q-mean(delta_Q)).^4) / 1000;
%             denominator_k = (sum((delta_Q-mean(delta_Q)).^2) / 1000).^2;
%             kurtosis = abs(numerator_k./denominator_k);
%             log_kurtosis = log10(kurtosis');
% 
%             log_vars = [log_vars log_var];
%             log_miniums = [log_miniums, log_minium];
%             log_skewnesses = [log_skewnesses, log_skewness];
%             log_kurtosises = [log_kurtosises, log_kurtosis];
%             cycle_lifes = [cycle_lifes cycle_life - j];
%     end
%     
%     dict = struct(...
%         'log_vars', log_vars,... 
%         'log_miniums', log_miniums,...
%         'log_skewnesses', log_skewnesses,... 
%         'log_kurtosises', log_kurtosises,...
%         'cycle_lifes', cycle_lifes...
%         );
%     
%     json_dict = jsonencode(dict);
%     fprintf(fid, '%s', json_dict);
% 
%     fclose(fid);

    fid = fopen('ALLdata.json', 'w+');
    battery_num = 5;
    start = 1;
    log_vars = [];
    log_miniums = [];
    log_skewnesses = [];
    log_kurtosises = [];
    cycle_lifes = [];
    for i = start:battery_num
        Q_2 = whole_batch(i).cycles(2).Qdlin;
        cycle_life = length(whole_batch(i).cycles);
        if isnan(cycle_life)
            continue;
        end
        for j = 3:cycle_life - 1

            Q_j = whole_batch(i).cycles(j).Qdlin;
            delta_Q = Q_j - Q_2;
            var = abs(sum((delta_Q - mean(delta_Q)).^2) / 1000);
            log_var = log10(var);

            % Minimum
            minium = abs(min(delta_Q));
            log_minium = log10(minium');

            % Skewness
            numerator_s = sum((delta_Q-mean(delta_Q)).^3) / 1000;
            denominator_s = sqrt(sum((delta_Q-mean(delta_Q)).^2)).^3;
            skewness = abs(numerator_s./denominator_s);
            log_skewness = log10(skewness');

            % Kurtosis
            numerator_k = sum((delta_Q-mean(delta_Q)).^4) / 1000;
            denominator_k = (sum((delta_Q-mean(delta_Q)).^2) / 1000).^2;
            kurtosis = abs(numerator_k./denominator_k);
            log_kurtosis = log10(kurtosis');
        
            log_vars = [log_vars log_var];
            log_miniums = [log_miniums, log_minium];
            log_skewnesses = [log_skewnesses, log_skewness];
            log_kurtosises = [log_kurtosises, log_kurtosis];
            cycle_lifes = [cycle_lifes cycle_life - j];
        end

    end
    
    dict = struct(...
        'log_vars', log_vars,... 
        'log_miniums', log_miniums,...
        'log_skewnesses', log_skewnesses,... 
        'log_kurtosises', log_kurtosises,...
        'cycle_lifes', cycle_lifes...
        );
    
    json_dict = jsonencode(dict);
    fprintf(fid, '%s', json_dict);

    fclose(fid);
    
    
    

 

