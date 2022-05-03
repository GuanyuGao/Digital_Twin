function [test_x, test_y, predictions] = load_prediction() 
    file_name = 'prediction.json';
    addpath('D:\matlab2020b\toolbox\jsonlab');
    data = loadjson(file_name);
    test_x = data.test_x;
    test_y = data.test_y;
    predictions = data.predictions;