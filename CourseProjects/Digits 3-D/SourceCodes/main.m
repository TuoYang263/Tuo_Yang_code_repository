%Task:Digits 3D testing main program(using mat files as experimental dataset)
%Notes:please run the first section from the start, it will take some some,
%please be patient, when the first section is over, all prediction results 
%will be stored inside the directory /evaluation_data, then you can run the
%second section next to visualize the prediction results and get the
%performance of classifer
clc,clear,close all;

%preprocess process:remove noisy samples and split up the training and 
%testing dataset with the percentage 7:3
%first step of preprocessing,draw figures out to find noisy samples
digit_sample_num = 100;        %every digit have 100 samples
file_names = dir(fullfile('.\mat_training\','*.mat'));
file_num  = size(file_names,1);

%this function is to visualize each digits' stroke data and rule out
%abnormal samples
%data_visualization(digit_sample_num, file_names,file_num)

%by observation,the following data samples should be removed from the
%dataset
removal_indices = [33 100 103 107 114 121 122 133 ...
                   138 146 162 164 167 168 169 171 177 ...
                   178 183 191 192 195 200 279 442 444 ...
                   446 468 499 593 623 667 677 694 764 ...
                   821 922];

[training_dataset,training_class, testing_dataset, testing_class] =...
    preprocess(file_names,removal_indices);

%it really takes some period of time
C = data_classify(training_dataset, training_class, testing_dataset, 2);
correct_classified_rate = sum(C == testing_class)/length(testing_class);
save('./evaluation_data/C.mat','C');
save('./evaluation_data/training_dataset.mat','training_dataset');
save('./evaluation_data/testing_dataset.mat','testing_dataset');
save('./evaluation_data/training_class.mat','training_class');
save('./evaluation_data/testing_class.mat','testing_class');
%% when get the best classification results, save it as mat format file
%then load it to draw all prediction results out and calculate correctly
%classified rate(this section is run respetively,just for saving time on training)
C = load('./evaluation_data/C.mat');
training_dataset = load('./evaluation_data/training_dataset.mat');
testing_dataset = load('./evaluation_data/testing_dataset.mat');
training_class = load('./evaluation_data/training_class.mat');
testing_class = load('./evaluation_data/testing_class.mat');

classification_visualization(testing_dataset,testing_class, C);


