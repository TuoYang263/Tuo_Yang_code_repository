%%DIIP Practical assignment:coins recogntion 
clc,clear,close all;

%define all needed files paths
bias_path = './DIIP-images/Bias/';
flat_path = './DIIP-images/Flat/';
dark_path = './DIIP-images/Dark/';

%define the format of image which will be read
file_format = '*.jpg';

%build one path sets used for traversing
paths_set = {bias_path, flat_path, dark_path};

%build one image sets used for storing images
images_set = cell(1,size(paths_set,2));

%build one length to record the number of each type of images
length_set = [];

%read all images into matlab workspace
for i = 1:size(paths_set, 2)
    %acquire all paths
    files = dir(fullfile(paths_set{1,i},file_format));
    length  = size(files,1);
    length_set = [length_set length];
    %traversing every image in the current path
    for j = 1:length
        file_name = strcat(paths_set{1,i}, files(i).name);
        image = imread(file_name);
        images_set{1,i}(:,:,:,j) = image;
    end
end

% get the mean value of bias, flat and dark to do the intensity
% calibration
bias_sum = 0;
flat_sum = 0;
dark_sum = 0;
for i = 1:length_set(1)
    bias_sum = bias_sum + images_set{1,1}(:,:,:,i);
    flat_sum = flat_sum + images_set{1,2}(:,:,:,i);
    dark_sum = dark_sum + images_set{1,3}(:,:,:,i);
end
mean_bias = bias_sum/length_set(2);
mean_dark = dark_sum/length_set(2);
mean_flat = flat_sum./flat_sum;

%%part of defining the input image
measurement_path = './DIIP-images/Measurements/';

% name of the image going to be recognized 
measurement_filename = '_DSC1772.JPG';
% %call the function estim_coins to get coin number measurement results

% define the path where we will extract images to be measured 
measurement = imread(strcat(measurement_path,measurement_filename));

tic;
coins = estim_coins(measurement,mean_bias,mean_dark,mean_flat);
toc;

% %output results
disp(['Statistical results(',measurement_filename,'):']);
disp(['the number of 2 euro:',num2str(coins(1))]);
disp(['the number of 1 euro:',num2str(coins(2))]);
disp(['the number of 50 cent:',num2str(coins(3))]);
disp(['the number of 20 cent:',num2str(coins(4))]);
disp(['the number of 10 cent:',num2str(coins(5))]);
disp(['the number of 5 cent:',num2str(coins(6))]);













