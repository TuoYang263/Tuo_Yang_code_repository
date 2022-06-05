function [training_dataset,training_class, testing_dataset, testing_class] =...
    preprocess(file_names,removal_indices)
    %this function is mainly used for do some processing to the data
    %to ensure the training performance of classifier, after removing abnormal
    %samples from datasets, take 70% samples of each digit's ones as training 
    %dataset, and the left 30% can be used as the testing dataset
    for i = 1:length(removal_indices)
        file_names(removal_indices(i)) = [];
    end
    digits_range = 0:9;               %totally we got 9 digits data samples
    training_percentage = 0.8;          
    training_dataset = {};
    testing_dataset = {};
    cell_names = {file_names(:).name}; %get all values of field name
    train_index = 1;
    test_index = 1;
    for j = digits_range
        matching_str = ['stroke_',num2str(j)];
        current_digit_samples = startsWith(cell_names,...
            matching_str,'IgnoreCase',true);
        current_sample_num = sum(current_digit_samples);
        current_real_indices = find(current_digit_samples == 1);
        random_indices = randperm(length(current_real_indices));
        training_indices = random_indices(1:round(current_sample_num*training_percentage));
        testing_indices = random_indices(round(current_sample_num*training_percentage)+1:end);
        for k = 1:length(training_indices)
            train_data = load(['./mat_training/',...
                cell_names{1,current_real_indices(training_indices(k))}]);
            training_dataset{1,train_index} = train_data.pos;
            training_dataset{2,train_index} = j+1;
            train_index = train_index + 1;
        end
        for m = 1:length(testing_indices)
            test_data = load(['./mat_training/',...
                cell_names{1,current_real_indices(testing_indices(m))}]);
            testing_dataset{1,test_index} = test_data.pos;
            testing_dataset{2,test_index} = j+1;
            test_index = test_index + 1;
        end
    end

    %resampling data,which makes the data own the same size
    training_length_sum = 0;
    testing_length_sum = 0;
    for i = 1:size(training_dataset,2)
         training_length_sum = training_length_sum + size(training_dataset{1,i},1);
    end
    for j = 1:size(testing_dataset,2)
         testing_length_sum = testing_length_sum + size(testing_dataset{1,j},1);
    end
    sample_rate = round((training_length_sum+testing_length_sum)/length(cell_names));
    
    %mess up the order of training dataset and testing dataset
    %1.firstly mess up the order of
    rng('shuffle'); %just to make sure different numbers generated every time
    train_messup_vector = 1:1:size(training_dataset,2);
    %numel:return the number of condition satisified elements
    %and the next line is to generate some order-
    randomly_taken_index = train_messup_vector(randperm(numel(train_messup_vector)));
    for i = 1:size(training_dataset,2)
        new_training_dataset(:,i) = training_dataset(:,randomly_taken_index(i));
    end
    training_dataset = new_training_dataset;
    test_messup_vector = 1:1:size(testing_dataset,2);
    randomly_taken_index = test_messup_vector(randperm(numel(test_messup_vector)));
    for i = 1:size(testing_dataset,2)
        new_testing_dataset(:,i) = testing_dataset(:,randomly_taken_index(i));
    end
    testing_dataset = new_testing_dataset;
    %acquire labels from two cell matices:training_dataset and testing_dataset
    training_class = cell2mat(training_dataset(2,:));
    training_dataset(2,:) = [];
    testing_class = cell2mat(testing_dataset(2,:));
    testing_dataset(2,:) = [];
end