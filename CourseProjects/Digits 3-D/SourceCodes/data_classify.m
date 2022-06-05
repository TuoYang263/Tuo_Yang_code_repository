function C = data_classify(training_data, training_class, testing_data, k)
    %this function is the main part of KNN algorithm, and calling dtw function
    %to calculate the distance between samples as well
    %get the classes' labels without repeatation
    unique_classes = unique(training_class);
    %the number of training data samples
    train_num = size(training_data,2);
    %the number of testing data samples
    test_num = size(testing_data,2);
    
    %initialize the vector C
    C = zeros(1,size(testing_data,2));
    
    %total number of classes
    class_num = length(unique_classes);
    
    %use the loop to find the unknown samples k-nearest neighbours
    for i = 1:test_num
        distance = zeros(2,train_num);
        for j = 1:train_num
            distance(:,j) = [dtw(testing_data{1, i}, training_data{1, j},30);...
                training_class(j)];
        end
        
        %decrease k when it's ambiguous
        ambiguous = 0;
        
        %use the loop to find the class label of this unknown sample
        while true
            mask = ones(1,train_num);
            classCount = zeros(1, class_num);
            
            %count the each class samples using k nearest neighbours
            for m = 1:(k-ambiguous)
                %get the nearest point to the unknown sample using min
                [min_value,index] = min(distance(1,mask == 1));
                %add this point to corresponding class counter
                classCount(distance(2, index)) = classCount(distance(2, index)) + 1;
                %the data has been counted 
                mask(index) = 0;
            end
            %get the class which has the most samples in k-nearest
            %neighbors 
            [max_value,index] = max(classCount);
            %if ambiguous situation happens, k will be deducted by
            %ambiguous
            if sum(classCount == max_value)>1
                ambiguous = ambiguous + 1;
            else
                C(i) = index;
                break;
            end
        end
    end
end