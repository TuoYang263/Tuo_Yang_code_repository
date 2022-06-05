function distance = dtw(test_data,train_data,w)
    %function used for computing dtw distance between two samples
    %calculate the frame number of each sequence
    n = size(test_data,1);
    m = size(train_data,1);
    %frame matching distantial matrix
    dis_matching = zeros(n, m);
    for i = 1:n
        for j = max(1,i-w):min(m, i+w)
            %use the eculidean distance (L2 distance) to get the distance
            %between each sample
            dis_matching(i,j) = calculate_eculidean(test_data(i,:),...
                train_data(j,:));
        end
    end
    %accumulating distance matrix
    D = ones(n,m)*realmax;
    D(1,1) = dis_matching(1,1);
    %dynamic programming
    for i = 2:n
        for j = max(1,i-w):min(m, i+w)
            D1 = D(i-1,j);
            if j>1
                D2 = D(i-1,j-1);
            else
                D2 = realmax;
            end
            if j>2
                D3 = D(i-1,j-2);
            else
                D3 = realmax;
            end
            D(i,j) = dis_matching(i,j) + min([D1,D2,D3]);
        end
    end
    distance = D(n,m);
end