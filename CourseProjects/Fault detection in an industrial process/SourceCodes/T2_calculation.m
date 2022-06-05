%%Stu No:0592964 Name:Subhashree Rautray
%%T2 calculation
function [T2,T2_threshold] = T2_calculation(train_data,test_data,sample_num,...
    remain_components,alpha)
    %use pca method to get principal components
    [t,p,r2] = pca(train_data);
    s = svd(train_data/sqrt(sample_num-1));
    P = p(:,1:remain_components);
    sigma2 = s(1:remain_components).^(-2);
    for i = 1:size(train_data,1)
        x = test_data(i,:)';
        T2(i) = x'*P.*sigma2'*P'*x;
    end
    %calculate the t2 threshold level
    ts = finv(1-alpha, remain_components, sample_num-remain_components);
    T2_threshold = ((sample_num.^2-1)*remain_components)/(sample_num*...
        (sample_num-remain_components)).*ts;
end
