%%Stu No:589715 Name:Yang Tuo
%%Q calculation
function [Q,Q_threshold] = Q_calculation(train_data,test_data,sample_num,...
    remained_components,alpha)
    [t0,p0,r2] = pca(train_data);
    s = svd(train_data/sqrt(sample_num-1));
    
    projected_t = test_data*p0;  %variable data(dataset) is projected to axis of
                            %d00_te
    %Q2
    Q = test_data - projected_t(:,1:remained_components)*p0(:,1:remained_components)';
    Q = Q';
    Q = sum(Q.^2);
    
    s = s(remained_components+1:end);
    theta1 = sum(s.^2);
    theta2 = sum(s.^4);
    theta3 = sum(s.^6);
    
    h0 = 1 - (2*theta1*theta3)/(3*theta2^2);
    ca = norminv(1-alpha);
    
    %Q threshold
    Q_threshold = theta1*((h0*ca*sqrt(2*theta2))/theta1 + 1 + (theta2*h0*...
        (h0-1))/theta1^2)^(1/h0);    
end