function [camMatrix,estimate_pixels] = calibrate(worldPoints,imagePoints)
    %do the normalization to image points and world points respectively
    xy_mean = mean(imagePoints);
    d_mean = mean(sum(sqrt((imagePoints-xy_mean).^2)));
    T = diag(ones(1,size(imagePoints,2)+1));
    for i = 1:size(imagePoints,2)
        T(i,i) = sqrt(size(imagePoints,2))/d_mean;
        T(i,end) = -sqrt(size(imagePoints,2))*xy_mean(i)/d_mean;
    end
    normalized_imagePoints = T * [imagePoints';ones(1,size(imagePoints,1))];
    
    XYZ_mean = mean(worldPoints);
    D_mean = mean(sum(sqrt((worldPoints-XYZ_mean).^2)));
    U = diag(ones(1,size(worldPoints,2)+1));
    for i = 1:size(worldPoints,2)
        U(i,i) = sqrt(size(worldPoints,2))/D_mean;
        U(i,end) = -sqrt(size(worldPoints,2))*XYZ_mean(i)/D_mean;
    end
    normalized_worldPoints = U * [worldPoints';ones(1,size(worldPoints,1))];
    
    %extract three coordinates of world points
    X = normalized_worldPoints(1,:)';
    Y = normalized_worldPoints(2,:)';
    Z = normalized_worldPoints(3,:)';
    
    %extract two coordinates of pixel coordinates
    u = normalized_imagePoints(1,:)';
    v = normalized_imagePoints(2,:)';
    
    M = numel(X);       %get the number of world points
    %build unit vector and zero vector, the number of which is
    %the number of points
    vec_1 = ones(M,1,'like',X);
    vec_0 = zeros(M,1,'like',X);
    
    A = [X     Y     Z     vec_1 vec_0 vec_0 vec_0 vec_0 -u.*X -u.*Y -u.*Z -u;
        vec_0  vec_0 vec_0 vec_0 X     Y     Z     vec_1 -v.*X -v.*Y -v.*Z -v];
    
    %decompose matrix A's right singular matrix
    [UU,SS,V] = svd(A);
    P = V(:,end);

    %get the projection matrix
    camMatrix = reshape(P,4,3)';
    
    %set back to the normalization transform
    camMatrix = inv(T)*camMatrix*U;
    
    %get reprojection points
    reprojection_points = [worldPoints';ones(1,size(worldPoints,1))];
    reprojection_points = camMatrix * reprojection_points;
    
    estimate_pixels = [reprojection_points(1,:)./reprojection_points(3,:);
        reprojection_points(2,:)./reprojection_points(3,:)];
end

