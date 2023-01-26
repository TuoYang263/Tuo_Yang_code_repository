function [worldPoint1, worldPoint2] = back_projection(imagePoints, zConst, K, Rc_w, Pc)
    worldPtCam = pinv(K)*[imagePoints';
    ones(1,size(imagePoints,1))];
    % worldPtCam = pinv(K)*[scene2_pixels';ones(1,size(scene2_pixels,1))];

    %calculate the depth
    t = - Rc_w * Pc;
    rightMatrix = inv(Rc_w)*worldPtCam;
    leftMatrix = inv(Rc_w)*t;
    s = (zConst + leftMatrix(3))/rightMatrix(3);

    %first way of getting world coordinates
    worldPoint1 = inv(Rc_w) * (s * worldPtCam - t);

    %second way of getting world coordinates
    T = [Rc_w t;0 0 0 1];       %extrinsic parameters matrix
    cameraPoint = worldPtCam * s;  %image->camera
    worldPoint2 = inv(T)*[cameraPoint;ones(1,size(cameraPoint,2))]; %camera->world
    worldPoint2 = [worldPoint2(1,:);worldPoint2(2,:);worldPoint2(3,:)];
end
