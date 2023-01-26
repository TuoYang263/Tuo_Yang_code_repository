function robotic_coordinates = imageToRobot(camMatrix, world_mass_center, zConst, K, Rc_w, Pc,R)
    temp_image_center = camMatrix * [world_mass_center;ones(1, size(world_mass_center,2))];
    image_center(1,:) = temp_image_center(1,:)./ temp_image_center(3,:);
    image_center(2,:) = temp_image_center(2,:)./ temp_image_center(3,:);
    image_center
    [worldPoint1, worldPoint2] = back_projection(image_center', zConst, K, Rc_w, Pc);
    world_center = worldPoint1
    %fix the z-axis
    %world_center(3,:) = 25;
    %trnaslation vector T
    T = world_mass_center(:,end);
    transform_matrix = [R T;0 0 0 1];
    %build homogeneous coordinates
    robotic_coordinates = transform_matrix*...
                         [world_center(:,1:end-1);
                         ones(1,size(world_mass_center,2)-1)];
    robotic_coordinates = robotic_coordinates(1:3,:);
end
