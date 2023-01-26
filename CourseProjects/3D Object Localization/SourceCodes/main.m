%% step 1:taking adavantage of color thresholding segmentation 
% to get mass centers coordinates of cubes, all mass centers can categoried into
% two parts, mass centers from pictures taken by the left camera and mass
% centers from pictures taken by the right camera
%the path used for reading images for calibration
clc,clear,close all;
img_dir = '.\\cubes-coords\\';
%the path used for storing segmentation results
segmentation_dir = '.\\cubes-coords-segmentation\\';
%the matrix used for storing thresholding values. For
%each column of every row, except for the first element presenting
%the number of the scene, left elements represent thresholding range of
%channels in different color spaces
thresh_range = [2 0.000 95.000 0.000 86.000 0.000 61.000;
                3 0 136 0 74 0 64;
                4 0 134 0 68 0 90;
                5 0 98 0 80 0 68;
                6 0 164 0 140 0 66;
                7 0 89 0 105 1 67;
                11 0.005 0.000 0.150 1.000 0.000 1.000;
                12 0.000 0.000 0.152 1.000 0.000 0.591;
                13 0.000 1.000 0.163 1.000 0.000 0.594;
                14 0 110 0 61 0 255];
%files format
ext = {'*.png'};
images = [];
for i = 1:length(ext)
    images = [images dir([img_dir ext{i}])];
end

% images are returned with absolute path and do some processing
scene_num = 1;
left_mass_centers = [];
right_mass_centers = [];
for i = 1:2:length(images)
    left_image_path = [img_dir images(i).name];
    right_image_path = [img_dir images(i+1).name];
    left_image = imread(left_image_path);
    right_image = imread(left_image_path);
    if scene_num == 1
        %do the segmentation to the left image of the first scene
        [R_BW_L,G_BW_L,B_BW_L,K_BW_L] = createfirst_scene_mask(left_image);
        R_BW_L = morphological_processing(R_BW_L);
        G_BW_L = morphological_processing(G_BW_L);
        B_BW_L = morphological_processing(B_BW_L);
        K_BW_L = morphological_processing(K_BW_L);
        left_segmentation_result = logical(R_BW_L)|logical(G_BW_L)|...
            logical(B_BW_L)|logical(K_BW_L);
        R_left_center = find_first_mass_centers(R_BW_L);
        G_left_center = find_first_mass_centers(G_BW_L);
        B_left_center = find_first_mass_centers(B_BW_L);
        K_left_center = find_first_mass_centers(K_BW_L);
        left_mass_centers = [left_mass_centers;R_left_center;
            G_left_center;B_left_center;K_left_center];
        gcf = figure;
        set(gcf,'visible','off');
        imshow(left_segmentation_result);
        hold on;
        draw_list = [R_left_center;G_left_center;B_left_center;K_left_center];
        text(draw_list(1,1),draw_list(1,2),num2str(1),'FontSize',10);
        hold on;
        text(draw_list(2,1),draw_list(2,2),num2str(2),'FontSize',10);
        hold on;
        text(draw_list(3,1),draw_list(3,2),num2str(3),'FontSize',10);
        hold on;
        text(draw_list(4,1),draw_list(4,2),num2str(4),'FontSize',10);
        hold on;
        saveas(gcf,[segmentation_dir,...
            '\scene_',num2str(scene_num),'_left.png']);
        
        %do the segmentation to the right image of the first scene
        [R_BW_R,G_BW_R,B_BW_R,K_BW_R] = createfirst_scene_mask(right_image);
        R_BW_R = morphological_processing(R_BW_R);
        G_BW_R = morphological_processing(G_BW_R);
        B_BW_R = morphological_processing(B_BW_R);
        K_BW_R = morphological_processing(K_BW_R);
        right_segmentation_result = logical(R_BW_R)|logical(G_BW_R)|...
            logical(B_BW_R)|logical(K_BW_R);
        imwrite(right_segmentation_result,[segmentation_dir,...
            '\scene_',num2str(scene_num),'_right.png']);
        R_right_center = find_first_mass_centers(R_BW_R);
        G_right_center = find_first_mass_centers(G_BW_R);
        B_right_center = find_first_mass_centers(B_BW_R);
        K_right_center = find_first_mass_centers(K_BW_R);
        right_mass_centers = [right_mass_centers;R_right_center;
            G_right_center;B_right_center;K_right_center];
        gcf = figure;
        set(gcf,'visible','off');
        imshow(right_segmentation_result);
        hold on;
        draw_list = [R_right_center;G_right_center;B_right_center;K_right_center];
        text(draw_list(1,1),draw_list(1,2),num2str(1),'FontSize',10);
        hold on;
        text(draw_list(2,1),draw_list(2,2),num2str(2),'FontSize',10);
        hold on;
        text(draw_list(3,1),draw_list(3,2),num2str(3),'FontSize',10);
        hold on;
        text(draw_list(4,1),draw_list(4,2),num2str(4),'FontSize',10);
        hold on;
        saveas(gcf,[segmentation_dir,...
            '\scene_',num2str(scene_num),'_right.png']);
        scene_num = scene_num + 1;
    elseif ismember(scene_num,[8 9 10])==0
        [row,col] = find(thresh_range == scene_num);
        if ismember(scene_num,[11 12 13])==1
            mark = 0;
            if ismember(scene_num,[11 12])==1
                mark = 1;
            else
                mark = 2;
            end
            [BW_left,mask_left] = createMask_HSV(left_image,...
                thresh_range(row,:),mark);
            BW_left = morphological_processing(BW_left);
            stats = regionprops(BW_left);
            centroids = cat(1,stats.Centroid);
            left_mass_centers = [left_mass_centers;centroids];
            gcf = figure;
            set(gcf,'visible','off');
            imshow(BW_left);
            hold on;
            text(centroids(1,1),centroids(1,2),num2str(1),'FontSize',10);
            hold on;
            text(centroids(2,1),centroids(2,2),num2str(2),'FontSize',10);
            hold on;
            text(centroids(3,1),centroids(3,2),num2str(3),'FontSize',10);
            hold on;
            text(centroids(4,1),centroids(4,2),num2str(4),'FontSize',10);
            hold on;
            saveas(gcf,[segmentation_dir,...
            '\scene_',num2str(scene_num),'_left.png']);
            
            
            [BW_right,mask_right] = createMask_HSV(right_image,...
                thresh_range(row,:),mark);
            BW_right = morphological_processing(BW_right);
            stats = regionprops(BW_right);
            centroids = cat(1,stats.Centroid);
            right_mass_centers = [right_mass_centers;centroids];
            gcf = figure;
            set(gcf,'visible','off');
            imshow(BW_right);
            hold on;
            text(centroids(1,1),centroids(1,2),num2str(1),'FontSize',10);
            hold on;
            text(centroids(2,1),centroids(2,2),num2str(2),'FontSize',10);
            hold on;
            text(centroids(3,1),centroids(3,2),num2str(3),'FontSize',10);
            hold on;
            text(centroids(4,1),centroids(4,2),num2str(4),'FontSize',10);
            hold on;
            saveas(gcf,[segmentation_dir,...
            '\scene_',num2str(scene_num),'_right.png']);
        else
            [BW_left,mask_left] = createMask_RGB(left_image,...
                thresh_range(row,:));
            BW_left = morphological_processing(BW_left);
            stats = regionprops(BW_left);
            centroids = cat(1,stats.Centroid);
            left_mass_centers = [left_mass_centers;centroids];
            gcf = figure;
            set(gcf,'visible','off');
            imshow(BW_left);
            hold on;
            text(centroids(1,1),centroids(1,2),num2str(1),'FontSize',10);
            hold on;
            text(centroids(2,1),centroids(2,2),num2str(2),'FontSize',10);
            hold on;
            text(centroids(3,1),centroids(3,2),num2str(3),'FontSize',10);
            hold on;
            text(centroids(4,1),centroids(4,2),num2str(4),'FontSize',10);
            hold on;
            saveas(gcf,[segmentation_dir,...
            '\scene_',num2str(scene_num),'_left.png']);
            
            
            [BW_right,mask_right] = createMask_RGB(right_image,...
                thresh_range(row,:));
            BW_right = morphological_processing(BW_right);
            stats = regionprops(BW_right);
            centroids = cat(1,stats.Centroid);
            right_mass_centers = [right_mass_centers;centroids];
            gcf = figure;
            set(gcf,'visible','off');
            imshow(BW_right);
            hold on;
            text(centroids(1,1),centroids(1,2),num2str(1),'FontSize',10);
            hold on;
            text(centroids(2,1),centroids(2,2),num2str(2),'FontSize',10);
            hold on;
            text(centroids(3,1),centroids(3,2),num2str(3),'FontSize',10);
            hold on;
            text(centroids(4,1),centroids(4,2),num2str(4),'FontSize',10);
            hold on;
            saveas(gcf,[segmentation_dir,...
            '\scene_',num2str(scene_num),'_right.png']);
        end
        scene_num = scene_num + 1;
    else
        scene_num = scene_num + 1;
    end
end

%% step 2:taking advantage of pixel coordinates and world coordinates
%to get projection matrix, two situations will be divided here: projection 
%matrix for the left camera and the projection matrix for the right camera
%use the scene 2 to get the calibration matrix, use the scene 3 as the test
%image

%for scene 2
%the order of cubes:red, green, blue, black
clc,clear,close all;
scene2_mass_center = [170 110 25;
                     330 182 25;
                     260 270 25;
                     150 250 25];
%calculate the corner's coordinates according to 
X = [170-25 170+25 170+25 330-25 330+25 330+25 260-25 260+25 260+25 150-25 150+25 150+25];
Y = [110+25 110+25 110-25 182+25 182+25 182-25 270+25 270+25 270-25 250+25 250+25 250-25];
Z = [25+25 25+25 25-25 25+25 25+25 25-25 25+25 25+25 25-25 25+25 25+25 25-25];
scene2_world_coordinates = [X' Y' Z'];
scene2_image = imread('.\data\100838-l.png');
figure;
imshow(scene2_image);
[xx, yy, button] = ginput(size(X,2));
scene2_pixels = [xx,yy];
save calibration.mat scene2_world_coordinates scene2_mass_center scene2_pixels

%collecting info used for testing calibration
%cube order:red, green, blue, black
scene5_mass_center = [170 110 25;
                      90 290 25;
                      10 470 25;
                      420 560 25];
X = [170-25 170+25 170+25 90-25 90+25 90+25 10-25 10+25 10+25 420-25 420+25 420+25];
Y = [110+25 110+25 110-25 290+25 290+25 290-25 470+25 470+25 470-25 560+25 560+25 560-25];
Z = [25+25 25+25 25-25 25+25 25+25 25-25 25+25 25+25 25-25 25+25 25+25 25-25];

scene5_world_coordinates = [X' Y' Z'];
scene5_image = imread('.\data\101407-l.png');
figure;
imshow(scene5_image);
[xx, yy, button] = ginput(size(X,2));
scene5_pixels = [xx,yy];
save calibration_test.mat scene5_world_coordinates scene5_mass_center scene5_pixels

%% step 3 cameraCalibration and back projection
clc,clear,close all;
load calibration.mat;

scene2_pixels
[camMatrix,projected_points] = calibrate(scene2_world_coordinates, scene2_pixels)
scene2_image = imread('.\cubes-coords-segmentation\scene_2_left.png');
scene2_truecolor = imread('.\cubes-coords\8130478-2021-04-07-100838-l.png');

%do the decomposition to the matrix M
[K, Rc_w, Pc, pp, pv] = Pdecomp(camMatrix);
new_camMatrix = K*[Rc_w -Rc_w*Pc]

%convert pixel coordinates into camera coordinates
%[x,y,z] = invK*[u,v,1]
scene2_cubes_centers = [270.221073691230,343.524508169390;
                        414.053418803419,278.418447293447;
                        348.948965517241,217.674022988506;
                        255.802576891106,236.624688279302];

figure;
imshow(scene2_truecolor);
hold on;
plot(scene2_cubes_centers(:,1), scene2_cubes_centers(:,2),'r*');
hold on;
plot(scene2_pixels(:,1), scene2_pixels(:,2),'b*');

%do the same thing to images which has the robotics block
%use the first image as the example
zConst = 25;
[cube_worldPoint1, cube_worldPoint2] = back_projection(scene2_cubes_centers, zConst, K, Rc_w, Pc)
%Experiment with robotics images
%1st image
world_mass_center1 = [50 550 200 400 450;
                      350 350 200 550 100;
                      25 25 25 25 25];
%rotation matrix R
%definition
%R = [cos(x,xb) cos(x,yb) cos(x,zb)
%     cos(y,xb) cos(y,yb) cos(y,zb)
%     cos(z,xb) cos(z,yb) cos(z,zb)];
R = [cosd(90) cosd(90) cosd(0);
     cosd(0) cosd(90) cosd(90);
     cosd(90) cosd(0) cosd(90)];
robotic_coordinates1 = imageToRobot(camMatrix, world_mass_center1, zConst, K, Rc_w, Pc, R);

%2nd image
world_mass_center2 = [50 550 200 400 50;
                      350 350 200 550 100;
                      25 25 25 25 25];
%rotation matrix R
%definition
%R = [cos(x,xb) cos(x,yb) cos(x,zb)
%     cos(y,xb) cos(y,yb) cos(y,zb)
%     cos(z,xb) cos(z,yb) cos(z,zb)];
R = [cosd(0) cosd(90) cosd(90);
     cosd(90) cosd(90) cosd(180);
     cosd(90) cosd(0) cosd(90)];
robotic_coordinates2 = imageToRobot(camMatrix, world_mass_center2, zConst, K, Rc_w, Pc, R);

%3rd image
world_mass_center3 = [50 550 200 150 50;
                      350 350 200 550 100;
                      25 25 25 25 25];
%rotation matrix R
%definition
%R = [cos(x,xb) cos(x,yb) cos(x,zb)
%     cos(y,xb) cos(y,yb) cos(y,zb)
%     cos(z,xb) cos(z,yb) cos(z,zb)];
R = [cosd(0) cosd(90) cosd(90);
     cosd(90) cosd(90) cosd(180);
     cosd(90) cosd(0) cosd(90)];
robotic_coordinates3 = imageToRobot(camMatrix, world_mass_center3, zConst, K, Rc_w, Pc, R);

%4th image
world_mass_center4 = [50 300 200 150 50;
                      350 350 200 550 100;
                      25 25 25 25 25];
%rotation matrix R
%definition
%R = [cos(x,xb) cos(x,yb) cos(x,zb)
%     cos(y,xb) cos(y,yb) cos(y,zb)
%     cos(z,xb) cos(z,yb) cos(z,zb)];
R = [cosd(0) cosd(90) cosd(90);
     cosd(90) cosd(90) cosd(180);
     cosd(90) cosd(0) cosd(90)];
robotic_coordinates4 = imageToRobot(camMatrix, world_mass_center4, zConst, K, Rc_w, Pc, R);

%5th image
world_mass_center5 = [50 300 200 150 400;
                      350 350 200 550 450;
                      25 25 25 25 25];
%rotation matrix R
%definition
%R = [cos(x,xb) cos(x,yb) cos(x,zb)
%     cos(y,xb) cos(y,yb) cos(y,zb)
%     cos(z,xb) cos(z,yb) cos(z,zb)];
R = [cosd(180) cosd(90) cosd(90);
     cosd(90) cosd(90) cosd(0);
     cosd(90) cosd(0) cosd(90)];
robotic_coordinates5 = imageToRobot(camMatrix, world_mass_center5, zConst, K, Rc_w, Pc, R);

%6th image
world_mass_center6 = [500 300 200 150 400;
                      500 350 200 550 450;
                      25 25 25 25 25];
%rotation matrix R
%definition
%R = [cos(x,xb) cos(x,yb) cos(x,zb)
%     cos(y,xb) cos(y,yb) cos(y,zb)
%     cos(z,xb) cos(z,yb) cos(z,zb)];
R = [cosd(90) cosd(90) cosd(0);
     cosd(180) cosd(90) cosd(90);
     cosd(90) cosd(0) cosd(90)];
robotic_coordinates6 = imageToRobot(camMatrix, world_mass_center6, zConst, K, Rc_w, Pc, R);

%7th image
world_mass_center7 = [500 300 200 150 400;
                      500 350 100 150 450;
                      25 25 25 25 25];
%rotation matrix R
%definition
%R = [cos(x,xb) cos(x,yb) cos(x,zb)
%     cos(y,xb) cos(y,yb) cos(y,zb)
%     cos(z,xb) cos(z,yb) cos(z,zb)];
R = [cosd(90) cosd(90) cosd(0);
     cosd(180) cosd(90) cosd(90);
     cosd(90) cosd(0) cosd(90)];
robotic_coordinates7 = imageToRobot(camMatrix, world_mass_center7, zConst, K, Rc_w, Pc, R);

%8th image
world_mass_center8 = [500 300 200 150 400;
                      500 350 100 250 450;
                      25 25 25 25 25];
%rotation matrix R
%definition
%R = [cos(x,xb) cos(x,yb) cos(x,zb)
%     cos(y,xb) cos(y,yb) cos(y,zb)
%     cos(z,xb) cos(z,yb) cos(z,zb)];
R = [cosd(90) cosd(90) cosd(0);
     cosd(180) cosd(90) cosd(90);
     cosd(90) cosd(0) cosd(90)];
robotic_coordinates8 = imageToRobot(camMatrix, world_mass_center8, zConst, K, Rc_w, Pc, R);

%9th image
world_mass_center9 = [500 450 200 150 400;
                      500 250 100 250 450;
                      25 25 25 25 25];
%rotation matrix R
%definition
%R = [cos(x,xb) cos(x,yb) cos(x,zb)
%     cos(y,xb) cos(y,yb) cos(y,zb)
%     cos(z,xb) cos(z,yb) cos(z,zb)];
R = [cosd(90) cosd(90) cosd(0);
     cosd(180) cosd(90) cosd(90);
     cosd(90) cosd(0) cosd(90)];
robotic_coordinates9 = imageToRobot(camMatrix, world_mass_center9, zConst, K, Rc_w, Pc, R);

%10th image
world_mass_center10 = [300 450 200 150 450;
                      350 250 100 250 450;
                      25 25 25 25 25];
%rotation matrix R
%definition
%R = [cos(x,xb) cos(x,yb) cos(x,zb)
%     cos(y,xb) cos(y,yb) cos(y,zb)
%     cos(z,xb) cos(z,yb) cos(z,zb)];
R = [cosd(0) cosd(90) cosd(90);
     cosd(90) cosd(90) cosd(180);
     cosd(90) cosd(0) cosd(90)];
robotic_coordinates10 = imageToRobot(camMatrix, world_mass_center10, zConst, K, Rc_w, Pc, R);

%11th image
world_mass_center11 = [300 300 200 150 450;
                      350 350 100 250 450;
                      25 25 25 25 25];
%rotation matrix R
%definition
%R = [cos(x,xb) cos(x,yb) cos(x,zb)
%     cos(y,xb) cos(y,yb) cos(y,zb)
%     cos(z,xb) cos(z,yb) cos(z,zb)];
R = [cosd(0) cosd(90) cosd(90);
     cosd(90) cosd(90) cosd(180);
     cosd(90) cosd(0) cosd(90)];
robotic_coordinates11 = imageToRobot(camMatrix, world_mass_center11, zConst, K, Rc_w, Pc, R);

%12th image
world_mass_center12 = [300 200 200 150 450;
                      350 150 100 250 450;
                      25 25 25 25 25];
%rotation matrix R
%definition
%R = [cos(x,xb) cos(x,yb) cos(x,zb)
%     cos(y,xb) cos(y,yb) cos(y,zb)
%     cos(z,xb) cos(z,yb) cos(z,zb)];
R = [cosd(0) cosd(90) cosd(90);
     cosd(90) cosd(90) cosd(180);
     cosd(90) cosd(0) cosd(90)];
robotic_coordinates12 = imageToRobot(camMatrix, world_mass_center12, zConst, K, Rc_w, Pc, R);

%13th image
world_mass_center13 = [300 200 260 150 400;
                      350 150 270 250 100;
                      25 25 25 25 25];
%rotation matrix R
%definition
%R = [cos(x,xb) cos(x,yb) cos(x,zb)
%     cos(y,xb) cos(y,yb) cos(y,zb)
%     cos(z,xb) cos(z,yb) cos(z,zb)];
R = [cosd(90) cosd(90) cosd(180);
     cosd(180) cosd(90) cosd(90);
     cosd(90) cosd(0) cosd(90)];
robotic_coordinates13 = imageToRobot(camMatrix, world_mass_center13, zConst, K, Rc_w, Pc, R);


