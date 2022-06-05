function coins = estim_coins(measurement, bias, dark, flat)
    %input:
    %measurement:one image to be recognized
    %bias:one processed bias image
    %dark:one processed dark image
    %flat:one processed flat image
    %coin_off_checkerboard:signal if the coin is off the checkerboard
    %output:
    %coins:one 1 by 6 vector, which records the number of 2 euro,
    %50 euro,1 euro, 20 euro,5 euro and 10 euro respectively
    magnification = 25;         %enlarge factor of the image
    figure;
    imshow(measurement,'InitialMagnification',magnification);
    title('Raw image to be measured');
    
    %illumination calibration process
    illumination_calibrated = (measurement - bias - dark)./flat;
    figure;
    imshow(illumination_calibrated,'InitialMagnification',magnification);
    title('Image after the illumination calibration process');

    measurement = illumination_calibrated;

    %because the geometric calibration requires at least two images
    %I got no ways but do like this
    measurement_set(:,:,:,1) = measurement;
    measurement_set(:,:,:,2) = measurement;

    %Detect the chessboard corners in the images
    [imagePoints, boardSize] = detectCheckerboardPoints(measurement_set);

    %Generate the world coordinates of the checkerboard corners
    %in the pattern-centric coordinate system, with the upper-left
    %corner at (0,0)
    squareSize = 12.5;      %in millimeters 23.25 cents(14/25)
    worldPoints = generateCheckerboardPoints(boardSize, squareSize);

    %Calibrate the camera
    imageSize = [size(measurement,1),size(measurement,2)];
    cameraParams = estimateCameraParameters(imagePoints,...
        worldPoints,'ImageSize',imageSize);

    %Evaluate cabibration accuracy
    figure;
    showReprojectionErrors(cameraParams);
    title('Reprojection Errors');

    %because the lens introduced little distortion, use 'full'
    %output view to illustrate that the image was undistored
    [calibrated_image,newOrigin] = undistortImage(measurement,cameraParams,...
        'OutputView','full');
    %display the image after geometric calibration
    figure;
    imshow(calibrated_image,'InitialMagnification',magnification);
    title('Undistorted Image');

    % %adjust illumination intensity using imadjust
    % R = calibrated_image(:,:,1);
    % G = calibrated_image(:,:,2);
    % B = calibrated_image(:,:,3);
    % maxR = im2double(max(max(R)));
    % maxG = im2double(max(max(G)));
    % maxB = im2double(max(max(B)));
    % k = (maxR + maxG + maxB)/3;
    % %after serveral times' trying, the threshold of k is 0.98
    % if k < 0.98
    %     RGB2 = imadjust(calibrated_image,[0 0 0; k k k],[0 0 0;1 1 1],0.6);
    % else
    %     RGB2 = calibrated_image;
    % end
    % figure;
    % imshow(RGB2);

    %Convert the image to the HSV color space
    imHSV = rgb2hsv(calibrated_image);

    %Get the saturation channel
    saturation = imHSV(:,:,2);

    %get the optimal threshold value of the image
    thresh = graythresh(saturation);    %OSTU method
    coin_image = (saturation > thresh/1.5);
    %use this image's saturation channel in the HSV space to select 
    %optimal threshold value used for binarization 
    figure;
    imshow(coin_image,'InitialMagnification',magnification);
    title('Binarization results in HSV saturation space');
    
    img_fill = coin_image;
    
    % Remove small objects (noise) and fill complete objects
    img_clearborder = imclearborder(img_fill, 6);
    img_fill = imfill(img_clearborder, 'holes');
    figure, imshow(img_fill,'InitialMagnification',magnification);
    title('Image after filling holes');
    
    %use the opening operation 
    erodeElement = strel('disk',25);
    img_fill = imerode(img_fill, erodeElement);
    dilateElement = strel('disk',25);
    img_fill = imdilate(img_fill, dilateElement);
    figure;
    imshow(img_fill,'InitialMagnification',magnification);
    title('After opening operation');
    
    %use the bwareaopen to get rid of all white area less than 80000
    %square pixels
    img_fill = bwareaopen(img_fill,80000);
    figure;
    imshow(img_fill,'InitialMagnification',magnification);
    title('After getting rid of small areas');

    %extract borders from binary image and visualize them
    [B,L] = bwboundaries(img_fill);
    figure;
    imshow(label2rgb(L,@jet,[.5 .5 .5]));
    title('boundaries');
    hold on;
    for k = 1:size(B,1)
        boundary = B{k,1};
        plot(boundary(:,2),boundary(:,1),'w','LineWidth',2);
    end

    %return corresponding info of connected area
    connected_bw = logical(img_fill);

    %use regionprops to find all connected areas
    stats = regionprops('table',connected_bw,'Centroid',...
        'MajorAxisLength','MinorAxisLength','Area');

    centers  = stats.Centroid;
    diameters = mean([stats.MajorAxisLength stats.MinorAxisLength],2);
    radius = diameters/2;

    figure;
    imshow(img_fill, 'InitialMagnification', magnification);
    title('All coins having been detected');
    hold on;
    viscircles(centers,radius);
    hold off;

    %use coins' area to judge the type of the coin
    %area's comparsion:2 euro > 50 euro > 1 euro> 20 euro
    %>5 euro> 10 euro
    coin_counter = zeros(1,6);  %build 1 by 6 vector as the counter of coins
    area = round(pi.*(radius.^2));        %all coins' areas
    for i = 1:size(area,1)
        if area(i,1)<220000 && area(i,1)>200000     %collect 2 euro coins   
            coin_counter(1) = coin_counter(1) + 1;
        elseif area(i,1)<200000 && area(i,1)>180000 %collect 50 cent coins
            coin_counter(3) = coin_counter(3) + 1;
        elseif area(i,1)<180000 && area(i,1)>165000 %collect 1 euro coins
            coin_counter(2) = coin_counter(2) + 1;
        elseif area(i,1)<165000 && area(i,1)>150000 %collect 20 cent coins
            coin_counter(4) = coin_counter(4) + 1;
        elseif area(i,1)<150000 && area(i,1)>140000 %collect 5 euro coins
            coin_counter(6) = coin_counter(6) + 1;
        elseif area(i,1)<130000 && area(i,1)>115000 %collect 10 cent coins
            coin_counter(5) = coin_counter(5) + 1;
        else 
            coin_counter(1) = coin_counter(1) + 1;  %abnormal samples  
        end
    end
    coins = coin_counter;
end