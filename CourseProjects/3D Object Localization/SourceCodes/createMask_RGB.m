function [BW,maskedRGBImage] = createMask_RGB(RGB, thresh_range)
    % Convert RGB image to chosen color space
    I = RGB;

    % Define thresholds for channel 1 based on histogram settings
    channel1Min = thresh_range(2);
    channel1Max = thresh_range(3);

    % Define thresholds for channel 2 based on histogram settings
    channel2Min = thresh_range(4);
    channel2Max = thresh_range(5);

    % Define thresholds for channel 3 based on histogram settings
    channel3Min = thresh_range(6);
    channel3Max = thresh_range(7);

    % Create mask based on chosen histogram thresholds
    sliderBW = (I(:,:,1) >= channel1Min ) & (I(:,:,1) <= channel1Max) & ...
        (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
        (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
    BW = sliderBW;

    % Initialize output masked image based on input image.
    maskedRGBImage = RGB;

    % Set background pixels where BW is false to zero.
    maskedRGBImage(repmat(~BW,[1 1 3])) = 0;

end
