function [R_BW,G_BW,B_BW,K_BW] = createfirst_scene_mask(RGB)
    % Convert RGB image to chosen color space
    I = RGB;
    I_HSV = rgb2hsv(RGB);

    % Define thresholds for channel 1 based on histogram settings
    R_channel1Min = 23.000;
    R_channel1Max = 219.000;
    B_channel1Min = 0.000;
    B_channel1Max = 23.000;
    G_channel1Min = 0.147;
    G_channel1Max = 0.611;
    K_channel1Min = 1.000;
    K_channel1Max = 30.000;

    % Define thresholds for channel 2 based on histogram settings
    R_channel2Min = 0.000;
    R_channel2Max = 30.000;
    B_channel2Min = 0.000;
    B_channel2Max = 25.000;
    G_channel2Min = 0.044;
    G_channel2Max = 0.596;
    K_channel2Min = 0.000;
    K_channel2Max = 31.000;
    
    % Define thresholds for channel 3 based on histogram settings
    R_channel3Min = 0.000;
    R_channel3Max = 255.000;
    B_channel3Min = 24.000;
    B_channel3Max = 255.000;
    G_channel3Min = 0.000;
    G_channel3Max = 0.291;
    K_channel3Min = 0.000;
    K_channel3Max = 26.000;
    

    % Create mask based on chosen histogram thresholds
    R_sliderBW = (I(:,:,1) >= R_channel1Min ) & (I(:,:,1) <= R_channel1Max) & ...
        (I(:,:,2) >= R_channel2Min ) & (I(:,:,2) <= R_channel2Max) & ...
        (I(:,:,3) >= R_channel3Min ) & (I(:,:,3) <= R_channel3Max);
    R_BW = R_sliderBW;
    
    G_sliderBW = (I_HSV(:,:,1) >= G_channel1Min ) & (I_HSV(:,:,1) <= G_channel1Max) & ...
        (I_HSV(:,:,2) >= G_channel2Min ) & (I_HSV(:,:,2) <= G_channel2Max) & ...
        (I_HSV(:,:,3) >= G_channel3Min ) & (I_HSV(:,:,3) <= G_channel3Max);
    G_BW = G_sliderBW;
    
    B_sliderBW = (I(:,:,1) >= B_channel1Min ) & (I(:,:,1) <= B_channel1Max) & ...
        (I(:,:,2) >= B_channel2Min ) & (I(:,:,2) <= B_channel2Max) & ...
        (I(:,:,3) >= B_channel3Min ) & (I(:,:,3) <= B_channel3Max);
    B_BW = B_sliderBW;
    
    K_sliderBW = (I(:,:,1) >= K_channel1Min ) & (I(:,:,1) <= K_channel1Max) & ...
        (I(:,:,2) >= K_channel2Min ) & (I(:,:,2) <= K_channel2Max) & ...
        (I(:,:,3) >= K_channel3Min ) & (I(:,:,3) <= K_channel3Max);
    K_BW = K_sliderBW;

    % Initialize output masked image based on input image.
    R_maskedRGBImage = RGB;
    G_maskedRGBImage = RGB;
    B_maskedRGBImage = RGB;
    K_maskedRGBImage = RGB;
    
    % Set background pixels where BW is false to zero.
    R_maskedRGBImage(repmat(~R_BW,[1 1 3])) = 0;
    G_maskedRGBImage(repmat(~G_BW,[1 1 3])) = 0;
    B_maskedRGBImage(repmat(~B_BW,[1 1 3])) = 0;
    K_maskedRGBImage(repmat(~K_BW,[1 1 3])) = 0;
end
