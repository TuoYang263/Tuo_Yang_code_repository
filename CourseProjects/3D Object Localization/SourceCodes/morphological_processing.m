function processed_area = morphological_processing(img)
    %this function is used to get rid of noise made by segmenation results
    %use the morphological operation
    se = strel('disk',1);
    after_opening = imopen(img,se);
%     figure;
%     imshow(after_opening);
    after_closing = imopen(after_opening,se);
%     figure;
%     imshow(after_closing);
    %get rid of small objects
    processed_area = removeLargeArea(after_closing,10000);
    processed_area = bwareaopen(processed_area,800);
end