function data_visualization(digit_sample_num, file_names,file_num)
    %function used for visualizing sample data in 2D plots to find
    %abnormal samples indexes
    for i = 1:digit_sample_num:file_num-digit_sample_num+1
        gcf = figure;
        %maxmize the figure window
        set(gcf,'outerposition',get(0,'screensize'));
        sgtitle(['Digits ',num2str(floor(i/100)),' two dimensional stroke data']);
        k = 1;         %subplot start index
        for j = i:i+digit_sample_num-1
            subplot(10,10,k);
            location_data = load([file_names(j).folder,'\', file_names(j).name]);
            plot(location_data.pos(:,1), location_data.pos(:,2));
            title(['sample ',num2str(j)]);
            k = k + 1;
        end
        %take advantage of print function to save maxmized window 
        print(gcf,'-djpeg',['.\digits_figures\stroke_',num2str(floor(i/100)),...
            '_figure.jpg']);
    end
end