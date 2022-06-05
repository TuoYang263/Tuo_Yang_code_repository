function classification_visualization(testing_dataset,testing_class, C)
    %this function is used for computing rightly classified rate, and visualize
    %classification results by using these three input variables and plots
    disp([num2str(length(testing_class.testing_class)),...
    ' samples correctly-classified rate:%',...
    num2str(sum(C.C == testing_class.testing_class)/...
    length(testing_class.testing_class)*100)]);

    figure_num = fix(length(testing_class.testing_class)/100)+1;
    k = 1;
    for i = 1:figure_num
        h = figure;
        set(h,'outerposition',get(0,'screensize'));
        sgtitle(['Classification results for ',...
            num2str(length(testing_class.testing_class)),' samples']);
        for j = 1:100
            subplot(10,10,j);
            plot(testing_dataset.testing_dataset{1,k}(:,1),...
            testing_dataset.testing_dataset{1,k}(:,2));
            if C.C(k) ~= testing_class.testing_class(k)
                title(['Prediction label:',num2str(C.C(k)-1)],'color','r');
            else
                title(['Prediction label:',num2str(C.C(k)-1)]);
            end
            k = k + 1;
            if k > length(testing_class.testing_class)
                break;
            end
        end
        print(h,'-djpeg',['.\classification_results\result_',num2str(i),...
            '_figure.jpg']);
    end
end