function eculidean_distance = calculate_eculidean(sample1,sample2)
    %three dimensional eculidean distance calculation formula,
    %sample1 and sample2 respectively are 1 by 3 matrixes
    eculidean_distance = sqrt((sample1(1)-sample2(1))^2 + ...
        (sample1(2)-sample2(2))^2+(sample1(3)-sample2(3))^2);
end

