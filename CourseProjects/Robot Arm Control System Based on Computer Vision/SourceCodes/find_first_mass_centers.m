function mass_center = find_first_mass_centers(binary_image)
    [row, col]= find(binary_image==1);
    pixel_coord = [min(col),min(row),max(col),...
        max(row)];
    mass_center = [round((pixel_coord(1) + pixel_coord(3))/2),...
        round((pixel_coord(2) + pixel_coord(4))/2)];
end
