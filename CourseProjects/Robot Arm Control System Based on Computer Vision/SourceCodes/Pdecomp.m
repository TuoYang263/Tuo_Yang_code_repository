function [K, Rc_w, Pc, pp, pv] = Pdecomp(P)
    % DECOMPOSECAMERA  Decomposition of a camera projection matrix
    %
    % Usage:  [K, Rc_w, Pc, pp, pv] = decomposecamera(P);
    %
    %    P is decomposed into the form P = K*[R -R*Pc]
    %
    % Argument:  P - 3 x 4 camera projection matrix
    % Returns:   
    %            K - Calibration matrix of the form
    %                  |  ax   s   ppx |
    %                  |   0   ay  ppy |
    %                  |   0   0    1  |
    %
    %                Where: 
    %                ax = f/pixel_width and ay = f/pixel_height,
    %                ppx and ppy define the principal point in pixels,
    %                s is the camera skew.
    %         Rc_w - 3 x 3 rotation matrix defining the world coordinate frame
    %                in terms of the camera frame. Columns of R transposed define
    %                the directions of the camera X, Y and Z axes in world
    %                coordinates. 
    %           Pc - Camera centre position in world coordinates.
    %           pp - Image principal point.
    %           pv - Principal vector  from the camera centre C through pp
    %                pointing out from the camera.  This may not be the same as  
    %                R'(:,3) if the principal point is not at the centre of the
    %                image, but it should be similar. 
    p1 = P(:,1);
    p2 = P(:,2);
    p3 = P(:,3);
    p4 = P(:,4);    
    M = [p1 p2 p3];
    m3 = M(3,:)';

    % Camera centre, analytic solution
    X =  det([p2 p3 p4]);
    Y = -det([p1 p3 p4]);
    Z =  det([p1 p2 p4]);
    T = -det([p1 p2 p3]);    

    Pc = [X;Y;Z;T];  
    Pc = Pc/Pc(4);   
    Pc = Pc(1:3);     % Make inhomogeneous

    % Pc = null(P,'r'); % numerical way of computing C

    % Principal point
    pp = M*m3;
    pp = pp/pp(3); 
    pp = pp(1:2);   % Make inhomogeneous

    % Principal ray pointing out of camera
    pv = det(M)*m3;
    pv = pv/norm(pv);

    % Perform RQ decomposition of M matrix. Note that rq3 returns K with +ve
    % diagonal elements, as required for the calibration matrix.
    [K,Rc_w] = RQ3(M);
end

