function [M, xnew] = calibration2(X3, x2)

[U, X] = myNormalization(X3);
[T, x] = myNormalization(x2);
X = X'; x = x';

j = 1;
for i = 1:size(x,1)
    A (j, :) = [X(i,:) 1 0 0 0 0 -x(i,1).*X(i,:) -x(i,1)];
    A (j + 1, :) = [0 0 0 0 X(i,:) 1 -x(i,2).*X(i,:) -x(i,2)];
    j = j + 2;
end
size(x,1)
[UU, DD, V]= svd(A);
u = V(:, end);
M_norm = reshape(u, 4, 3)';
M = inv(T)*M_norm*U;

p3 = X3';
p3 = [p3; ones(1, size(p3,2))];

xp = M * p3;
xnew = [xp(1,:)./xp(3,:); xp(2,:)./xp(3,:)];

end