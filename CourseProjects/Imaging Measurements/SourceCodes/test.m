clc,clear,close all;

x = -3:0.1:3;
y = -3:0.1:3;
[x, y] = meshgrid(x, y);
z = (x * y)./(x + y);
surf(x, y, z);
xlabel('x');
ylabel('y');
zlabel('z');