% randomly generate 500 points

a = 25;
b = 53;
c = 121;

y = [1:500]';
x = [1:500]';

y = y + randn(size(x));

z = a*x + b*y + c;
z = z + randn(size(x));

A = [x y ones(size(x))];

k = inv(A'*A)*A'*z;

fprintf("\nAsolute error is\n")
error = abs([a - k(1); b - k(2); c - k(3)])



