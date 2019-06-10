using LinearAlgebra, StaticArrays

a1, a2, a3, a4 = randn(9), randn(9), randn(9), randn(9)
b1, b2, b3, b4 = randn(10), randn(10), randn(10), randn(10)
c1, c2, c3, c4 = randn(9), randn(9), randn(9), randn(9)
t = [Tridiagonal(a1, b1, c1), Tridiagonal(a2, b2, c2), Tridiagonal(a3, b3, c3), Tridiagonal(a4, b4, c4)];
T = zeros(20, 20);
T[1:2:20, 1:2:20] += t[1];
T[1:2:20, 2:2:20] += t[2];
T[2:2:20, 1:2:20] += t[3];
T[2:2:20, 2:2:20] += t[4];
d = randn(20);
x = zero(d);

x1 = T\d;
