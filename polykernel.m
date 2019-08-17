function ker = polykernel(x, z, degree)

m = x * z';
ker = m.^(degree);

end