function ker = gaussiankernel(x1, x2, sigma)

ker = exp(-(x1 - x2) * (x1 - x2)'/ (2 * sigma^2))


end