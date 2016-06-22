function y = leaky_ReLU_derivative(x, alpha)
    y = x;
    y(y < 0) = alpha;
    y(y > 0) = 1;
end