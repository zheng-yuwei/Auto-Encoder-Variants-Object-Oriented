function y = ReLU_derivative(x)
    y = x;
    y(y < 0) = 0;
    y(y > 0) = 1;
end