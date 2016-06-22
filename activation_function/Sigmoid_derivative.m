function y = Sigmoid_derivative(x)
    y = Sigmoid(x) .* (1 - Sigmoid(x));
end