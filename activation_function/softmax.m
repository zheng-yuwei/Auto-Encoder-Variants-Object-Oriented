function y = softmax(x)
    y = exp(x);
    y = bsxfun(@rdivide, y, sum(y, 1));
end