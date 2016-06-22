function y = softmax_derivative( x, labels )
    indexRow = labels';
    indexCol = 1:length(indexRow);
    index    = (indexCol - 1) * max(labels) + indexRow;
    
    y = softmax(x);
    y(index) = y(index) - 1;
end