clc,clear;

% 加载数据
[ images4Train, labels4Train ] = loadMNISTData( 'dataSet/train-images.idx3-ubyte',...
    'dataSet/train-labels.idx1-ubyte', 'MinMaxScaler', 0 );

% 要先把 AutoEncoder类中calc_cost_grad method后面的(Access = private)注释掉
[diff, numGradient, grad] = checkAE(images4Train); % 测试AutoEncoder类是否正确
fprintf(['AE中计算梯度的分析方法与数值方法的差异性：'...
    num2str(mean(abs(numGradient - grad)))...
    ' 及 ' num2str(diff) '\n']);

% 要先把 BackPropagation类中calc_cost_grad method后面的(Access = private)注释掉
[diff, numGradient, grad] = checkBP(images4Train, labels4Train);
fprintf(['BP中计算梯度的分析方法与数值方法的差异性：'...
    num2str(mean(abs(numGradient - grad)))...
    ' 及 ' num2str(diff) '\n']);








