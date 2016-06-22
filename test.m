%% 测试BPNN
clear,clc;
input  = [rand(30, 20) + 4, rand(30, 20) + 8, rand(30, 20) + 12, rand(30, 20) + 16];
target = [ones(20, 1); ones(20, 1) * 2; ones(20, 1) * 3; ones(20, 1) * 4];
architecture = [30, 20, 4];
activations = {'ReLU', 'softmax'};

BackPropagation.description();
bpnn = BackPropagation(architecture, activations);
bpnn.train(input, target, 1000);
bpnn.test(input, target)
result = bpnn.predict(input);

% 用原来文件验证 calc_cost_grad
% delta = 1e-3;
% grad_test = zeros(size(bpnn.theta));
% theta = bpnn.theta;
% for i = 1:length(bpnn.theta)
%     theta(i)   = theta(i) + delta;
%     [cost1, ~] = bpnn.calc_cost_grad(input, target, theta);
%     theta(i)   = theta(i) - delta;
%     [cost2, ~] = bpnn.calc_cost_grad(input, target, theta);
%     grad_test(i) = (cost1 - cost2) / delta;
% end
% 
% [cost, grad] = bpnn.calc_cost_grad(input, target, bpnn.theta);
% 
% mean(abs(grad - grad_test))
%% 测试AENN
clear,clc;
input = rand(30, 80);
architecture = [30, 29, 30];
activations = {'ReLU', 'ReLU'};
options.is_sparse    = 1;
options.sparse_rho   = 0.1;
options.sparse_beta  = 0.03;
options.is_denoising = 1;
options.noise_rate   = 0.15;
options.noise_mode   = 'on_off';
options.is_weighted_cost = 1;
options.weighted_cost    = rand(sum(architecture(1:(end-2))), 1) * 2; % [0,2]之间的权重

AutoEncoder.description();
aenn = AutoEncoder(architecture, activations, options);
aenn.train(input, 5000);
aenn.test(input)
predict_input = aenn.predict(input);
sum(sum(abs(input - predict_input))) / size(input,2) / size(input,1) % 大概为0.0X，因为rand每位为0.5，所以误差为1/10

% 用原来文件验证 calc_cost_grad
% delta = 1e-3;
% grad_test = zeros(size(aenn.theta));
% theta = aenn.theta;
% for i = 1:length(aenn.theta)
%     theta(i)   = theta(i) + delta;
%     [cost1, ~] = aenn.calc_cost_grad(input, theta);
%     theta(i)   = theta(i) - delta;
%     [cost2, ~] = aenn.calc_cost_grad(input, theta);
%     grad_test(i) = (cost1 - cost2) / delta;
% end
% 
% [cost, grad] = aenn.calc_cost_grad(input, aenn.theta);
% 
% mean(abs(grad - grad_test))

%% 测试SAE
clear,clc;
input  = [rand(30, 20) + 4, rand(30, 20) + 8, rand(30, 20) + 12, rand(30, 20) + 16];
target = [ones(20, 1); ones(20, 1) * 2; ones(20, 1) * 3; ones(20, 1) * 4];
architecture = [30, 25, 20, 4];
activations = {'ReLU', 'ReLU', 'softmax'};
options.is_sparse    = 0; % 因为input得到的AE中间层大于1，用KL散度概率不好测量
options.sparse_rho   = 0.1;
options.sparse_beta  = 0.03;
options.is_denoising = 1;
options.noise_rate   = 0.15;
options.noise_mode   = 'on_off';
options.is_weighted_cost = 1;
options.weighted_cost    = rand(sum(architecture(1:(end-2))), 1) * 2; % [0,2]之间的权重

Stacked_AutoEncoder.description();
sae = Stacked_AutoEncoder(architecture, activations, options);
sae.train(input, target);
sae.test(input, target)
result = sae.predict(input);

sae.train(input, target, 500, 500, sae.theta4AEs);
sae.test(input, target)
result = sae.predict(input);

%% 测试GAE
clear,clc;
input = rand(30, 80);
architecture = [30, 28, 20, 28, 30];
activations = {'ReLU', 'ReLU', 'ReLU', 'ReLU'};
options.is_sparse    = 1;
options.sparse_rho   = 0.1;
options.sparse_beta  = 0.03;
options.is_denoising = 1;
options.noise_rate   = 0.15;
options.noise_mode   = 'on_off';
% options.is_weighted_cost = 1;
% options.weighted_cost    = rand(sum(architecture(1:(length(architecture) - 1) / 2)), 1) * 2; % [0,2]之间的权重

Generative_AutoEncoder.description();
gaenn = Generative_AutoEncoder(architecture, activations, options);
gaenn.train(input, 500, 1000);
gaenn.test(input)
predict_input = gaenn.predict(input);
sum(sum(abs(input - predict_input))) / size(input,2) / size(input,1) % 大概为0.0X，因为rand每位为0.5，所以误差为1/10

%% 测试 Enhanced_Learning
clear,clc;
input  = [rand(30, 20) + 4, rand(30, 20) + 8, rand(30, 20) + 12, rand(30, 20) + 16];
target = [ones(20, 1); ones(20, 1) * 2; ones(20, 1) * 3; ones(20, 1) * 4];
architecture = [30, 28, 25, 28, 30, 4];
is_enhancive = [1 1 1 1 1 0];
activations = {'ReLU', 'ReLU', 'ReLU', 'ReLU', 'softmax'};
options.is_sparse    = 1;
options.sparse_rho   = 0.1;
options.sparse_beta  = 0.03;
options.is_denoising = 1;
options.noise_rate   = 0.15;
options.noise_mode   = 'on_off';
% options.is_weighted_cost = 1;
% options.weighted_cost    = rand(sum(architecture(1:(length(architecture) - 1) / 2)), 1) * 2; % [0,2]之间的权重

Enhancive_Learning_SAE.description();
enhancive_SAE = Enhancive_Learning_SAE(architecture, activations, options, is_enhancive);
enhancive_SAE.train(input, target, 500, 1000);
enhancive_SAE.test(input, target)
predict_input = enhancive_SAE.predict(input);











%% 读取 image 及 label
% [ images4Train, labels4Train ] = loadMNISTData( 'dataSet/train-images.idx3-ubyte',...
%     'dataSet/train-labels.idx1-ubyte', 'MinMaxScaler', 0 );
% images4Train = images4Train( :, 1:6000 );
% labels4Train = labels4Train( 1:6000, : );
% [ images4Test, labels4Test ] = loadMNISTData( 'dataSet/t10k-images.idx3-ubyte',...
%     'dataSet/t10k-labels.idx1-ubyte', 'MinMaxScaler', 0 );
% 
% bpnn = Backprop_Neural_Network([784, 400, 200, 10], {'ReLU', 'ReLU', 'softmax'});
% bpnn.train(images4Train, labels4Train);
% bpnn.test(images4Test, labels4Test)


input  = [rand(30, 20) + 4, rand(30, 20) + 8, rand(30, 20) + 12, rand(30, 20) + 16];
target = [ones(20, 1); ones(20, 1) * 2; ones(20, 1) * 3; ones(20, 1) * 4];

bpnn = BackPropagation([30, 20, 4]);
bpnn.train(input, target);
bpnn.test(input, target);
bpnn.predict(input)





