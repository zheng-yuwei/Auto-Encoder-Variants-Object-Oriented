classdef BackPropagation < Feedforward_Neural_Network
    % 一个全连接前向BP网络
    % by 郑煜伟 Aewil 2016-05
    
    % 网络结构、网络参数只有初始化时决定，便再也不可改；网络选项可改
    properties(SetAccess = private, GetAccess = public)
        %网络结构
        architecture
        %每一层激活函数类型
        activations
        %网络参数
        theta
    end
    properties(SetAccess = public, GetAccess = public)
        %网络选项：因为要修改weighted_cost
        options
    end
    properties(Hidden, SetAccess = private, GetAccess = public)
        %参数个数
        parameters_num = 0;
        %层数
        layers         = 0;
    end
    properties(Hidden, Constant)
        %可选激活函数列表
        activations_list = char('Sigmoid', 'tanh',...
            'ReLU', 'leaky_ReLU', 'parameter_ReLU', 'softmax');
    end
    
    methods
        %实现网络的基本功能：初始化、训练、预测、测试、展示
        
        function obj = BackPropagation(architecture, activations, options, theta)
            %构造函数，输入网络结构、激活函数胞元数组、网络预选项、网络参数
            if isa(architecture, 'double')
                obj.architecture = architecture;
            else
                error('BP网络结构必须是一个数组列表!');
            end
            obj.layers = length(obj.architecture);
            
            if exist('activations', 'var')
                obj.initialize_activations(activations);
            else
                obj.initialize_activations();
            end
            
            if exist('options', 'var')
                obj.initialize_options(options);
            else
                obj.initialize_options();
            end
            
            obj.parameters_num = sum((obj.architecture(1:end-1) + 1) .* obj.architecture(2:end));
            if strcmp(obj.activations{end}, 'softmax')
                obj.parameters_num = obj.parameters_num - obj.architecture(end);    
            end
            
            if exist('theta', 'var')
                obj.initialize_parameters(theta);
            else
                obj.initialize_parameters();
            end
        end
        function target = predict(obj, input)
            %前向计算输出
            for layer_num = 1:(obj.layers - 1)
                [~, input] = obj.predict_next_layer(input, layer_num);
            end
            target = input;
        end
        function train(obj, input, target, maxIter, theta)
            %后向训练BP网络
            disp(sprintf('\n 训练BP！'));
            % 函数 calc_cost_grad 可以根据当前点计算 cost 和 gradient，但是步长不确定
            % 这里，调用Mark Schmidt的包来优化迭代 步长：用了l-BFGS
            % Mark Schmidt (http://www.di.ens.fr/~mschmidt/Software/minFunc.html) [仅供学术]
            addpath minFunc/
            option.Method = 'lbfgs';
            if exist('maxIter', 'var')
                option.maxIter = maxIter; % L-BFGS 的最大迭代代数
            else
                option.maxIter = 100;
            end
            option.display = 'on';
            % option.TolX = 1e-3;

            if exist('theta', 'var')
                obj.initialize_parameters(theta);
            end
            
            [obj.theta, ~] = minFunc(@(x) obj.calc_cost_grad(input, target, x), ...
                obj.theta, option);  
        end
        function accuracy = test(obj, input, target)
            %测试网络预测的准确率
            result = obj.predict(input);
            if strcmp(obj.activations{end},'softmax') % 标签类精度
                % 将预测的概率矩阵中，每列最大概率的值置1，其他置0
                result = bsxfun(@eq, result, max(result));
                
                indexRow = target';
                indexCol = 1:length(indexRow);
                index    = (indexCol - 1) .* obj.architecture(end) + indexRow;
                
                accuracy = sum(result(index))/length(indexRow);
            else % 实值类精度
                accuracy = sum(sum((target - result).^2)) / (2 * size(target,2));
            end
        end
        function disp(obj)
            %对网络对象的描述
            obj.description();
            
            nn_info = sprintf('-----------------------------------------------\n');
            nn_info = [nn_info, ...
                sprintf('%s !\n', ...
                ['该网络具有 ' num2str(obj.layers) ' 层：' num2str(obj.architecture)])];
            
            nn_activations = '';
            for i = 1:length(obj.activations)
                nn_activations = [nn_activations '  ' obj.activations{i}];
            end
            nn_info = [nn_info, ...
                sprintf('每层激活函数分别为：%s ~\n', nn_activations)];
            nn_info = [nn_info, ...
                sprintf('该网络的权重衰减权重为：%d ~\n', obj.options.decay_lambda)];
            nn_info = [nn_info, sprintf('-----------------------------------------------\n')];
            disp(nn_info);
        end
    end
    methods(Access = private)
        %用于网络前向计算和后向计算
        
        function [hidden_V, hidden_X]  = predict_next_layer(obj, input, layer_num)
            %计算网络隐藏层layer_num的下一层的 诱导局部域hiddenV 和 输出hiddenX
            addpath('.\activation_function');
            this_layer_size = obj.architecture(layer_num);
            next_layer_size = obj.architecture(layer_num + 1);
            start_index = (obj.architecture + 1) .* [obj.architecture(2:end) 0];
            start_index = cumsum(start_index([end 1:end-1])) + 1;
            
            start_index = start_index(layer_num);
            end_index   = start_index + next_layer_size * this_layer_size - 1;
            
            active_func = str2func(obj.activations{layer_num}); % 激活函数
            % 得到 系数w 和 b（softmax则没有）,并计算 诱导局部域 及 输出
            w = reshape(obj.theta(start_index : end_index), next_layer_size, this_layer_size);
            if strcmp(obj.activations{layer_num}, 'softmax')
                hidden_V = w * input;
            else
                start_index = end_index + 1;
                end_index   = end_index + next_layer_size;
                b = obj.theta(start_index : end_index);
                hidden_V = bsxfun(@plus, w * input, b);
            end
            hidden_X = active_func(hidden_V);
        end
        function [cost, grad] = calc_cost_grad(obj, input, target, theta)
            %计算网络误差、梯度
            addpath('.\activation_function');
            
            samples_num = size(input, 2); % 样本数
            grad        = zeros(size(theta));
            % 初始化一些参数：诱导局部域数据、输出/输入数据
            hidden_V    = cell(1, obj.layers - 1);
            hidden_X    = cell(1, obj.layers);
            hidden_X{1} = input;
            cost        = 0;
            % feed-forward阶段
            startIndex = 1; % 存储变量的下标起点
            for i = 1:(obj.layers - 1)
                visibleSize = obj.architecture(i);
                hiddenSize  = obj.architecture(i + 1);
                
                activation_func = str2func(obj.activations{i}); % 将 激活函数名 转为 激活函数
                
                % 先将 theta 转换为 (W, b) 的矩阵/向量 形式，以便后续处理（与initializeParameters文件相对应）
                endIndex   = hiddenSize * visibleSize + startIndex - 1; % 存储变量的下标终点
                W          = reshape(theta(startIndex : endIndex), hiddenSize, visibleSize);
                
                if strcmp(obj.activations{i}, 'softmax') % softmax那一层不用偏置b
                    startIndex = endIndex + 1; % 存储变量的下标起点
                    
                    hidden_V{i} = W * hidden_X{i};% 求和 -> 得到诱导局部域 V
                else
                    startIndex = endIndex + 1; % 存储变量的下标起点
                    endIndex   = hiddenSize + startIndex - 1; % 存储变量的下标终点
                    b          = theta( startIndex : endIndex );
                    startIndex = endIndex + 1;
                    
                    hidden_V{i} = bsxfun(@plus, W * hidden_X{i}, b); % 求和 -> 得到诱导局部域 V
                end
                hidden_X{i + 1} = activation_func(hidden_V{i}); % 激活函数
                % 计算正则项的罚函数
                cost = cost + 0.5 * obj.options.decay_lambda * sum(sum(W .^ 2));
            end
            
            % 求cost function + regularization
            if strcmp(obj.activations{end}, 'softmax') % 标签类cost
                % softmax的cost，但我并没有求对数，并且加了1. 用于模仿准确率
                indexRow = target';
                indexCol = 1:samples_num;
                index    = (indexCol - 1) .* obj.architecture(end) + indexRow;
                cost = cost - sum(log(hidden_X{end}(index))) / samples_num;
            else % 实值类cost
                cost = cost + sum( sum((target - hidden_X{end}).^2) ) ./ 2 / samples_num;
            end
            
            % Back Propagation 阶段：链式法则求导
            % 求最后一层
            activation_func_deriv = str2func([obj.activations{end}, '_derivative']);
            if strcmp(obj.activations{end}, 'softmax' ) % softmax那一层求导需要额外labels信息
                dError_dHiddenV = activation_func_deriv(hidden_V{end}, target);
            else
                % dError/dOutputV = dError/dOutputX * dOutputX/dOutputV
                dError_dHiddenV = -( target - hidden_X{end} ) .* ...
                    activation_func_deriv( hidden_V{end} );
            end
            % dError/dW = dError/dOutputV * dOutputV/dW
            dError_dW   = dError_dHiddenV * hidden_X{obj.layers - 1}';
            
            end_index   = obj.parameters_num; % 存储变量的下标终点
            if strcmp( obj.activations{end}, 'softmax' ) % softmax那一层不用偏置b
                start_index = end_index + 1; % 存储变量的下标起点
            else
                % 更新梯度 b
                start_index = end_index - obj.architecture(end)  + 1; % 存储变量的下标起点
                dError_db   = sum(dError_dHiddenV, 2);
                grad(start_index:end_index) = dError_db ./ samples_num;
            end
            % 更新梯度 W
            end_index   = start_index - 1; % 存储变量的下标终点
            start_index = end_index - obj.architecture(end - 1) * obj.architecture(end)  + 1; % 存储变量的下标起点
            W           = reshape(theta(start_index:end_index), ...
                obj.architecture(end), obj.architecture(end - 1));
            WGrad       = dError_dW ./ samples_num + obj.options.decay_lambda * W;
            grad( start_index:end_index ) = WGrad(:);
            
            % 误差回传 error back-propagation
            for i = (obj.layers - 2):-1:1
                activation_func_deriv = str2func([obj.activations{i}, '_derivative']);
                % dError/dHiddenV = dError/dHiddenX * dHiddenX/dHiddenV
                % dError/dHiddenX = dError/dOutputV * dOutputV/dHiddenX
                dError_dHiddenV = W' * dError_dHiddenV .* activation_func_deriv(hidden_V{i});
                % dError/dW1 = dError/dHiddenV * dHiddenV/dW1
                dError_dW = dError_dHiddenV * hidden_X{i}';
                
                dError_db = sum(dError_dHiddenV, 2);
                % 更新梯度 b
                end_index   = start_index - 1; % 存储变量的下标终点
                start_index = end_index - obj.architecture(i + 1)  + 1; % 存储变量的下标起点
                % b           = theta(start_index : end_index);
                grad(start_index:end_index) = dError_db ./ samples_num;
                
                % 更新梯度 W
                end_index   = start_index - 1; % 存储变量的下标终点
                start_index = end_index - ...
                    obj.architecture(i) * obj.architecture(i + 1)  + 1; % 存储变量的下标起点
                W           = reshape(theta(start_index:end_index), ...
                    obj.architecture(i + 1), obj.architecture(i) );
                WGrad       = dError_dW ./ samples_num + obj.options.decay_lambda * W;
                grad(start_index:end_index) = WGrad(:);
            end
        end
    end
    methods(Hidden, Access = private)
        %用于初始化
        
        function initialize_activations(obj, activations)
            %初始化网络的激活函数类型列表
            if exist('activations', 'var')
                if ~isa(activations, 'cell')
                    error('激活函数列表 必须是胞元数组！');
                elseif length(activations) ~= obj.layers - 1
                    error('激活函数列表 和 网络层数 不一致！');
                else
                    for i = 1:length(activations)
                        if isempty(activations{i})
                            activations{i} = 'Sigmoid';
                        else
                            flag = 0;
                            for j = 1:size(obj.activations_list, 1)
                                if strcmp(strtrim(obj.activations_list(j, :)), activations{i})
                                    flag = 1;
                                    break;
                                end
                            end
                            if flag == 0
                                error(['激活函数设置错误： ' activations{i} ' 不存在！']);
                            end
                        end
                    end
                end
                obj.activations = activations;
            else
                obj.activations = cell(obj.layers - 1, 1);
                for i = 1:(obj.layers - 2)
                    obj.activations{i} = 'Sigmoid';
                end
                obj.activations{length(obj.activations)} = 'softmax';
            end
        end
        function initialize_options(obj, options)
            %初始化BP网络选项 options
            % decay_lambda：  权重衰减系数——正则项罚项权重;
            if ~exist('options', 'var')
                options = [];
            end
            
            if isfield( options, 'decay_lambda' )
                obj.options.decay_lambda = options.decay_lambda;
            else
                obj.options.decay_lambda = 0.01;
            end
        end
        function initialize_parameters(obj, theta)
            %初始化网络参数
            if exist('theta', 'var')
                if length(theta) == obj.parameters_num
                    obj.theta = theta;
                else
                    error(['传入的theta参数维度错误：应该为 ' ...
                        num2str(obj.parameters_num) ' 维！']);
                end
            else
                % 根据 Hugo Larochelle 建议
                obj.theta = zeros(obj.parameters_num, 1);
                
                start_index = 1; % 设置每层网络w的下标起点
                for layer = 1:(obj.layers - 1) % layer  -> layer + 1
                    % 设置每层网络W的下标终点
                    end_index = start_index + ...
                        obj.architecture(layer + 1) * obj.architecture(layer) - 1;
                    
                    r = sqrt( 6 ) / sqrt( obj.architecture(layer + 1) + obj.architecture(layer) );
                    obj.theta(start_index:end_index, 1) = ...
                        rand( obj.architecture(layer + 1) * obj.architecture(layer), 1 ) * 2 * r - r;
                    
                    % 设置下一层网络W的下标起点（跳过b）
                    start_index = end_index + obj.architecture(layer + 1) + 1;
                end
            end
        end
    end
    methods(Static)
        function description()
            %对该网络类型的描述
            disp_info = [sprintf('\n这是一个全连接BP神经网络！\n'), ...
                sprintf('工作机制为：前向计算网络输出，误差反向回传调整参数。\n'),...
                sprintf('-必须初始化的参数为：网络框架 architecture；\n'),...
                sprintf('-可选初始化的参数为：激活函数列表 activations，网络选项 options，网络参数 theta；\n'),...
                sprintf('\t 可选的激活函数activations有：Sigmoid, tanh, ReLU, leaky_ReLU, parameter_ReLU, softmax。\n'),...
                sprintf('\t 可选的 网络选项options 有：\n'),...
                sprintf('\t\t decay_lambda：     权重衰减系数——正则项罚项权重，默认为0.01。\n'), ...
                sprintf('\t 默认初始化 网络参数 theta 使用：Hugo Larochelle 建议，[-sqrt(6/h1/h2),sqrt(6/h1/h2)]。\n'),...
                sprintf('\n')];
            disp(disp_info);
        end
    end
end




