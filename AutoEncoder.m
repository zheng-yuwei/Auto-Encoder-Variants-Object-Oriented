classdef AutoEncoder < Feedforward_Neural_Network
    % 一个全连接前向AE网络
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
    end
    properties(Hidden, Constant)
        %层数
        layers           = 3;
        %可选激活函数列表
        activations_list = char('Sigmoid', 'tanh',...
            'ReLU', 'leaky_ReLU', 'parameter_ReLU');
    end
    
    methods
        %实现网络的基本功能：初始化、训练、预测、测试、得到中间层、展示
        
        function obj = AutoEncoder(architecture, activations, options, theta)
            %构造函数
            % archietecture
            if isa(architecture, 'double') && length(architecture) == obj.layers
                obj.architecture = architecture;
            else
                error('AE网络结构必须是一个数组列表，并且结构为3层!');
            end
            % activations
            if exist('activations', 'var')
                obj.initialize_activations(activations);
            else
                obj.initialize_activations();
            end
            % options
            if exist('options', 'var')
                obj.initialize_options(options);
            else
                obj.initialize_options();
            end
            % parameters_num & theta
            obj.parameters_num = sum((obj.architecture(1:end-1) + 1) .* obj.architecture(2:end));
            
            if exist('theta', 'var')
                obj.initialize_parameters(theta);
            else
                obj.initialize_parameters();
            end
        end
        function train(obj, input, maxIter, theta)
            %后向训练BP网络
            disp(sprintf('\n 训练AE！'));
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
            
            % 判断该 countAE层 AE是否需要添加noise 以 使用denoising规则
            [ is_denoising, corrupted_input ] = obj.denoising_switch(input);
            if is_denoising
                [obj.theta, ~] = minFunc(@(x) obj.calc_cost_grad(input, x, corrupted_input), ...
                    obj.theta, option);
            else
                [obj.theta, ~] = minFunc(@(x) obj.calc_cost_grad(input, x), ...
                    obj.theta, option);
            end
        end
        function target   = predict(obj, input)
            %前向计算输出
            target = input;
            for layer_num = 1:(obj.layers - 1)
                [~, target] = obj.predict_next_layer(target, layer_num);
            end
        end
        function accuracy = test(obj, input)
            %测试网络预测的准确率
            result = obj.predict(input);
            accuracy = sum(sum((input - result).^2)) / (2 * size(input,2));
        end
        function code     = encode(obj, input)
            %得到AE网络的中间层表示
            [~, code] = obj.predict_next_layer(input, 1);
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
            active_func = str2func(obj.activations{layer_num}); % 激活函数
            
            start_index = (obj.architecture + 1) .* [obj.architecture(2:end) 0];
            start_index = cumsum(start_index([end 1:end-1])) + 1;
            
            start_index = start_index(layer_num);
            end_index   = start_index + next_layer_size * this_layer_size - 1;
            
            % 得到 系数w 和 b（softmax则没有）,并计算 诱导局部域 及 输出
            w = reshape(obj.theta(start_index : end_index), next_layer_size, this_layer_size);
            
            start_index = end_index + 1;
            end_index   = end_index + next_layer_size;
            b = obj.theta(start_index : end_index);
            
            hidden_V = bsxfun(@plus, w * input, b);
            hidden_X = active_func(hidden_V);
        end
        function [cost, grad] = calc_cost_grad(obj, input, theta, corrupted_input)
            %计算网络误差、梯度
            addpath('.\activation_function');
            
            samples_num = size(input, 2); % 样本数
            visibleSize = obj.architecture(1);
            hiddenSize  = obj.architecture(2);
            
            W1 = reshape(theta(1:(hiddenSize * visibleSize)), ...
                hiddenSize, visibleSize);
            b1 = theta((hiddenSize * visibleSize + 1):(hiddenSize * visibleSize + hiddenSize));
            W2 = reshape(theta((hiddenSize * visibleSize + hiddenSize + 1):(2 * hiddenSize * visibleSize + hiddenSize)), ...
                visibleSize, hiddenSize);
            b2 = theta((2 * hiddenSize * visibleSize + hiddenSize + 1) : end);
            
            cost = 0;
            % feed-forward阶段
            activation_func = str2func(obj.activations{1}); % 将 激活函数名 转为 激活函数
            if exist('corrupted_input', 'var')
                hidden_V = bsxfun(@plus, W1 * corrupted_input, b1);
            else
                hidden_V = bsxfun(@plus, W1 * input, b1);
            end
            hidden_X = activation_func(hidden_V);
            % sparse
            if obj.options.is_sparse
                rho_hat = sum(hidden_X, 2) / samples_num;
                KL = getKL( obj.options.sparse_rho, rho_hat);
                cost = cost + obj.options.sparse_beta * sum(KL);
            end
            
            activation_func = str2func(obj.activations{2});
            output_V = bsxfun(@plus, W2 * hidden_X, b2);
            output_X = activation_func(output_V);
            
            if obj.options.is_weighted_cost
                cost = cost + sum(obj.options.weighted_cost' * (output_X - input).^2) / samples_num / 2;
            else
                cost = cost + sum(sum((output_X - input) .^ 2)) / samples_num / 2;
            end
            cost = cost + obj.options.decay_lambda * (sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2))) / 2;
            % Back Propagation 阶段：链式法则求导
            activation_func_deriv = str2func([obj.activations{1}, '_derivative'] );
            % 链式法则求导
            % dError/dOutputV = dError/dOutputX * dOutputX/dOutputV
            if obj.options.is_weighted_cost
                dError_dOutputV   = bsxfun(@times, -(input - output_X), obj.options.weighted_cost) .* ...
                    activation_func_deriv(output_V);
            else
                dError_dOutputV   = -(input - output_X) .* activation_func_deriv(output_V);
            end
            
            % dError/dW2 = dError/dOutputV * dOutputV/dW2
            dError_dW2   = dError_dOutputV * hidden_X';
            
            W2Grad       = dError_dW2 ./ samples_num + obj.options.decay_lambda * W2;
            % dError/dHiddenV = ( dError/dHiddenX + dSparse/dHiddenX ) * dHiddenX/dHiddenV
            dError_dHiddenX   = W2' * dError_dOutputV; % = dError/dOutputV * dOutputV/dHiddenX
            dHiddenX_dHiddenV = activation_func_deriv(hidden_V);
            if obj.options.is_sparse
                dSparse_dHiddenX = obj.options.sparse_beta .* getKL_deriv( obj.options.sparse_rho, rho_hat );
                dError_dHiddenV  = (dError_dHiddenX + repmat(dSparse_dHiddenX, 1, samples_num)) .* dHiddenX_dHiddenV;
            else
                dError_dHiddenV  = dError_dHiddenX .* dHiddenX_dHiddenV;
            end
            % dError/dW1 = dError/dHiddenV * dHiddenV/dW1
            dHiddenV_dW1 = input';
            dError_dW1   = dError_dHiddenV * dHiddenV_dW1;
            W1Grad       = dError_dW1 ./ samples_num + obj.options.decay_lambda * W1;
            
            % 求偏置的导数
            dError_db2 = sum(dError_dOutputV, 2);
            b2Grad     = dError_db2 ./ samples_num;
            dError_db1 = sum(dError_dHiddenV, 2);
            b1Grad     = dError_db1 ./ samples_num;
            
            grad = [ W1Grad(:); b1Grad(:); W2Grad(:); b2Grad(:) ];
        end
        function [is_denoising, corrupted_input ] = denoising_switch(obj, input)
            %判断该层AE是否需要添加noise以使用denoising规则
            % 返回 是否is_denoising的标志 及 加噪后的信号
            
            % is_denoising：	是否使用 denoising 规则
            % noise_rate：	每一位添加噪声的概率
            % noise_mode：	添加噪声的模式：'on_off' or 'Guass'
            % noise_mean：	高斯模式：均值
            % noise_sigma：	高斯模式：标准差
            
            is_denoising    = 0;
            corrupted_input = [];
            if obj.options.is_denoising
                is_denoising = 1;
                corrupted_input = input;
                indexCorrupted = rand(size(input)) < obj.options.noise_rate;
                switch obj.options.noise_mode
                    case 'Guass'
                        % 均值为 noiseMean，标准差为 noiseSigma 的高斯噪声
                        noise = obj.options.noise_mean + ...
                            randn(size(input)) * obj.options.noise_sigma;
                        noise(~indexCorrupted) = 0;
                        corrupted_input = corrupted_input + noise;
                    case 'OnOff'
                        corrupted_input(indexCorrupted) = 0;
                end
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
                for i = 1:(obj.layers - 1)
                    obj.activations{i} = 'Sigmoid';
                end
            end
        end
        function initialize_options(obj, options)
            %初始化AE网络选项 options
            % decay_lambda：     权重衰减系数——正则项罚项权重
            
            % is_sparse：        是否使用 sparse hidden level 的规则；
            % sparse_rho：       稀疏性中rho；
            % sparse_beta：      稀疏性罚项权重；
            
            % is_denoising：     是否使用 denoising 规则
            % noise_rate：       每一位添加噪声的概率
            % noise_mode：       添加噪声的模式：'on_off' or 'Guass'
            % noise_mean：       高斯模式：均值
            % noise_sigma：      高斯模式：标准差
            
            % is_weighted_cost： 是否对每一位数据的cost进行加权对待
            % weighted_cost：    加权cost的权重
            
            if ~exist('options', 'var')
                options = [];
            end
            % decay
            if isfield(options, 'decay_lambda')
                obj.options.decay_lambda = options.decay_lambda;
            else
                obj.options.decay_lambda = 0.01;
            end
            % sparse
            if isfield(options, 'is_sparse')
                obj.options.is_sparse = options.is_sparse;
            else
                obj.options.is_sparse = 0;
            end
            if obj.options.is_sparse
                if isfield(options, 'sparse_rho')
                    obj.options.sparse_rho = options.sparse_rho;
                else
                    obj.options.sparse_rho = 0.01;
                end
                if isfield(options, 'sparse_beta')
                    obj.options.sparse_beta = options.sparse_beta;
                else
                    obj.options.sparse_beta = 0.3;
                end
            end
            
            % de-noising
            if isfield(options, 'is_denoising')
                obj.options.is_denoising = options.is_denoising;
                if options.is_denoising
                    % 噪声模式：高斯 或 开关
                    if isfield(options, 'noise_mode')
                        obj.options.noise_mode = options.noise_mode;
                    else
                        obj.options.noise_mode = 'on_off';
                    end
                    switch options.noise_mode
                        case 'Guass'
                            if isfield(options, 'noise_mean')
                                obj.options.noise_mean = options.noise_mean;
                            else
                                obj.options.noise_mean = 0;
                            end
                            if isfield(options, 'noise_sigma')
                                obj.options.noise_sigma = options.noise_sigma;
                            else
                                obj.options.noise_sigma = 0.01;
                            end
                        case 'on_off'
                            % 噪声概率
                            if isfield(options, 'noise_rate')
                                obj.options.noise_rate = options.noise_rate;
                            else
                                obj.options.noise_rate = 0.15;
                            end
                    end
                end
            else
                obj.options.is_denoising = 0;
            end
            
            % weightedCost
            if isfield(options, 'is_weighted_cost')
                obj.options.is_weighted_cost = options.is_weighted_cost;
            else
                obj.options.is_weighted_cost = 0;
            end
            if obj.options.is_weighted_cost
                if isfield(options, 'weighted_cost')
                    obj.options.weighted_cost = options.weighted_cost;
                else
                    error('加权cost一定要自己设置权重向量！');
                end
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
                    
                    r = sqrt(6 / (obj.architecture(layer + 1) + obj.architecture(layer)));
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
            disp_info = [sprintf('\n这是一个全连接自编码器 Auto-Encoder！\n'), ...
                sprintf('工作机制为：利用back-propagation实现输出拟合输入的3层 fully-connected feedforward neural networks，\n'),...
                sprintf('\t\t   从而实现 encode & decode 过程。\n\n'),...
                sprintf('-必须初始化的参数为：网络框架 architecture；\n'),...
                sprintf('-可选初始化的参数为：激活函数列表 activations，网络选项 options，网络参数 theta；\n'),...
                sprintf('\t 可选的激活函数activations有：Sigmoid, tanh, ReLU, leaky_ReLU, parameter_ReLU。\n'),...
                sprintf('\t 可选的 网络选项options 有：\n'),...
                sprintf('\t\t decay_lambda：     权重衰减系数——正则项罚项权重，默认为0.01；\n'),...
                sprintf('\t\t is_sparse：        是否使用 sparse hidden level 的规则，默认不使用；\n'),...
                sprintf('\t\t\t sparse_rho：   稀疏性中rho，默认为0.01；\n'),...
                sprintf('\t\t\t sparse_beta：  稀疏性罚项权重，默认为0.3；\n'),...
                sprintf('\t\t is_denoising：     是否使用 denoising 规则，默认不使用;\n'),...
                sprintf('\t\t\t noise_rate：   每一位添加噪声的概率，默认为0.15;\n'),...
                sprintf('\t\t\t noise_mode：   添加噪声的模式："on_off" or "Guass"，默认为on_off;\n'),...
                sprintf('\t\t\t noise_mean：   高斯模式：均值，默认为0;\n'),...
                sprintf('\t\t\t noise_sigma：  高斯模式：标准差，默认为0.01;\n'),...
                sprintf('\t\t is_weighted_cost： 是否对每一位数据的cost进行加权对待，默认不使用;\n'),...
                sprintf('\t\t\t weighted_cost：加权cost的权重。\n'),...
                sprintf('\t 默认初始化 网络参数 theta 使用：Hugo Larochelle 建议，[-sqrt(6/h1/h2),sqrt(6/h1/h2)]。\n'),...
                sprintf('\n')];
            disp(disp_info);
        end
    end
end






