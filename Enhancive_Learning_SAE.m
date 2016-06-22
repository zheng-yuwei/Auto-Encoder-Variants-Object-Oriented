classdef Enhancive_Learning_SAE < Feedforward_Neural_Network
    % 一个全连接前向enhanced-learning网络
    % by 郑煜伟 Aewil 2016-05
    
    % 网络结构、网络参数只有初始化时决定，便再也不可改；网络选项可改
    properties(SetAccess = private, GetAccess = public)
        %网络结构
        architecture
        %网络增强的层级
        is_enhancive
        %每一层激活函数类型
        activations
        %SAE网络参数
        theta
        %每个AE网络 + 最后一层BP网络的参数
        theta4AEs
    end
    properties(SetAccess = public, GetAccess = public)
        %网络选项：因为要修改weighted_cost
        options
    end
    properties(Hidden, SetAccess = private, GetAccess = public)
        %参数个数
        parameters_num     = 0;
        parameters_num4AEs = 0;
        %层数
        layers             = 0;
        %记录增强的层数
        enhancive_layers   = [];
    end
    properties(Hidden, Constant)
        %可选激活函数列表
        activations_list = char('Sigmoid', 'tanh',...
            'ReLU', 'leaky_ReLU', 'parameter_ReLU', 'softmax');
    end
    
    methods
        %实现网络的基本功能：初始化、训练、预测、测试、展示
        
        function obj = Enhancive_Learning_SAE(architecture, activations, options, is_enhancive, theta4AEs)
            %构造函数
            % architecture & layers
            if isa(architecture, 'double')
                obj.architecture = architecture;
            else
                error('网络结构必须是一个数组列表!');
            end
            obj.layers = length(architecture);
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
            % parameters_num & parameters_num4AEs
            obj.parameters_num = sum((obj.architecture(1:end-1) + 1) .* obj.architecture(2:end));
            if strcmp(obj.activations{end}, 'softmax')
                obj.parameters_num = obj.parameters_num - obj.architecture(end);    
            end
            obj.parameters_num4AEs = sum((obj.architecture(1:end-1) + 1) .* obj.architecture(2:end)) + ...
                sum(obj.architecture(1:end-2) .* (obj.architecture(2:end-1) + 1));
            if strcmp(obj.activations{end}, 'softmax')
                obj.parameters_num4AEs = obj.parameters_num4AEs - obj.architecture(end);    
            end
            % is_enhancive
            if ~exist('is_enhancive', 'var')
                obj.initialize_enhancive_layers();
            else
                obj.initialize_enhancive_layers(is_enhancive);
            end
            % theta4AEs & theta
            if exist('theta4AEs', 'var')
                obj.initialize_parameters(theta4AEs);
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
        function train(obj, input, target, maxIter4AE, maxIter4BP, theta4AEs)
            %后向训练SAE网络
            if ~exist('maxIter4AE', 'var')
                maxIter4AE = 100;
            end
            if ~exist('maxIter4BP', 'var')
                maxIter4BP = 200;
            end
            if exist('theta4AEs', 'var')
                obj.initialize_parameters(theta4AEs);
            end
            obj.pre_training(input, target, maxIter4AE, maxIter4BP);
            obj.fine_tune(input, target, maxIter4BP)
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
        function pre_training(obj, input, target, maxIter4AE, maxIter4BP)
            %对enhancive learning SAE进行预训练：训练 多个AE/GAE 及 一个BP
            
            % layer-wise 训练AE
            input4AE        = input;
            option4AE       = obj.options; % AE/GAE网络的选项
            weight_start    = 1; % weighted-cost（若有）的下标起点
            weight_end      = 0; % weighted-cost（若有）的下标终点
            
            start_index4AEs = 1;
            end_index4AEs   = 0;
            
            layer           = 1;
            count_enhancive = 1;
            while layer <= obj.layers-2
                % 判断是AE还是GAE
                if obj.is_enhancive(layer) == 0 % AE
                    % 设置每个AE的weight_end（若需要）
                    if obj.options.is_weighted_cost
                        weight_end = weight_end + obj.architecture(layer);
                        option4AE.weighted_cost = ...
                            obj.options.weighted_cost(weight_start:weight_end);
                        weight_start = weight_end + 1;
                    end
                    % 判断是否de-noising
                    if option4AE.is_denoising && ...
                            strcmp(option4AE.noising_layer, 'first_layer') && ...
                            layer ~= 1
                        option4AE.is_denoising = 0;
                    end
                    
                    % 重置参数下标终点
                    end_index4AEs = end_index4AEs + ...
                        2 * obj.architecture(layer) * obj.architecture(layer + 1) + ...
                        obj.architecture(layer) + obj.architecture(layer + 1);
                    % 设置 AE网络的结构 和 激活函数
                    architecture4AE = [obj.architecture(layer), obj.architecture(layer + 1), ...
                        obj.architecture(layer)];
                    activations4AE = {obj.activations{layer}, obj.activations{layer}};
                    
                    % 创建AE，并进行训练
                    if sum(obj.theta4AEs(start_index4AEs:end_index4AEs)) ~= 0
                        Autoencoder = AutoEncoder(architecture4AE, activations4AE, ...
                            option4AE, obj.theta4AEs(start_index4AEs:end_index4AEs));
                    else
                        Autoencoder = AutoEncoder(architecture4AE, activations4AE, option4AE);
                    end
                    Autoencoder.train(input4AE, maxIter4AE);
                    input4AE = Autoencoder.encode(input4AE);
                    % 获得AE的参数
                    obj.theta4AEs(start_index4AEs:end_index4AEs) = Autoencoder.theta;
                    % 重置参数下标其实点
                    start_index4AEs = end_index4AEs + 1;
                    % 更新layer
                    layer = layer + 1;
                else % GAE
                    % 设置每个AE的weight_end（若需要）
                    if obj.options.is_weighted_cost
                        weight_end = weight_end + ...
                            sum(obj.architecture(layer:...
                        (layer + (obj.enhancive_layers(count_enhancive) - 1) / 2 - 1)));
                        option4AE.weighted_cost = ...
                            obj.options.weighted_cost(weight_start:weight_end);
                        weight_start = weight_end + 1;
                    end
                    % 判断是否de-noising
                    if option4AE.is_denoising && ...
                            strcmp(option4AE.noising_layer, 'first_layer') && ...
                            layer ~= 1
                        option4AE.is_denoising = 0;
                    end
                    
                    % 设置GAE参数
                    architecture4GAE = obj.architecture(layer:...
                        (layer + obj.enhancive_layers(count_enhancive) - 1));
                    activations4GAE  = obj.activations(layer:...
                        (layer + obj.enhancive_layers(count_enhancive) - 2));
                    % 重置参数下标终点
                    end_index4AEs = end_index4AEs + ...
                        2 * sum(obj.architecture(layer:...
                        (layer + (obj.enhancive_layers(count_enhancive) - 1) / 2 - 1)) .* ...
                        obj.architecture((layer + 1):...
                        (layer + (obj.enhancive_layers(count_enhancive) - 1) / 2))) + ...
                        sum(obj.architecture((layer + 1):...
                        (layer + obj.enhancive_layers(count_enhancive) - 1)));
                    if sum(obj.theta4AEs(start_index4AEs:end_index4AEs)) ~= 0
                        Generative_Autoencoder = ...
                            Generative_AutoEncoder(architecture4GAE, activations4GAE, option4AE, ...
                            obj.theta4AEs(start_index4AEs:end_index4AEs));
                    else
                        Generative_Autoencoder = ...
                            Generative_AutoEncoder(architecture4GAE, activations4GAE, option4AE);
                    end
                    Generative_Autoencoder.train(input, maxIter4AE, maxIter4BP);
                    
                    % 获得AE的参数
                    obj.theta4AEs(start_index4AEs:end_index4AEs) = Generative_Autoencoder.theta;
                    % 重置参数下标其实点
                    start_index4AEs = end_index4AEs + 1;
                    % 更新layer
                    layer = layer + obj.enhancive_layers(count_enhancive) - 1;
                end
            end
            % 最后一层BP的训练
            architecture4BP = [obj.architecture(end - 1), obj.architecture(end)];
            activations4BP  = obj.activations(end);
            if sum(obj.theta4AEs(start_index4AEs:end)) ~= 0
                Backpropagation = BackPropagation(architecture4BP, activations4BP, ...
                    obj.options, obj.theta4AEs(start_index4AEs:end));
            else
                Backpropagation = BackPropagation(architecture4BP, activations4BP, obj.options);
            end
            Backpropagation.train(input4AE, target, maxIter4BP);
            obj.theta4AEs(start_index4AEs:end) = Backpropagation.theta;
            
            %将 AEs 的部分参数赋值给 SAE
            obj.theta4AEs_to_theta();
        end
        function fine_tune(obj, input, target, maxIter4BP)
            %对SAE进行最后的微调
            Backpropagation = BackPropagation(obj.architecture, obj.activations, ...
                    obj.options, obj.theta);
            Backpropagation.train(input, target, maxIter4BP);
            obj.theta = Backpropagation.theta;
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
            if isfield( options, 'decay_lambda' )
                obj.options.decay_lambda = options.decay_lambda;
            else
                obj.options.decay_lambda = 0.01;
            end
            % sparse
            if isfield( options, 'is_sparse' )
                obj.options.is_sparse = options.is_sparse;
            else
                obj.options.is_sparse = 0;
            end
            if obj.options.is_sparse
                if isfield( options, 'sparse_rho' )
                    obj.options.sparse_rho = options.sparse_rho;
                else
                    obj.options.sparse_rho = 0.01;
                end
                if isfield( options, 'sparse_beta' )
                    obj.options.sparse_beta = options.sparse_beta;
                else
                    obj.options.sparse_beta = 0.3;
                end
            end
            
            % de-noising
            if isfield( options, 'is_denoising' )
                obj.options.is_denoising = options.is_denoising;
                if options.is_denoising
                    % 加入噪声的网络层
                    if isfield(options, 'noising_layer')
                        obj.options.noising_layer = options.noising_layer;
                    else
                        obj.options.noising_layer = 'first_layer';
                    end
                    % 噪声模式：高斯 或 开关
                    if isfield( options, 'noise_mode' )
                        obj.options.noise_mode = options.noise_mode;
                    else
                        obj.options.noise_mode = 'on_off';
                    end
                    switch options.noise_mode
                        case 'Guass'
                            if isfield( options, 'noise_mean' )
                                obj.options.noise_mean = options.noise_mean;
                            else
                                obj.options.noise_mean = 0;
                            end
                            if isfield( options, 'noise_sigma' )
                                obj.options.noise_sigma = options.noise_sigma;
                            else
                                obj.options.noise_sigma = 0.01;
                            end
                        case 'on_off'
                            % 噪声概率
                            if isfield( options, 'noise_rate' )
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
            if isfield( options, 'is_weighted_cost' )
                obj.options.is_weighted_cost = options.is_weighted_cost;
            else
                obj.options.is_weighted_cost = 0;
            end
            if obj.options.is_weighted_cost
                if isfield( options, 'weighted_cost' )
                    obj.options.weighted_cost = options.weighted_cost;
                else
                    error( '加权cost一定要自己设置权重向量！' );
                end
            end
        end
        function initialize_enhancive_layers(obj, is_enhancive)
            %判断增强学习的网络层级结构是否合法，并初始化
            
            if ~exist('is_enhancive', 'var')
                warning(sprintf('\n\t并没有使用到enhanced learning （GAE），建议直接使用类 Stacked_AutoEncoder！\n'));
                obj.is_enhancive = zeros(obj.layers);
            else
                % 检测增强的网络层数和层级 及其 合法性
                % 记录 及 调整AEs参数数量 parameters_num4AEs
                start = 1;
                while start <= obj.layers
                    if is_enhancive(start) ~= 0
                        enhancive_start = start;
                        enhancive_end   = start;
                        while enhancive_end <= obj.layers
                            obj.parameters_num4AEs = obj.parameters_num4AEs - ...
                                obj.architecture(enhancive_end) * obj.architecture(enhancive_end + 1) -...
                                obj.architecture(enhancive_end + 1);
                            
                            enhancive_end = enhancive_end + 1;
                            if is_enhancive(enhancive_end) == 0
                                enhancive_end = enhancive_end - 1;
                                break;
                            end
                        end
                        
                        obj.parameters_num4AEs = obj.parameters_num4AEs + ...
                            obj.architecture(enhancive_end) * obj.architecture(enhancive_end + 1) +...
                            obj.architecture(enhancive_end + 1);
                        start = enhancive_end;
                        
                        obj.enhancive_layers = [obj.enhancive_layers ...
                            enhancive_end - enhancive_start + 1];
                    end
                    if mod(obj.enhancive_layers(end), 2) == 0
                        error(['enhancive部分的网络结构必须是一个数组列表，' ...
                            '并且结构为对称奇数层!']);
                    else
                        flag = 1;
                        for i = 1:floor(obj.enhancive_layers(end) / 2)
                            if obj.architecture(enhancive_start + i - 1) ~= ...
                                    obj.architecture(enhancive_end - i + 1)
                                flag = 0;
                                break;
                            end
                        end
                    end
                    if ~flag
                        error(['enhancive部分的网络结构必须是一个数组列表，' ...
                            '并且结构为对称奇数层!']);
                    end
                    
                    start = start + 1;
                end
                
                obj.is_enhancive = is_enhancive;
            end
        end
        function initialize_parameters(obj, theta4AEs)
            %初始化网络参数
            
            obj.theta = zeros(obj.parameters_num, 1);
            if exist('theta4AEs', 'var')
                if length(theta4AEs) == obj.parameters_num4AEs
                    obj.theta4AEs = theta4AEs;
                    % 将 AEs 的部分参数赋值给 SAE
                    obj.theta4AEs_to_theta();
                else
                    error(['传入的theta参数维度错误：应该为 ' ...
                        num2str(obj.parameters_num4AEs) ' 维！']);
                end
            else
                % 直接初始化为0，是因为调用AE类，AE类会自行初始化
                obj.theta4AEs = zeros(obj.parameters_num4AEs, 1);
            end
        end
        function theta4AEs_to_theta(obj)
            %将 AEs 的部分参数赋值给 Enhancive_Learning_SAE
            
            start_index     = 1; % Enhancive_Learning_SAE 系数的下标（起点）
            start_index4AEs = 1; % AEs 系数的下标（起点）
            for layer = 1:(obj.layers - 2)
                % Enhancive_Learning_SAE 及 AEs 系数的下标（终点）
                end_index = start_index - 1 + ...
                    (obj.architecture(layer) + 1) * obj.architecture(layer + 1);
                end_index4AEs = start_index4AEs - 1 + ...
                    (obj.architecture(layer) + 1) * obj.architecture(layer + 1);
                
                obj.theta(start_index:end_index) = ...
                    obj.theta4AEs(start_index4AEs:end_index4AEs);
                
                % 更新起点
                if obj.is_enhancive(layer) == 0
                    start_index     = end_index + 1;
                    start_index4AEs = end_index4AEs + ...
                        obj.architecture(layer) * (obj.architecture(layer + 1) + 1) + 1;
                else
                    start_index     = end_index + 1;
                    start_index4AEs = end_index4AEs + 1;
                end
            end
            obj.theta(start_index:end) = ...
                obj.theta4AEs(start_index4AEs:end);
        end
    end
    methods(Static)
        function description()
            %对该网络类型的描述
            disp_info = [sprintf('\n这是一个全连接增强学习栈式自编码器 Enhancive Learning Stacked Auto-Encoder！\n'), ...
                sprintf('工作机制为：先利用AE实现预训练，最后用BP微调，\n'),...
                sprintf('\t\t   但其中部分网络结构使用了Generative AutoEncoder，以实现增强或残差学习。\n\n'),...
                sprintf('本类的实现上调用了AutoEncoder、Generative_AutoEncoder类进行预训练，最后自身实现微调。\n'),...
                sprintf('-必须初始化的参数为：网络框架 architecture；\n'),...
                sprintf('-可选初始化的参数为：激活函数列表 activations，网络选项 options，网络参数 theta；\n'),...
                sprintf('\t 可选的激活函数activations有：Sigmoid, tanh, ReLU, leaky_ReLU, parameter_ReLU, softmax。\n'),...
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
                sprintf('\n')];
            disp(disp_info);
        end
    end
end