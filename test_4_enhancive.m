% clear,clc;
% obj.architecture = [30, 28, 25, 28, 30, 4];
% obj.is_enhancive = [1 1 1 1 1 0];
% obj.activations = {'ReLU', 'ReLU', 'ReLU', 'ReLU', 'softmax'};
% obj.enhancive_layers = [];
% obj.layers = length(obj.architecture);
% 
% 
% obj.parameters_num = sum((obj.architecture(1:end-1) + 1) .* obj.architecture(2:end));
% if strcmp(obj.activations{end}, 'softmax')
%     obj.parameters_num = obj.parameters_num - obj.architecture(end);
% end
% obj.parameters_num4AEs = sum((obj.architecture(1:end-1) + 1) .* obj.architecture(2:end)) + ...
%     sum(obj.architecture(1:end-2) .* (obj.architecture(2:end-1) + 1));
% if strcmp(obj.activations{end}, 'softmax')
%     obj.parameters_num4AEs = obj.parameters_num4AEs - obj.architecture(end);
% end
% 
% 
% start = 1;
% while start <= obj.layers
%     if obj.is_enhancive(start)
%         enhancive_start = start;
%         enhancive_end   = start;
%         while enhancive_end <= obj.layers
%             obj.parameters_num4AEs = obj.parameters_num4AEs - ...
%                 obj.architecture(enhancive_end) * obj.architecture(enhancive_end + 1) -...
%                 obj.architecture(enhancive_end + 1);
%             
%             enhancive_end = enhancive_end + 1;
%             if ~obj.is_enhancive(enhancive_end)
%                 enhancive_end = enhancive_end - 1;
%                 break;
%             end
%         end
%         
%         obj.parameters_num4AEs = obj.parameters_num4AEs + ...
%             obj.architecture(enhancive_end) * obj.architecture(enhancive_end + 1) +...
%             obj.architecture(enhancive_end + 1);
%         start = enhancive_end;
%         
%         obj.enhancive_layers = [obj.enhancive_layers ...
%             enhancive_end - enhancive_start + 1];
%     end
%     if mod(obj.enhancive_layers(end), 2) == 0
%         error(['enhancive部分的网络结构必须是一个数组列表，' ...
%             '并且结构为对称奇数层!']);
%     else
%         flag = 1;
%         for i = 1:floor(obj.enhancive_layers(end) / 2)
%             if obj.architecture(enhancive_start + i - 1) ~= ...
%                     obj.architecture(enhancive_end - i + 1)
%                 flag = 0;
%                 break;
%             end
%         end
%     end
%     if ~flag
%         error(['enhancive部分的网络结构必须是一个数组列表，' ...
%             '并且结构为对称奇数层!']);
%     end
%     
%     start = start + 1;
% end
% 














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
gaenn = Enhancive_Learning_SAE(architecture, activations, options, is_enhancive);
gaenn.train(input, 500, 1000);
