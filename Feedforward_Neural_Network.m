classdef (Abstract) Feedforward_Neural_Network < handle
    %一个全连接前向神经网络虚基类
    
    methods
        %前向计算输出
        target = predict(obj, input)
        %后向训练网络
        train(obj, input, target, option, theta)
    end
    
    methods(Static)
        function description()
            %对该网络类型的描述
            disp_info = [sprintf('\n这是一个全连接前向神经网络!\n'), ...
                sprintf('目前还是一个虚基类...\n')];
            disp(disp_info);
        end
    end
end