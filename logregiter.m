function weights = logregiter(features, target, num_steps, learning_rate, add_intercept)
    
    % returns weights of very large magnitude - just make sure you
    % normalize this 
    % you also might want to just generate this from feat?? 

    if add_intercept
        intercept = ones(size(features,1), 1);
        features = [intercept; features]; % Does R matrix already have 1's?
    end 
    
    weights = zeros(size(features,2), 1);
    
    for step = 1:num_steps
        scores = features * weights; %added 2 for dim %features * weights;
        predictions = sigmoid(scores);
        
        % update weights w/ gradient 
        output_error_signal = target - predictions;
        gradient = features' * output_error_signal; %dot(features', output_error_signal);
        weights = weights + learning_rate * gradient;
        
        % display log-likelihood every so often
        if mod(step, 1000) == 0
            disp(num2str(log_likelihood(features, target,weights)));
        end 
    end 

    function b = sigmoid(a) 
         b = 1./(1+exp(-1.*a));
    end 
    
    function ll = log_likelihood(features, target, weights) 
        scores = features * weights;
        ll = sum(target.*scores - log(1+exp(scores)));
    end 

    
%     function b = net_input(theta, x)
%         b = dot(x, theta);
%     end 
% 
%     function p = probability(theta, x)
%         p = sigmoid(net_inout(theta, x));
%     end 
% 
%     function total_cost = cost_function(theta, x, y)
%         m = size(x, 1);
%         total_cost = -1*(1/m) * sum(y * log(probability(theta, x)) + (1-y) * log(1 - probability(theta,x)));
%     end 
% 
%     function g = gradient(theta, x, y)
%         m = size(x, 1);
%         g = (1/m) * dot(x', sigmoid(net_input(theta, x)) - y);
%     end 
% 
%     function opt_weights = fit(x, y, theta)
%         options = optimoptions('fminunc','SpecifyObjectiveGradient',true);
%         opt_weights = fminunc(cost_function, theta, options);
%     end 

end

