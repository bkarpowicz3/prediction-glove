% function train(radio, sales, weight, bias, learning_rate, iters)
function [Y, weights, cost_history] = linregiter(features, targets, weights, learning_rate, iters)
    % returns   Y: predictions
    %           weights: matrix to multiply features by to get predictions
    %           cost_history: MSE values over iterations

    cost_history = zeros(1, iters);
    
    features = normalize(features);
    
%     bias = np.ones(shape=(len(features),1))
%     features = np.append(bias, features, axis=1)
    bias = ones(size(features, 1), 1);
    features = [bias features];

%     for i in range(iters):
    for i = 1:iters
%         weight,bias = update_weights(radio, sales, weight, bias, learning_rate)
        weights = update_weights_vectorized(features, targets, weights, learning_rate);

%         #Calculate cost for auditing purposes
        cost = cost_function(features, targets, weights);
%         cost_history.append(cost)
        cost_history(i) = cost;

%         # Log Progress
%         if i % 10 == 0:
        if mod(i, 250) == 0
%             print "iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2}".format(i, weight, bias, cost)
            disp(['iter: ', num2str(i), ' cost: ', num2str(cost)])
        end
    end
    
%     return weight, bias, cost_history
    Y = predict(features, weights);
end

function features = normalize(features)
%     **
%     features     -   (200, 3)
%     features.T   -   (3, 200)
% 
%     We transpose the input matrix, swapping
%     cols and rows to make vector math easier
%     **

%     for feature in features.T:
    for i = 1:size(features, 2) % iterate over column
%         fmean = np.mean(feature)
%         frange = np.amax(feature) - np.amin(feature)
        fmean = mean(features(:, i));
        frange = max(features(:, i)) - min(features(:, i));

%         #Vector Subtraction
%         feature -= fmean
        features(:, i) = features(:, i) - fmean;

%         #Vector Division
%         feature /= frange
        features(:, i) = features(:, i) / frange;
    end
end

function weightsnew = update_weights_vectorized(features, targets, weights, lr)
%     **
%     gradient = X.T * (predictions - targets) / N
%     X: (200, 3)
%     Targets: (200, 1)
%     Weights: (3, 1)
%     **

%     companies = len(X)
    companies = size(features, 1);

%     #1 - Get Predictions
    predictions = predict(features, weights);

%     #2 - Calculate error/loss
    error = targets - predictions;

%     #3 Transpose features from (200, 3) to (3, 200)
%     # So we can multiply w the (200,1)  error matrix.
%     # Returns a (3,1) matrix holding 3 partial derivatives --
%     # one for each feature -- representing the aggregate
%     # slope of the cost function across all observations

%     gradient = np.dot(-X.T,  error)
    gradient = (-features')*error;

%     #4 Take the average error derivative for each feature
%     gradient /= companies
    gradient = gradient / companies;

%     #5 - Multiply the gradient by our learning rate
%     gradient *= lr
    gradient = gradient * lr;

%     #6 - Subtract from our weights to minimize cost
%     weights -= gradient
    weights = weights - gradient;

    weightsnew = weights;
end

function cost = cost_function(features, targets, weights)
%     **
%     features:(200,3)
%     targets: (200,1)
%     weights:(3,1)
%     returns average squared error among predictions
%     **

%     N = len(targets)
    N = size(targets, 1);

    predictions = predict(features, weights);

    % Matrix math lets use do this without looping
%     sq_error = (predictions - targets)**2
    sq_error = (predictions - targets).^2;

    % Return average squared error among predictions
%     return 1.0/(2*N) * sq_error.sum()
    cost = 1/(2*N) * sum(sq_error);

end

function predictions = predict(features, weights)
%   **
%   features - (200, 3)
%   weights - (3, 1)
%   predictions - (200,1)
%   **

%   predictions = np.dot(features, weights)
    predictions = features*weights;
end