function Y = lassoReg(R, labels, testR)
% input:    R - R matrix for training
%           labels - matrix of training glove positions for 1 finger
%           testR - R matrix to test model on
% output: predictions of linear regression model

N = 4;

% train model
model = lasso(R, labels(N:end)); 

% depending on testing data
if isequal(R, testR)
    Y = R*model(:, 1);
else
    Y = testR*model(:, 1);
end

end
