function Y = linreg(features, labels, testing)
% input:    features - matrix of predictors where rows = time bins, cols =
% features for every channel for training
%           labels - matrix of training glove positions for each finger
%           testing - matrix to test model on
% output: predictions of linear regression model

N = 3;
M = size(features, 1);
numFeats = size(features, 2);
featsPer = 6;                   % features/channel
numChannels = numFeats/featsPer;       % assuming 6 features per channel

% create R matrix for testing (if different than training)
testM = size(testing, 1);
testR = ones(testM-N, numFeats*N+1);
if ~isequal(features, testing)
    for i = 1:(testM-N+1)
        for k = 1:numChannels       % over # channels
            row = testing(i:i+N-1, ((k-1)*featsPer+1):((k-1)*featsPer+featsPer))';
            row = reshape(row, 1, featsPer*N);
            idx = (k-1)*N*featsPer+2;
            testR(i, idx:idx+N*featsPer-1) = row;
        end
    end
end

R = ones(M-N, numFeats*N+1);
%for each row
for i = 1:(M-N+1)
    % fixed pretty sure...
    for j = 1:numChannels       % over # channels
        row = features(i:i+N-1, ((j-1)*featsPer+1):((j-1)*featsPer+featsPer))';
        row = reshape(row, 1, featsPer*N);
        idx = (j-1)*N*featsPer+2;
        R(i, idx:idx+N*featsPer-1) = row;
    end
end 

a = R'*R;
ainv = a \ eye(size(a, 1)); % need to compute inverse this way or else you get Inf 
B = ainv*(R'*labels(N:end, :));

% depending on testing data
if isequal(features, testing)
    Y = R*B;
else
    Y = testR*B;
end

end
