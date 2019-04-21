function Y = linreg(features, labels)

R = [];
N = 3;
M = 147500-N;
% Dimensions of R
n = 1 + 62*(N+1)*6; % 1488, number of features/repeats/channels, column
cols = 2:4:n; % All unique feature column indices
% First column is the intercept term
R(:, 1) = ones(M, 1);

for c = 1:length(cols)
    for i = 1:4
        R(:, cols(c) + (i-1)) = features(i:(M + i-1), c);
    end
end

if size(R, 1) ~= M && size(R, 2) ~= n
    error('Size of R matrix incorrect')
end

% Compute the linear regression
a = R'*R;
ainv = a / eye(size(a)); % need to compute inverse this way or else you get Inf 
B = ainv*R'*labels;
Y = R*B;
end