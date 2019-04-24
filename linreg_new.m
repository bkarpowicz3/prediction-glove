function Y = linreg_new(features, labels, testing)

N = 3;
M = size(features, 1)-N;

%create R matrix for test (if different than train) 
testR = [];
start = 1;
if ~isequal(features, testing)
    %for each row in R 
    for i = 1:M
        row = [];
        row(end+1) = 1;
        %for each feature
        for j = 1:size(testing,2)
            row(end+1:end+N) = testing(start:start+N-1, j);
        end 
        start = start + 1;
        testR = [testR; row];
    end 
end 

%training R matrix 
%for each row in R 
R = [];
start = 1;
for i = 1:M
    row = [];
    row(end+1) = 1;
    %for each feature
    for j = 1:size(features,2)
        row(end+1:end+N) = features(start:start+N-1, j);
    end 
    start = start + 1;
    R = [R; row];
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