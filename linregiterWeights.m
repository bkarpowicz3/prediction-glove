N = 4;
numFeats = 10;

features = makeR(feat1, numFeats);
features = features(:, 2:end);
weights1 = zeros(size(features, 2)+1, 5);
for finger = 1:5
    
    targets = [glove1_down(N:end, finger); glove1_down(1:N-1, finger)];
    [Y1, weights, cost_history] = linregiter(features, targets, weights1old(:, finger), .01, 5000);
    weights1(:, finger) = weights;
    
    disp(['subject 1 finger: ' num2str(finger)])
end

%%
features = makeR(feat2, numFeats);
features = features(:, 2:end);
weights2 = zeros(size(features, 2)+1, 5);
for finger = 1:5
    
    targets = [glove2_down(N:end, finger); glove2_down(1:N-1, finger)];
    [Y2, weights, cost_history] = linregiter(features, targets, weights2old(:, finger), .01, 5000);
    weights2(:, finger) = weights;
    
    disp(['subject 2 finger: ' num2str(finger)])
end

%%
features = makeR(feat3, numFeats);
features = features(:, 2:end);
weights3 = zeros(size(features, 2)+1, 5);
for finger = 1:5
    
    targets = [glove3_down(N:end, finger); glove3_down(1:N-1, finger)];
    [Y3, weights, cost_history] = linregiter(features, targets, weights3old(:, finger), .01, 5000);
    weights3(:, finger) = weights;
    
    disp(['subject 3 finger: ' num2str(finger)])
end

save('linRegWeights10.mat', 'weights1', 'weights2', 'weights3')