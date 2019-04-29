%% Testing extract features

numFeats = 6;
testfeat1 = extractFeatures(test1, sR, numFeats);
testfeat2 = extractFeatures(test2, sR, numFeats);
testfeat3 = extractFeatures(test3, sR, numFeats);

save('testfeatures.mat', 'testfeat1', 'testfeat2', 'testfeat3');

%% Normalize features

for i = 1:size(testfeat1, 2)
    testfeat1(:, i) = (testfeat1(:, i) - mean(testfeat1(:, i)))./ std(testfeat1(:, i));
end

for i = 1:size(feat2, 2)
    testfeat2(:, i) = (testfeat2(:, i) - mean(testfeat2(:, i)))./ std(testfeat2(:, i));
end

for i = 1:size(testfeat3, 2)
    testfeat3(:, i) = (testfeat3(:, i) - mean(testfeat3(:, i)))./ std(testfeat3(:, i));
end

%%
numFeats = 9;
% testpred1 = linreg(feat1, glove1_down, testfeat1, numFeats);
% testpred2 = linreg(feat2, glove2_down, testfeat2, numFeats);
% testpred3 = linreg(feat3, glove2_down, testfeat3, numFeats);

testR1 = makeR(testfeat1, numFeats);
testR2 = makeR(testfeat2, numFeats);
testR3 = makeR(testfeat3, numFeats);

%% normalize R
features = testR1;
for i = 2:size(features, 2) % iterate over column
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
testR1 = features;

features = testR2;
for i = 2:size(features, 2) % iterate over column
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
testR2 = features;

features = testR3;
for i = 2:size(features, 2) % iterate over column
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
testR3 = features;

%%

testpred1 = [];
testpred2 = [];
testpred3 = [];
for i = 1:5
    testpred1(:, i) = testR1*weights1(:, i);
    testpred2(:, i) = testR2*weights2(:, i);
    testpred3(:, i) = testR3*weights3(:, i);
end

testup1 = [];
testup2 = [];
testup3 = [];

for i = 1:5
    testup1(:, i) = spline(1:size(testpred1, 1), testpred1(:, i), 1:1/50:size(testpred1, 1)); %off by 1 problem?? should be 1/50
    testup2(:, i) = spline(1:size(testpred2, 1), testpred2(:, i), 1:1/50:size(testpred2, 1));
    testup3(:, i) = spline(1:size(testpred3, 1), testpred3(:, i), 1:1/50:size(testpred3, 1));
end 

testup1 = [zeros(150, 5); testup1; zeros(99, 5)];
testup2 = [zeros(150, 5); testup2; zeros(99, 5)];
testup3 = [zeros(150, 5); testup3; zeros(99, 5)];

testup1 = testup1(1:147500, :);
testup2 = testup2(1:147500, :);
testup3 = testup3(1:147500, :);

% testup1 = [zeros(200, 5); testup1(1:147300, :)];
% testup2 = [zeros(200, 5); testup2(1:147300, :)];
% testup3 = [zeros(200, 5); testup3(1:147300, :)];

%% Postprocess with low pass

fc2 = 3;    % cutoff frequency
[b2, a2] = butter(6, fc2/(sR/2));
for i = 1:5
    testup1(:, i) = filtfilt(b2, a2, testup1(:, i));
    testup2(:, i) = filtfilt(b2, a2, testup2(:, i));
    testup3(:, i) = filtfilt(b2, a2, testup3(:, i));
end

%% Postprocess with low pass

fc2 = 3;    % cutoff frequency
[b2, a2] = butter(6, fc2/(sR/2));
for i = 1:5
    testup1(:, i) = filtfilt(b2, a2, testup1(:, i));
    testup2(:, i) = filtfilt(b2, a2, testup2(:, i));
    testup3(:, i) = filtfilt(b2, a2, testup3(:, i));
end

%% Predict binary state of glove using features

resub_bin1 = zeros(2949, 5);
resub_bin2 = zeros(2949, 5);
resub_bin3 = zeros(2949, 5);

for k = 1:5
%     model1 = fitcknn(feat1, glove1_bin(:,k), ...
%         'Distance', 'Euclidean', ...
%         'Exponent', [], ...
%         'NumNeighbors', 1, ...
%         'DistanceWeight', 'Equal', ...
%         'Standardize', true, ...
%         'ClassNames', [0; 1]);
%     model1 = fitctree(feat1, glove1_bin(:,k));
    model1 = TreeBagger(30, feat1, glove1_bin(:,k), 'Method', 'regression', 'OOBPrediction', 'on');
    [resubs, ~, ~] = predict(model1, testfeat1);
    resub_bin1(:,k) = resubs;
    
%     model2 = fitcknn(feat2, glove2_bin(:,k), ...
%         'Distance', 'Euclidean', ...
%         'Exponent', [], ...
%         'NumNeighbors', 1, ...
%         'DistanceWeight', 'Equal', ...
%         'Standardize', true, ...
%         'ClassNames', [0; 1]);
%     model2 = fitctree(feat2, glove2_bin(:,k));
    model2 = TreeBagger(30, feat2, glove2_bin(:,k), 'Method', 'regression', 'OOBPrediction', 'on');
    [resubs, ~, ~] = predict(model2, testfeat2);
    resub_bin2(:,k) = resubs;
    
%     model3 = fitcknn(feat3, glove3_bin(:,k), ...
%         'Distance', 'Euclidean', ...
%         'Exponent', [], ...
%         'NumNeighbors', 1, ...
%         'DistanceWeight', 'Equal', ...
%         'Standardize', true, ...
%         'ClassNames', [0; 1]);
%     model3 = fitctree(feat3, glove3_bin(:,k));
    model3 = TreeBagger(30, feat3, glove3_bin(:,k), 'Method', 'regression', 'OOBPrediction', 'on');
    [resubs, ~, ~] = predict(model3, testfeat3);
    resub_bin3(:,k) = resubs;
end 

%% Upsample Binary Predictions 

resub1 = zeros(147500, 5);
resub2 = zeros(147500, 5);
resub3 = zeros(147500, 5);

for i = 1:5 
    up = spline(1:size(resub_bin1, 1), resub_bin1(:,i), 1:1/50:size(resub_bin1, 1)); 
    up = [zeros(150, 1); up'; zeros(99, 1)];
    resub1(:,i) = up(1:147500);
    
    up = spline(1:size(resub_bin2, 1), resub_bin2(:,i), 1:1/50:size(resub_bin2, 1)); 
    up = [zeros(150, 1); up'; zeros(99, 1)];
    resub2(:,i) = up(1:147500);
    
    up = spline(1:size(resub_bin3, 1), resub_bin3(:,i), 1:1/50:size(resub_bin3, 1)); 
    up = [zeros(150, 1); up'; zeros(99, 1)];
    resub3(:,i) = up(1:147500);
end

%% Postprocess finger predictions 
sR = 1000;
thresh = 0.8;
below_thresh1 = testup1 .* (testup1 <= thresh);
above_thresh1 = testup1 .* (testup1 > thresh); 
thresh = 0.8;
below_thresh2 = testup2 .* (testup2 <= thresh);
above_thresh2 = testup2 .* (testup2 > thresh); 
thresh = 0.7;
below_thresh3 = testup3 .* (testup3 <= thresh);
above_thresh3 = testup3 .* (testup3 > thresh); 
M = @(x) mean(x);
winLen = 800e-3;
winDisp = 400e-3;

up1_new = zeros(147500, 5);
up2_new = zeros(147500, 5);
up3_new = zeros(147500, 5);

for i = 1:5
%     winLen = 800e-3;
%     winDisp = 400e-3;
    smoothed_glove1 = MovingWinFeats(below_thresh1(:,i), sR, winLen, winDisp, M);
    smoothed_glove1(end+1:end+3) = [0 0 0];
    smoothed_glove1_ = spline(1:length(smoothed_glove1), smoothed_glove1, 1:1/400:length(smoothed_glove1));  %zoInterp(smoothed_glove, 100);
    up1_new(:,i) = above_thresh1(:,i) + smoothed_glove1_(1:end-101)';
    
%     winLen = 1000e-3;
%     winDisp = 500e-3;
    smoothed_glove2 = MovingWinFeats(below_thresh2(:,i), sR, winLen, winDisp, M);
    smoothed_glove2(end+1:end+3) = [0 0 0];
    smoothed_glove2_ = spline(1:length(smoothed_glove2), smoothed_glove2, 1:1/400:length(smoothed_glove2));  %zoInterp(smoothed_glove, 100);
    up2_new(:,i) = above_thresh2(:,i) + smoothed_glove2_(1:end-101)';
    
%     winLen = 500e-3;
%     winDisp = 250e-3;
    smoothed_glove3 = MovingWinFeats(below_thresh3(:,i), sR, winLen, winDisp, M);
    smoothed_glove3(end+1:end+3) = [0 0 0];
    smoothed_glove3_ = spline(1:length(smoothed_glove3), smoothed_glove3, 1:1/400:length(smoothed_glove3));  %zoInterp(smoothed_glove, 100);
    up3_new(:,i) = above_thresh3(:,i) + smoothed_glove3_(1:end-101)';
end

%% Visualize prediction of train data 

minx = 1;
maxx = 16000;

figure();
subplot(4,1,1)
<<<<<<< HEAD
plot(up1_new(:, 1));
xlim([minx maxx])
subplot(4,1,2)
plot(up1_new(:, 2))
xlim([minx maxx])
subplot(4,1,3)
plot(up1_new(:, 3));
xlim([minx maxx])
subplot(4,1,4)
xlim([minx maxx])
plot(up1_new(:, 5)); 
xlim([minx maxx])

figure();
xlim([minx maxx])
subplot(4,1,1)
plot(up2_new(:, 1));
subplot(4,1,2)
plot(up2_new(:, 2))
subplot(4,1,3)
plot(up2_new(:, 3));
subplot(4,1,4)
plot(up2_new(:, 5)); 

% plot(testup2(:, 1).*resub2(:,1));
% subplot(4,1,2)
% plot(testup2(:, 2).*resub2(:,2))
% subplot(4,1,3)
% plot(testup2(:, 3).*resub2(:,3));
% subplot(4,1,4)
% plot(testup2(:, 5).*resub2(:,5)); 

figure();
xlim([minx maxx])
subplot(4,1,1)
plot(up3_new(:, 1));
subplot(4,1,2)
plot(up3_new(:, 2))
subplot(4,1,3)
plot(up3_new(:, 3));
subplot(4,1,4)
plot(up3_new(:, 5)); 

% plot(testup3(:, 1).*resub3(:,1));
% subplot(4,1,2)
% plot(testup3(:, 2).*resub3(:,2))
% subplot(4,1,3)
% plot(testup3(:, 3).*resub3(:,3));
% subplot(4,1,4)
% plot(testup3(:, 5).*resub3(:,5)); 

%% Submit

predicted_dg = cell(3, 1);

predicted_dg{1} = up1_new(1:147500, 1:5);
predicted_dg{2} = up2_new(1:147500, 1:5);
predicted_dg{3} = up3_new(1:147500, 1:5);

% predicted_dg{1} = testup1(1:147500, 1:5).*resub1;
% predicted_dg{2} = testup2(1:147500, 1:5).*resub2;
% predicted_dg{3} = testup3(1:147500, 1:5).*resub3;

save('linregitertest.mat', 'predicted_dg');