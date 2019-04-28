%% Testing extract features

testfeat1 = extractFeatures_v1(test1, sR);
testfeat2 = extractFeatures_v1(test2, sR);
testfeat3 = extractFeatures_v1(test3, sR);

save('testfeatures.mat', 'testfeat1', 'testfeat2', 'testfeat3');

%%
numFeats = 9;
testpred1 = linreg(feat1, glove1_down, testfeat1, numFeats);
testpred2 = linreg(feat2, glove2_down, testfeat2, numFeats);
testpred3 = linreg(feat3, glove2_down, testfeat3, numFeats);

testup1 = [];
testup2 = [];
testup3 = [];

for i = 1:5
    testup1(:, i) = spline(1:size(testpred1, 1), testpred1(:, i), 1:1/50:size(testpred1, 1)); %off by 1 problem?? should be 1/50
    testup2(:, i) = spline(1:size(testpred2, 1), testpred2(:, i), 1:1/50:size(testpred2, 1));
    testup3(:, i) = spline(1:size(testpred3, 1), testpred3(:, i), 1:1/50:size(testpred3, 1));
end 

testup1 = [zeros(150, 5); testup1(1:147350, :)];
testup2 = [zeros(150, 5); testup2(1:147350, :)];
testup3 = [zeros(150, 5); testup3(1:147350, :)];

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

min = 1;
max = 16000;

figure();
subplot(4,1,1)
plot(up1(:, 1));
xlim([min max])
subplot(4,1,2)
plot(up1(:, 2))
xlim([min max])
subplot(4,1,3)
plot(up1(:, 3));
xlim([min max])
subplot(4,1,4)
xlim([min max])
plot(up1(:, 5)); 
xlim([min max])

figure();
xlim([min max])
subplot(4,1,1)
plot(up2(:, 1));
subplot(4,1,2)
plot(up2(:, 2))
subplot(4,1,3)
plot(up2(:, 3));
subplot(4,1,4)
plot(up2(:, 5)); 

figure();
xlim([min max])
subplot(4,1,1)
plot(up3(:, 1));
subplot(4,1,2)
plot(up3(:, 2))
subplot(4,1,3)
plot(up3(:, 3));
subplot(4,1,4)
plot(up3(:, 5)); 

%% Submit

predicted_dg = cell(3, 1);
predicted_dg{1} = testup1(1:147500, 1:5);
predicted_dg{2} = testup2(1:147500, 1:5);
predicted_dg{3} = testup3(1:147500, 1:5);

save('linregtest.mat', 'predicted_dg');