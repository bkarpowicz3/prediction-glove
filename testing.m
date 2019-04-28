%% Alternative prediction workflow:
% Prediction using linreg --> post process --> binary classifier

%% Obtain training and testing predictions
% Training
%% Linear Regression 
numFeats = 9;

Y1 = linreg(feat1, glove1_down, feat1, numFeats);
Y2 = linreg(feat2, glove2_down, feat2, numFeats);
Y3 = linreg(feat3, glove3_down, feat3, numFeats);

%% Cubic Interpolation of Results 
% Bring data from every 50ms back to 1000 Hz. 

up1 = [];
up2 = [];
up3 = [];

for i = 1:5
    up1(:, i) = spline(1:size(Y1, 1), Y1(:, i), 1:1/50:size(Y1, 1)); %off by 1 problem?? should be 1/50
    up2(:, i) = spline(1:size(Y2, 1), Y2(:, i), 1:1/50:size(Y2, 1));
    up3(:, i) = spline(1:size(Y3, 1), Y3(:, i), 1:1/50:size(Y3, 1));
end 

%% Zero pad upsampled 

% up1 = [zeros(150, 5); up1; zeros(99, 5)];   % pad equivalent of 3 windows in the beginning
% up2 = [zeros(150, 5); up2; zeros(99, 5)];
% up3 = [zeros(150, 5); up3; zeros(99, 5)];

up1 = [zeros(150, 5); up1(1:299850, :)];   % pad equivalent of 3 windows in the beginning
up2 = [zeros(150, 5); up2(1:299850, :)];
up3 = [zeros(150, 5); up3(1:299850, :)];

%% Postprocess finger predictions 
sR = 1000;
thresh = 0.8;
below_thresh1 = up1 .* (up1 <= thresh);
above_thresh1 = up1 .* (up1 > thresh); 
thresh = 0.8;
below_thresh2 = up2 .* (up2 <= thresh);
above_thresh2 = up2 .* (up2 > thresh); 
thresh = 0.7;
below_thresh3 = up3 .* (up3 <= thresh);
above_thresh3 = up3 .* (up3 > thresh); 
M = @(x) mean(x);
winLen = 800e-3;
winDisp = 400e-3;

trainpostproc1 = zeros(300000, 5);
trainpostproc2 = zeros(300000, 5);
trainpostproc3 = zeros(300000, 5);

for i = 1:5
%     winLen = 800e-3;
%     winDisp = 400e-3;
    smoothed_glove1 = MovingWinFeats(below_thresh1(:,i), sR, winLen, winDisp, M);
    smoothed_glove1(end+1:end+3) = [0 0 0];
    smoothed_glove1_ = spline(1:length(smoothed_glove1), smoothed_glove1, 1:1/400:length(smoothed_glove1));  %zoInterp(smoothed_glove, 100);
    trainpostproc1(:,i) = above_thresh1(:,i) + smoothed_glove1_(1:end-401)';
    
%     winLen = 1000e-3;
%     winDisp = 500e-3;
    smoothed_glove2 = MovingWinFeats(below_thresh2(:,i), sR, winLen, winDisp, M);
    smoothed_glove2(end+1:end+3) = [0 0 0];
    smoothed_glove2_ = spline(1:length(smoothed_glove2), smoothed_glove2, 1:1/400:length(smoothed_glove2));  %zoInterp(smoothed_glove, 100);
    trainpostproc2(:,i) = above_thresh2(:,i) + smoothed_glove2_(1:end-401)';
    
%     winLen = 500e-3;
%     winDisp = 250e-3;
    smoothed_glove3 = MovingWinFeats(below_thresh3(:,i), sR, winLen, winDisp, M);
    smoothed_glove3(end+1:end+3) = [0 0 0];
    smoothed_glove3_ = spline(1:length(smoothed_glove3), smoothed_glove3, 1:1/400:length(smoothed_glove3));  %zoInterp(smoothed_glove, 100);
    trainpostproc3(:,i) = above_thresh3(:,i) + smoothed_glove3_(1:end-401)';
end
trainpostproc = {trainpostproc1; trainpostproc2; trainpostproc3};

% Testing 
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

testpostproc1 = zeros(147500, 5);
testpostproc2 = zeros(147500, 5);
testpostproc3 = zeros(147500, 5);

for i = 1:5
%     winLen = 800e-3;
%     winDisp = 400e-3;
    smoothed_glove1 = MovingWinFeats(below_thresh1(:,i), sR, winLen, winDisp, M);
    smoothed_glove1(end+1:end+3) = [0 0 0];
    smoothed_glove1_ = spline(1:length(smoothed_glove1), smoothed_glove1, 1:1/400:length(smoothed_glove1));  %zoInterp(smoothed_glove, 100);
    testpostproc1(:,i) = above_thresh1(:,i) + smoothed_glove1_(1:end-101)';
    
%     winLen = 1000e-3;
%     winDisp = 500e-3;
    smoothed_glove2 = MovingWinFeats(below_thresh2(:,i), sR, winLen, winDisp, M);
    smoothed_glove2(end+1:end+3) = [0 0 0];
    smoothed_glove2_ = spline(1:length(smoothed_glove2), smoothed_glove2, 1:1/400:length(smoothed_glove2));  %zoInterp(smoothed_glove, 100);
    testpostproc2(:,i) = above_thresh2(:,i) + smoothed_glove2_(1:end-101)';
    
%     winLen = 500e-3;
%     winDisp = 250e-3;
    smoothed_glove3 = MovingWinFeats(below_thresh3(:,i), sR, winLen, winDisp, M);
    smoothed_glove3(end+1:end+3) = [0 0 0];
    smoothed_glove3_ = spline(1:length(smoothed_glove3), smoothed_glove3, 1:1/400:length(smoothed_glove3));  %zoInterp(smoothed_glove, 100);
    testpostproc3(:,i) = above_thresh3(:,i) + smoothed_glove3_(1:end-101)';
end
testpostproc = {testpostproc1; testpostproc2; testpostproc3};

%% Train logistic classifier from predictions and binary glove labels
% Create binary glove labels

threshold1 = 1;
biglove1 = (glove1 > threshold1);
threshold2 = 0.8;
biglove2 = (glove2 > threshold2);
threshold3 = 0.6;
biglove3 = (glove3 > threshold3);

biglove1 = double(biglove1);
biglove2 = double(biglove2);
biglove3 = double(biglove3);

%%
% Average neighboring class 1 labels within 4s
neighborLen = 4*sR;
biglove = {biglove1, biglove2, biglove3};
for g = 1:3
    currglove = biglove{1, g};
    for finger = 1:5
        data = currglove(:, finger);
        newlabels = data;
        for i = 1:length(data) - neighborLen
            window = data(i:i+neighborLen);
            indices = find(window == 1);
            if length(indices) > 1
                window(indices(1):indices(end)) = ones(1, indices(end)-indices(1) + 1);
            end
            newlabels(i:i+neighborLen) = window;
        end
        currglove(:, finger) = newlabels;
    end
    biglove{1, g} = currglove;
end
disp('Finished logistic thresholding')

for s = 1:3
    biglove{s}(biglove{s}(:,:) == 1) = 2;
    biglove{s}(biglove{s}(:,:) == 0) = 1;
end

%% Perform logistic regression
% On training
log_weights = cell(3, 5);
probabilities = cell(3, 5);
for sub = 1:3
    for finger = 1:5
        log_weights{sub, finger} = ...
            mnrfit(trainpostproc{sub}(:, finger), biglove{sub}(:, finger));
        probabilities{sub, finger} = ...
            mnrval(log_weights{sub,finger}, trainpostproc{sub}(:, finger));
    end
end

% On testing
testpredictions = cell(3, 5);
for sub = 1:3
    for finger = 1:5
        testpredictions{sub, finger} = ...
            mnrval(log_weights{sub,finger}, testpostproc{sub}(:, finger));
    end
end
disp('Finished log reg')

%% Visualize the binary predicted training data
figure
subplot(3,1,1)
plot(glove1(:,1))
subplot(3,1,2)
plot(biglove{1}(:,1))
subplot(3,1,3)
plot(probabilities{1,1}(:,2))

%% Visualize binary predicted testing data
figure % sub 1
subplot(4,1,1)
plot(testup1(:,1))
hold on
plot(testpredictions{1,1}(:,2))
subplot(4,1,2)
plot(testup1(:,2))
hold on
plot(testpredictions{1,2}(:,2))
subplot(4,1,3)
plot(testup1(:,3))
hold on
plot(testpredictions{1,3}(:,2))
subplot(4,1,4)
plot(testup1(:,5))
hold on
plot(testpredictions{1,5}(:,2))

figure % sub 2
subplot(4,1,1)
plot(testup2(:,1))
hold on
plot(testpredictions{2,1}(:,2))
subplot(4,1,2)
plot(testup2(:,2))
hold on
plot(testpredictions{2,2}(:,2))
subplot(4,1,3)
plot(testup2(:,3))
hold on
plot(testpredictions{2,3}(:,2))
subplot(4,1,4)
plot(testup2(:,5))
hold on
plot(testpredictions{2,5}(:,2))

figure % sub 3
subplot(4,1,1)
plot(testup3(:,1))
hold on
plot(testpredictions{3,1}(:,2))
subplot(4,1,2)
plot(testup3(:,2))
hold on
plot(testpredictions{3,2}(:,2))
subplot(4,1,3)
plot(testup3(:,3))
hold on
plot(testpredictions{3,3}(:,2))
subplot(4,1,4)
plot(testup3(:,5))
hold on
plot(testpredictions{3,5}(:,2))

%% Combine binary classifier probability with predictions
weightedtest1 = [];
weightedtest2 = [];
weightedtest3 = [];

for f = 1:5
    weightedtest1(:,f) = testup1(:,f).*testpredictions{1,f}(:,2);
    weightedtest2(:,f) = testup2(:,f).*testpredictions{2,f}(:,2);
    weightedtest3(:,f) = testup3(:,f).*testpredictions{3,f}(:,2);
end

%% Visualize submission
figure
for i = 1:4
    subplot(4,1,i)
    if i == 4
        i = 5;
    end
    plot(weightedtest1(:,i))
end

figure
for i = 1:4
    subplot(4,1,i)
    if i == 4
        i = 5;
    end
    plot(weightedtest2(:,i))
end

figure
for i = 1:4
    subplot(4,1,i)
    if i == 4
        i = 5;
    end
    plot(weightedtest3(:,i))
end
%% Save checkpoint
predicted_dg = cell(3, 1);
predicted_dg{1} = weightedtest1;
predicted_dg{2} = weightedtest2;
predicted_dg{3} = weightedtest3;

save('altworkflow.mat', 'predicted_dg');