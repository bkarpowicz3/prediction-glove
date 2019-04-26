%% Load subject data 
% According to guide/recitation 

userID = 'renyueqi';
userlogin = 'ren_ieeglogin.bin';

session = IEEGSession('I521_Sub1_Training_ecog', userID, userlogin);
ecog1 = session.data(1).getvalues(1:300000, 1:62);
session = IEEGSession('I521_Sub1_Training_dg', userID, userlogin);
glove1 = session.data(1).getvalues(1:300000, 1:5);
session = IEEGSession('I521_Sub1_Leaderboard_ecog', userID, userlogin);
test1 = session.data(1).getvalues(1:147500, 1:62);

sR = session.data.sampleRate;

session = IEEGSession('I521_Sub2_Training_ecog', userID, userlogin);
ecog2 = session.data(1).getvalues(1:300000, 1:48);
session = IEEGSession('I521_Sub2_Training_dg', userID, userlogin);
glove2 = session.data(1).getvalues(1:300000, 1:5);
session = IEEGSession('I521_Sub2_Leaderboard_ecog', userID, userlogin);
test2 = session.data(1).getvalues(1:147500, 1:48);

session = IEEGSession('I521_Sub3_Training_ecog', userID, userlogin);
ecog3 = session.data(1).getvalues(1:300000, 1:64);
session = IEEGSession('I521_Sub3_Training_dg', userID, userlogin);
glove3 = session.data(1).getvalues(1:300000, 1:5);
session = IEEGSession('I521_Sub3_Leaderboard_ecog', userID, userlogin);
test3 = session.data(1).getvalues(1:147500, 1:64);

%% Preprocess glove data
% Eliminate DC component
glove1 = glove1 - mean(glove1);
glove2 = glove2 - mean(glove2);
glove3 = glove3 - mean(glove3);

% Sliding window average
M = @(x) mean(x);
winLen = 0.1; % s
winDisp = 0.05; % s
glove1 = MovingWinFeats(glove1, sR, winLen, winDisp, M);
glove2 = MovingWinFeats(glove2, sR, winLen, winDisp, M);
glove3 = MovingWinFeats(glove3, sR, winLen, winDisp, M);

% Common average referencing montage
car_ecog1 = ecog1;
car_ecog2 = ecog2;
car_ecog3 = ecog3;

for t = 1:size(ecog1, 1)
    car_ecog1(t, :) = ecog1(t, :) - mean(ecog1(t, :));
    car_ecog2(t, :) = ecog2(t, :) - mean(ecog2(t, :));
    car_ecog3(t, :) = ecog3(t, :) - mean(ecog3(t, :));
end

%% CAR visually does not make big difference
figure
plot(ecog1(:, 60))
hold on
plot(car_ecog1(:, 60))

%% Extract Features 

feat1 = extractFeatures_v1(ecog1, sR);
feat2 = extractFeatures_v1(ecog2, sR);
feat3 = extractFeatures_v1(ecog3, sR);

save('features.mat', 'feat1', 'feat2', 'feat3');

%% Downsample glove data 
% Need to bring samples down to every 50ms to align with features.

glove1_down = [];
glove2_down = [];
glove3_down = [];
for i = 1:5
    glove1_down(:, end+1) = decimate(glove1(:, i), 50);
    glove2_down(:, end+1) = decimate(glove2(:, i), 50);
    glove3_down(:, end+1) = decimate(glove3(:, i), 50);
end 

glove1_down = glove1_down(1:end-1, :);
glove2_down = glove2_down(1:end-1, :);
glove3_down = glove3_down(1:end-1, :);

%% Linear Regression 

Y1 = linreg_v2(feat1, glove1_down, feat1);
Y2 = linreg_v2(feat2, glove2_down, feat2);
Y3 = linreg_v2(feat3, glove3_down, feat3);

%% Cubic Interpolation of Results 
% Bring data from every 50ms back to 1000 Hz. 

up1 = [];
up2 = [];
up3 = [];

for i = 1:5
    up1(:, i) = spline(1:size(Y1, 1), Y1(:, i), 1:1/50:size(Y1, 1)); %off by 1 problem?? should be 1/50
    up2(:,i) = spline(1:size(Y2, 1), Y2(:,i), 1:1/50:size(Y2, 1));
    up3(:,i) = spline(1:size(Y3, 1), Y3(:,i), 1:1/50:size(Y3, 1));
end 

%% Zero pad upsampled 

up1 = [zeros(150, 5); up1; zeros(49, 5)];   % pad equivalent of 2 windows in the beginning
up2 = [zeros(150, 5); up2; zeros(49, 5)];
up3 = [zeros(150, 5); up3; zeros(49, 5)];

%% Visualize prediction of train data 

figure();
plot(up1(:, 1));
hold on;
plot(glove1(:, 1)); 

% this looks quite bad

%% Calculate correlation

corr1 = zeros(1, 5);
corr2 = zeros(1, 5);
corr3 = zeros(1, 5);
for i = 1:5             % iterate over fingers
    corr1(i) = corr(glove1(:, i), up1(:, i));
    corr2(i) = corr(glove2(:, i), up2(:, i));
    corr3(i) = corr(glove3(:, i), up3(:, i));
end

avgcorr1 = mean(corr1)
avgcorr2 = mean(corr2)
avgcorr3 = mean(corr3)

%% Cross Validation

rng default

numfold = 10;   % number of folds

ind = (size(glove1_down, 1)-mod(size(glove1_down, 1), numfold));    % truncate data to even #
ind = 1:ind;
numelem = length(ind)/numfold;    % # elements per fold
folds = cell(1, numfold);
for i = 0:length(folds)-1
    folds{i+1} = ind((i*numelem+1):((i+1)*numelem));
end

ind = 1:300000;     % fold indices for 1000 Hz data
numelem = length(ind)/numfold;    % # elements per fold
foldsfull = cell(1, numfold);
for i = 0:length(foldsfull)-1
    foldsfull{i+1} = ind((i*numelem+1):((i+1)*numelem));
end

% calculate validation error for each fold
crosscorr1 = zeros(numfold, 5);
crosscorr2 = zeros(numfold, 5);
crosscorr3 = zeros(numfold, 5);
for i = 1%:length(folds)     % fold that is testing set
    trainfold1 = [];
    fingers1 = [];
    trainfold2 = [];
    fingers2 = [];
    trainfold3 = [];
    fingers3 = [];

    % accumulate training data
    for j = 1:length(folds)
        if i ~= j
            trainfold1 = [trainfold1; feat1(folds{j}, :)];
            fingers1 = [fingers1; glove1_down(folds{j}, :)];
            
            trainfold2 = [trainfold2; feat2(folds{j}, :)];
            fingers2 = [fingers2; glove2_down(folds{j}, :)];
            
            trainfold3 = [trainfold3; feat3(folds{j}, :)];
            fingers3 = [fingers3; glove3_down(folds{j}, :)];
        end
    end
    
    % train model
    Y1 = linreg(trainfold1, fingers1, feat1(folds{i}, :));
    Y2 = linreg(trainfold2, fingers2, feat2(folds{i}, :));
    Y3 = linreg(trainfold3, fingers3, feat3(folds{i}, :));
    
    up1 = [];
    up2 = [];
    up3 = [];
    
    for l = 1:5
        up1(:, l) = spline(1:size(Y1, 1), Y1(:, l), 1:1/50:size(Y1, 1)); %off by 1 problem?? should be 1/50
        up2(:, l) = spline(1:size(Y2, 1), Y2(:, l), 1:1/50:size(Y2, 1));
        up3(:, l) = spline(1:size(Y3, 1), Y3(:, l), 1:1/50:size(Y3, 1));
    end
    
    up1 = [zeros(150, 5); up1; zeros(99, 5)];   % pad equivalent of 2 windows in the beginning
    up2 = [zeros(150, 5); up2; zeros(99, 5)];
    up3 = [zeros(150, 5); up3; zeros(99, 5)];
    
    testlabel1 = glove1(foldsfull{i}, :);
    testlabel2 = glove2(foldsfull{i}, :);
    testlabel3 = glove3(foldsfull{i}, :);
    for k = 1:5
        crosscorr1(i, k) = corr(testlabel1(:, k), up1(:, k));
        crosscorr2(i, k) = corr(testlabel2(:, k), up2(:, k));
        crosscorr3(i, k) = corr(testlabel3(:, k), up3(:, k));
    end
end

avgcorr1 = mean(crosscorr1)
avgcorr2 = mean(crosscorr2)
avgcorr3 = mean(crosscorr3)

totalcorr = [avgcorr1([1, 2, 3, 5]), avgcorr2([1, 2, 3, 5]), avgcorr3([1, 2, 3, 5])];
avgcorr = mean(totalcorr)

%% Testing extract features

testfeat1 = extractFeatures_v1(test1, sR);
testfeat2 = extractFeatures_v1(test2, sR);
testfeat3 = extractFeatures_v1(test3, sR);

save('testfeatures.mat', 'testfeat1', 'testfeat2', 'testfeat3');

%%

testpred1 = linreg_v2(feat1, glove1_down, testfeat1);
testpred2 = linreg_v2(feat2, glove2_down, testfeat2);
testpred3 = linreg_v2(feat3, glove2_down, testfeat3);

testup1 = [];
testup2 = [];
testup3 = [];

for i = 1:5
    testup1(:, i) = spline(1:size(testpred1, 1), testpred1(:, i), 1:1/50:size(testpred1, 1)); %off by 1 problem?? should be 1/50
    testup2(:, i) = spline(1:size(testpred2, 1), testpred2(:, i), 1:1/50:size(testpred2, 1));
    testup3(:, i) = spline(1:size(testpred3, 1), testpred3(:, i), 1:1/50:size(testpred3, 1));
end 

testup1 = [zeros(150, 5); testup1; zeros(49, 5)];
testup2 = [zeros(150, 5); testup2; zeros(49, 5)];
testup3 = [zeros(150, 5); testup3; zeros(49, 5)];

predicted_dg = cell(3, 1);
predicted_dg{1} = testup1(1:147500, 1:5);
predicted_dg{2} = testup2(1:147500, 1:5);
predicted_dg{3} = testup3(1:147500, 1:5);

save('checkpoint1.mat', 'predicted_dg');

%% Logistic Regression: Create Labels
% Threshold at 0.3 <-- 0.5 <-- 1.4 
threshold = 0.3;
biglove1 = double((glove1 > threshold));
biglove2 = double((glove2 > threshold));
biglove3 = double((glove3 > threshold));
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

%% Visualize threshold over raw data
figure % change threshold for glove 3 to be at 0.3
plot(glove1(:,3))
hold on
plot(biglove{1, 1}(:,3))

%% Downsample new labels to match features
biglove1_down = [];
biglove2_down = [];
biglove3_down = [];
% Temporary: can remove following 3 lines
biglove1 = double(biglove{1, 1});
biglove2 = double(biglove{1, 2});
biglove3 = double(biglove{1, 3});

for i = 1:5
    biglove1_down(:, end+1) = decimate(biglove1(:, i), 50);
    biglove2_down(:, end+1) = decimate(biglove2(:, i), 50);
    biglove3_down(:, end+1) = decimate(biglove3(:, i), 50);
end 

biglove1_down = biglove1_down(1:end-1, :);
biglove2_down = biglove2_down(1:end-1, :);
biglove3_down = biglove3_down(1:end-1, :);

%% Train logistic classifier --> Takes a looooooooooooooooooong time to run
% Tip: run each logistic training in parallel or on different machines
log1 = mnrfit(feat1, biglove1_down); % weights
prob1 = mnrval(log1, feat1); % probability of training data
log2 = mnrfit(feat2, biglove2_down);
prob2 = mnrval(log2, feat2);
log3 = mnrfit(feat3, biglove3_down);
prob3 = mnrval(log3, feat3);

save('logweights.mat', 'log1', 'log2', 'log3');
save('trainlogprob.mat', 'prob1', 'prob2', 'prob3');

%% Combine logistic with linear regression to generate predictions
testprob1 = mnrval(log1, testfeat1);
testprob2 = mnrval(log2, testfeat2);
testprob3 = mnrval(log3, testfeat3);

% Multiply linear with logistic regression result; alpha = 1
logweighted1 = testpred1.*testprob1;
logweighted2 = testpred2.*testprob2;
logweighted3 = testpred3.*testprob3;

% Spline interpolate
testup1 = [];
testup2 = [];
testup3 = [];

for i = 1:5
    testup1(:, i) = spline(1:size(logweighted1, 1), logweighted1(:, i), 1:1/50:size(logweighted1, 1)); %off by 1 problem?? should be 1/50
    testup2(:, i) = spline(1:size(logweighted2, 1), logweighted2(:, i), 1:1/50:size(logweighted2, 1));
    testup3(:, i) = spline(1:size(logweighted3, 1), logweighted3(:, i), 1:1/50:size(logweighted3, 1));
end 

testup1 = [zeros(150, 5); testup1; zeros(49, 5)];
testup2 = [zeros(150, 5); testup2; zeros(49, 5)];
testup3 = [zeros(150, 5); testup3; zeros(49, 5)];

predicted_dg = cell(3, 1);
predicted_dg{1} = testup1(1:147500, 1:5);
predicted_dg{2} = testup2(1:147500, 1:5);
predicted_dg{3} = testup3(1:147500, 1:5);

save('logcheckpoint1.mat', 'predicted_dg');