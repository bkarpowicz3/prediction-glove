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

%% Extract Features 

feat1 = extractFeatures_v2(ecog1, sR);
feat2 = extractFeatures_v2(ecog2, sR);
feat3 = extractFeatures_v2(ecog3, sR);

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

%% Testing extract features

testfeat1 = extractFeatures_v2(test1, sR);
testfeat2 = extractFeatures_v2(test2, sR);
testfeat3 = extractFeatures_v2(test3, sR);

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

%% Logistic Regression

% Threshold at 1.4 
threshold = 1.4;
biglove1 = (glove1 > threshold);
biglove2 = (glove2 > threshold);
biglove3 = (glove3 > threshold);
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
figure % change threshold for glove 3 to be at 0.5
plot(glove3(:,3))
hold on
plot(biglove{1, 3}(:,3))

%% Combine logistic with linear regression
