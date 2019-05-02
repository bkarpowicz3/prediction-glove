%% Load subject data 
% According to guide/recitation 

session = IEEGSession('I521_Sub1_Training_ecog', 'bkarpowicz3', 'bka_ieeglogin.bin');
ecog1 = session.data(1).getvalues(1:300000, 1:62);
session = IEEGSession('I521_Sub1_Training_dg', 'bkarpowicz3', 'bka_ieeglogin.bin');
glove1 = session.data(1).getvalues(1:300000, 1:5);
session = IEEGSession('I521_Sub1_Leaderboard_ecog', 'bkarpowicz3', 'bka_ieeglogin.bin');
test1 = session.data(1).getvalues(1:147500, 1:62);

sR = session.data.sampleRate;

session = IEEGSession('I521_Sub2_Training_ecog', 'bkarpowicz3', 'bka_ieeglogin.bin');
ecog2 = session.data(1).getvalues(1:300000, 1:48);
session = IEEGSession('I521_Sub2_Training_dg', 'bkarpowicz3', 'bka_ieeglogin.bin');
glove2 = session.data(1).getvalues(1:300000, 1:5);
session = IEEGSession('I521_Sub2_Leaderboard_ecog', 'bkarpowicz3', 'bka_ieeglogin.bin');
test2 = session.data(1).getvalues(1:147500, 1:48);

session = IEEGSession('I521_Sub3_Training_ecog', 'bkarpowicz3', 'bka_ieeglogin.bin');
ecog3 = session.data(1).getvalues(1:300000, 1:64);
session = IEEGSession('I521_Sub3_Training_dg', 'bkarpowicz3', 'bka_ieeglogin.bin');
glove3 = session.data(1).getvalues(1:300000, 1:5);
session = IEEGSession('I521_Sub3_Leaderboard_ecog', 'bkarpowicz3', 'bka_ieeglogin.bin');
test3 = session.data(1).getvalues(1:147500, 1:64);

%% Load features 

load('4-27_features9.mat')

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

%% Get binary labels for glove data 

thresh = 1;
glove1_bin = glove1_down >= thresh;
thresh = 0.8;
glove2_bin = glove2_down >= thresh;
thresh = 0.6;
glove3_bin = glove3_down >= thresh;

%% Linear Regression 

numFeats = 9;

load('linRegIterTrainWeights.mat');

R1 = makeR(feat1, numFeats); 
for i = 2:size(R1, 2)
    R1(:,i) = (R1(:,i) - mean(R1(:,i)))/range(R1(:,i)); 
end 
R2 = makeR(feat2, numFeats); 
for i = 2:size(R2, 2)
    R2(:,i) = (R2(:,i) - mean(R2(:,i)))/range(R2(:,i)); 
end 
R3 = makeR(feat3, numFeats); 
for i = 2:size(R3, 2)
    R3(:,i) = (R3(:,i) - mean(R3(:,i)))/range(R3(:,i)); 
end 

Y1 = R1*weights1;
Y2 = R2*weights2;
Y3 = R3*weights3;

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

up1 = [zeros(150, 5); up1(1:299850, :)];   % pad equivalent of 2 windows in the beginning
up2 = [zeros(150, 5); up2(1:299850, :)];
up3 = [zeros(150, 5); up3(1:299850, :)];

%% Postprocess linreg predictions by filtering

fc2 = 3;    % cutoff frequency
[b2, a2] = butter(6, fc2/(sR/2));
for i = 1:5
    up1(:, i) = filtfilt(b2, a2, up1(:, i));
    up2(:, i) = filtfilt(b2, a2, up2(:, i));
    up3(:, i) = filtfilt(b2, a2, up3(:, i));
end

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

up1 = zeros(300000, 5);
up2 = zeros(300000, 5);
up3 = zeros(300000, 5);

for i = 1:5
%     winLen = 800e-3;
%     winDisp = 400e-3;
    smoothed_glove1 = MovingWinFeats(below_thresh1(:,i), sR, winLen, winDisp, M);
    smoothed_glove1(end+1:end+2) = [0 0];
    smoothed_glove1_ = spline(1:length(smoothed_glove1), smoothed_glove1, 1:1/400:length(smoothed_glove1));  %zoInterp(smoothed_glove, 100);
    up1(:,i) = above_thresh1(:,i) + smoothed_glove1_(1:end-1)';
    
%     winLen = 1000e-3;
%     winDisp = 500e-3;
    smoothed_glove2 = MovingWinFeats(below_thresh2(:,i), sR, winLen, winDisp, M);
    smoothed_glove2(end+1:end+2) = [0 0];
    smoothed_glove2_ = spline(1:length(smoothed_glove2), smoothed_glove2, 1:1/400:length(smoothed_glove2));  %zoInterp(smoothed_glove, 100);
    up2(:,i) = above_thresh2(:,i) + smoothed_glove2_(1:end-1)';
    
%     winLen = 500e-3;
%     winDisp = 250e-3;
    smoothed_glove3 = MovingWinFeats(below_thresh3(:,i), sR, winLen, winDisp, M);
    smoothed_glove3(end+1:end+2) = [0 0];
    smoothed_glove3_ = spline(1:length(smoothed_glove3), smoothed_glove3, 1:1/400:length(smoothed_glove3));  %zoInterp(smoothed_glove, 100);
    up3(:,i) = above_thresh3(:,i) + smoothed_glove3_(1:end-1)';
end

%% Visualize prediction of train data 

figure();
plot(up1(:, 1));
hold on;
plot(glove1(:, 1)); 

%% Predict probabilities of binary state of glove using logreg 

f_weights1 = zeros(size(R1, 2), 5);
f_weights2 = zeros(size(R2, 2), 5);
f_weights3 = zeros(size(R3, 2), 5);

for i = 1:5 
    w1 = logregiter(R1, double(glove1_bin(:,i)), 10000, 0.001, false);
    f_weights1(:,i) = w1;
end 

for i = 1:5 
    w2 = logregiter(R2, double(glove2_bin(:,i)), 10000, 0.001, false);
    f_weights2(:,i) = w2;
end 

for i = 1:5 
    w3 = logregiter(R3, double(glove3_bin(:,i)), 10000, 0.001, false);
    f_weights3(:,i) = w3;
end 

binlab1 = R1 * f_weights1;
binlab2 = R2 * f_weights2;
binlab3 = R3 * f_weights3;

%% Upsample finger predictions 

up1_finger = [];
up2_finger = [];
up3_finger = [];

for i = 1:5
    up1_finger(:, i) = spline(1:size(binlab1, 1), binlab1(:, i), 1:1/50:size(binlab1, 1)); %off by 1 problem?? should be 1/50
    up2_finger(:, i) = spline(1:size(binlab2, 1), binlab2(:, i), 1:1/50:size(binlab2, 1));
    up3_finger(:, i) = spline(1:size(binlab3, 1), binlab3(:, i), 1:1/50:size(binlab3, 1));
end 

up1_finger = [zeros(150, 5); up1_finger(1:299850, :)];   % pad equivalent of 2 windows in the beginning
up2_finger = [zeros(150, 5); up2_finger(1:299850, :)];
up3_finger = [zeros(150, 5); up3_finger(1:299850, :)];

%% Normalize binary prediction

for i = 1:5
    up1_finger(:,i) = up1_finger(:,i) / max(up1_finger(:,i));
    up2_finger(:,i) = up2_finger(:,i) / max(up2_finger(:,i));
    up3_finger(:,i) = up3_finger(:,i) / max(up3_finger(:,i));
end 

%% For all the places where it's predicted to be off, multiply the probabilities by the linreg result 

where_above = up1_finger >= .5; % where on
filteredfinger1 = up1 .* (~where_above .* up1_finger) + up1 .* (where_above); %up1 .* (up1_finger/max(max(up1_finger))); 

where_above = up1_finger >= .5; % where on
filteredfinger2 = up2 .* (~where_above .* up2_finger) + up2 .* where_above;

where_above = up1_finger >= .5; % where on
filteredfinger3 = up3 .* (~where_above .* up3_finger) + up3 .* where_above;

%%
figure()
plot(filteredfinger1(:,3));
hold on;
plot(glove1(:,3));

%% Correlation - resub

corr1 = zeros(1, 5);
corr2 = zeros(1, 5);
corr3 = zeros(1, 5);
for i = 1:5             % iterate over fingers
    corr1(i) = corr(glove1(:, i), filteredfinger1(:,i));
    corr2(i) = corr(glove2(:, i), filteredfinger2(:,i));
    corr3(i) = corr(glove3(:, i), filteredfinger3(:,i));
end

avgcorr1 = mean(corr1)
avgcorr2 = mean(corr2)
avgcorr3 = mean(corr3)

