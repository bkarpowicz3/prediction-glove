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

%% Extract Features 

numFeats = 6;
feat1 = extractFeatures(ecog1, sR, numFeats);
feat2 = extractFeatures(ecog2, sR, numFeats);
feat3 = extractFeatures(ecog3, sR, numFeats);

save('features.mat', 'feat1', 'feat2', 'feat3');

%%
% subject 1 - 55
% subject 2 - 21 & 38

% feat1 = [feat1(:, 1:329) feat1(:, 336:end)];
% feat2 = [feat2(:, 1:125) feat2(:, 132:227) feat2(:, 234:end)];

%% Filter Glove data with Lowpass

fc = 3;    % cutoff frequency
[b,a] = butter(6,fc/(sR/2));
for i = 1:5
    glove1(:, i) = filtfilt(b, a, glove1(:, i));
    glove2(:, i) = filtfilt(b, a, glove2(:, i));
    glove3(:, i) = filtfilt(b, a, glove3(:, i));
end

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

<<<<<<< HEAD
%% Normalize features (not helpful for linreg)

for i = 1:size(feat1, 2)
    feat1(:, i) = (feat1(:, i) - mean(feat1(:, i)))./ std(feat1(:, i));
end

for i = 1:size(feat2, 2)
    feat2(:, i) = (feat2(:, i) - mean(feat2(:, i)))./ std(feat2(:, i));
end

for i = 1:size(feat3, 2)
    feat3(:, i) = (feat3(:, i) - mean(feat3(:, i)))./ std(feat3(:, i));
end
=======
%% Get binary labels for glove data 

thresh = 1;
glove1_bin = glove1_down >= thresh;
thresh = 0.8;
glove2_bin = glove2_down >= thresh;
thresh = 0.6;
glove3_bin = glove3_down >= thresh;
>>>>>>> e1cc62c421eacca0684856cac781fb0e5e73f929

%% Linear Regression 
numFeats = 9;       % CHANGE

Y1 = linreg(feat1, glove1_down, feat1, numFeats);
Y2 = linreg(feat2, glove2_down, feat2, numFeats);
Y3 = linreg(feat3, glove3_down, feat3, numFeats);

%% Alternative iterative regression (use prerun weights and multiply by features (R)
finger = 1;

features = makeR(feat1(1:4000, :), 9);
features = features(:, 2:end);
targets = [glove1_down(N:4000, finger); glove1_down(1:N-1, finger)];
[Y1, weights, cost_history] = linregiter(features, targets, zeros(size(features, 2)+1, 1), .01, 10000);

%% Make train R matrix & normalize
numFeats = 9;
R1 = makeR(feat1, numFeats);
R2 = makeR(feat2, numFeats);
R3 = makeR(feat3, numFeats);

features = R1;
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
R1 = features;

features = R2;
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
R2 = features;

features = R3;
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
R3 = features;

%% Calculate predicted training using lin & log regression
Y1 = R1*weights1;
Y2 = R2*weights2;
Y3 = R3*weights3;

% Y1l = R1*logsig(f_weights1);
% Y2l = R2*logsig(f_weights2);
% Y3l = R3*logsig(f_weights3);

Y1l = logsig(R1*f_weights1);
Y2l = logsig(R2*f_weights2);
Y3l = logsig(R3*f_weights3);

Y1 = Y1.*(Y1l.^0.31);
Y2 = Y2.*(Y2l.^0.67);
Y3 = Y3.*(Y3l.^0.22);

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

% up1 = [zeros(150, 5); up1; zeros(99, 5)];   % pad equivalent of 2 windows in the beginning
% up2 = [zeros(150, 5); up2; zeros(99, 5)];
% up3 = [zeros(150, 5); up3; zeros(99, 5)];

up1 = [zeros(150, 5); up1(1:299850, :)];   % pad equivalent of 2 windows in the beginning
up2 = [zeros(150, 5); up2(1:299850, :)];
up3 = [zeros(150, 5); up3(1:299850, :)];

%% Postprocess with low pass

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

%% Avg Training Correlation

corr1 = zeros(1, 5);
corr2 = zeros(1, 5);
corr3 = zeros(1, 5);
for finger = 1:5
    corr1(finger) = corr(glove1(:, finger), up1(:, finger));
    corr2(finger) = corr(glove2(:, finger), up2(:, finger));
    corr3(finger) = corr(glove3(:, finger), up3(:, finger));
end

mcorr1 = mean(corr1);
mcorr2 = mean(corr2);
mcorr3 = mean(corr3);
disp([num2str(mcorr1), ' ', num2str(mcorr2), ' ', num2str(mcorr3)])
totalcorr = [corr1([1, 2, 3, 5]), corr2([1, 2, 3, 5]), corr3([1, 2, 3, 5])];
avgcorr = mean(totalcorr)

%% Predict binary state of glove using features

resub_bin1 = zeros(5999, 5);
resub_bin2 = zeros(5999, 5);
resub_bin3 = zeros(5999, 5);

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
    [resubs, ~, ~] = predict(model1, feat1);
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
    [resubs, ~, ~] = predict(model2, feat2);
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
    [resubs, ~, ~] = predict(model3, feat3);
    resub_bin3(:,k) = resubs;
end 

%% Upsample Binary Predictions 

resub1 = zeros(300000, 5);
resub2 = zeros(300000, 5);
resub3 = zeros(300000, 5);

for i = 1:5 
    up = spline(1:size(resub_bin1, 1), resub_bin1(:,i), 1:1/50:size(resub_bin1, 1)); 
    up = [zeros(150, 1); up'; zeros(99, 1)];
    resub1(:,i) = up(1:300000);
    
    up = spline(1:size(resub_bin2, 1), resub_bin2(:,i), 1:1/50:size(resub_bin2, 1)); 
    up = [zeros(150, 1); up'; zeros(99, 1)];
    resub2(:,i) = up(1:300000);
    
    up = spline(1:size(resub_bin3, 1), resub_bin3(:,i), 1:1/50:size(resub_bin3, 1)); 
    up = [zeros(150, 1); up'; zeros(99, 1)];
    resub3(:,i) = up(1:300000);
end 

%% Calculate correlation

corr1 = zeros(1, 5);
corr2 = zeros(1, 5);
corr3 = zeros(1, 5);
for i = 1:5             % iterate over fingers
    corr1(i) = corr(glove1(:, i), up1(:, i).*resub1(:,i));
    corr2(i) = corr(glove2(:, i), up2(:, i).*resub2(:,i));
    corr3(i) = corr(glove3(:, i), up3(:, i).*resub3(:,i));
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
for i = 1:3%length(folds)     % fold that is testing set
    disp(['Running Fold ' num2str(i)]);
    
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
    
    % make binary labels by thresholding 
    thresh = 1;
    fingers1_bin = fingers1 >= thresh;
    thresh = 0.8;
    fingers2_bin = fingers2 >= thresh;
    thresh = 0.6;
    fingers3_bin = fingers3 >= thresh;

    % train model
    Y1 = linreg(trainfold1, fingers1, feat1(folds{i}, :), numFeats);
    Y2 = linreg(trainfold2, fingers2, feat2(folds{i}, :), numFeats);
    Y3 = linreg(trainfold3, fingers3, feat3(folds{i}, :), numFeats);

    up1 = [];
    up2 = [];
    up3 = [];
    
    for l = 1:5
        up1(:, l) = spline(1:size(Y1, 1), Y1(:, l), 1:1/50:size(Y1, 1)); %off by 1 problem?? should be 1/50
        up2(:, l) = spline(1:size(Y2, 1), Y2(:, l), 1:1/50:size(Y2, 1));
        up3(:, l) = spline(1:size(Y3, 1), Y3(:, l), 1:1/50:size(Y3, 1));
    end
    
%     up1 = [zeros(200, 5); up1(1:29800, :)];   % pad equivalent of 2 windows in the beginning
%     up2 = [zeros(200, 5); up2(1:29800, :)];
%     up3 = [zeros(200, 5); up3(1:29800, :)];
    
    up1 = [zeros(150, 5); up1; zeros(99, 5)];   % pad equivalent of 2 windows in the beginning
    up2 = [zeros(150, 5); up2; zeros(99, 5)];
    up3 = [zeros(150, 5); up3; zeros(99, 5)];
    
%     for n = 1:5
%         up1(:, n) = filtfilt(b2, a2, up1(:, n));
%         up2(:, n) = filtfilt(b2, a2, up2(:, n));
%         up3(:, n) = filtfilt(b2, a2, up3(:, n));
%     end
    
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
    
    up1 = zeros(30000, 5);
    up2 = zeros(30000, 5);
    up3 = zeros(30000, 5);
    
    for m = 1:5
        %     winLen = 800e-3;
        %     winDisp = 400e-3;
        smoothed_glove1 = MovingWinFeats(below_thresh1(:,m), sR, winLen, winDisp, M);
        smoothed_glove1(end+1:end+2) = [0 0];
        smoothed_glove1_ = spline(1:length(smoothed_glove1), smoothed_glove1, 1:1/400:length(smoothed_glove1));  %zoInterp(smoothed_glove, 100);
        up1(:,m) = above_thresh1(:,m) + smoothed_glove1_(1:end-1)';
        
        %     winLen = 1000e-3;
        %     winDisp = 500e-3;
        smoothed_glove2 = MovingWinFeats(below_thresh2(:,m), sR, winLen, winDisp, M);
        smoothed_glove2(end+1:end+2) = [0 0];
        smoothed_glove2_ = spline(1:length(smoothed_glove2), smoothed_glove2, 1:1/400:length(smoothed_glove2));  %zoInterp(smoothed_glove, 100);
        up2(:,m) = above_thresh2(:,m) + smoothed_glove2_(1:end-1)';
        
        %     winLen = 500e-3;
        %     winDisp = 250e-3;
        smoothed_glove3 = MovingWinFeats(below_thresh3(:,m), sR, winLen, winDisp, M);
        smoothed_glove3(end+1:end+2) = [0 0];
        smoothed_glove3_ = spline(1:length(smoothed_glove3), smoothed_glove3, 1:1/400:length(smoothed_glove3));  %zoInterp(smoothed_glove, 100);
        up3(:,m) = above_thresh3(:,m) + smoothed_glove3_(1:end-1)';
    end
    
    % get binary predictions
    resub_bin1 = [];
    resub_bin2 = [];
    resub_bin3 = [];
    
    for k = 1:5
%         model1 = fitcknn(trainfold1, fingers1_bin(:,k), ...
%             'Distance', 'Euclidean', ...
%             'Exponent', [], ...
%             'NumNeighbors', 3, ...
%             'DistanceWeight', 'Equal', ...
%             'Standardize', true, ...
%             'ClassNames', [0; 1]);
%         model1 = fitctree(trainfold1, fingers1_bin(:,k));
        model1 = TreeBagger(30, trainfold1, fingers1_bin(:,k), 'Method', 'regression', 'OOBPrediction', 'on');
        [resubs, ~, ~] = predict(model1, feat1(folds{i}, :));
        resub_bin1(:,k) = resubs;

%         model2 = fitcknn(trainfold2, fingers2_bin(:,k), ...
%             'Distance', 'Euclidean', ...
%             'Exponent', [], ...
%             'NumNeighbors', 3, ...
%             'DistanceWeight', 'Equal', ...
%             'Standardize', true, ...
%             'ClassNames', [0; 1]);
%         model2 = fitctree(trainfold2, fingers2_bin(:,k));
        model2 = TreeBagger(30, trainfold2, fingers2_bin(:,k), 'Method', 'regression', 'OOBPrediction', 'on');
        [resubs, ~, ~] = predict(model2, feat2(folds{i}, :));
        resub_bin2(:,k) = resubs;

%         model3 = fitcknn(trainfold3, fingers3_bin(:,k), ...
%             'Distance', 'Euclidean', ...
%             'Exponent', [], ...
%             'NumNeighbors', 3, ...
%             'DistanceWeight', 'Equal', ...
%             'Standardize', true, ...
%             'ClassNames', [0; 1]);
%         model3 = fitctree(trainfold3, fingers3_bin(:,k));
        model3 = TreeBagger(30, trainfold3, fingers3_bin(:,k), 'Method', 'regression', 'OOBPrediction', 'on');
        [resubs, ~, ~] = predict(model3, feat3(folds{i}, :));
        resub_bin3(:,k) = resubs;
    end 
    
    % upsample binary predictions 
    resub1 = [];
    resub2 = [];
    resub3 = [];

    for s = 1:5 
        up = spline(1:size(resub_bin1, 1), resub_bin1(:,s), 1:1/50:size(resub_bin1, 1)); 
        up = [zeros(150, 1); up'; zeros(99, 1)];
        resub1(:,s) = up(1:30000);

        up = spline(1:size(resub_bin2, 1), resub_bin2(:,s), 1:1/50:size(resub_bin2, 1)); 
        up = [zeros(150, 1); up'; zeros(99, 1)];
        resub2(:,s) = up(1:30000);

        up = spline(1:size(resub_bin3, 1), resub_bin3(:,s), 1:1/50:size(resub_bin3, 1)); 
        up = [zeros(150, 1); up'; zeros(99, 1)];
        resub3(:,s) = up(1:30000);
    end 
    
    testlabel1 = glove1(foldsfull{i}, :);
    testlabel2 = glove2(foldsfull{i}, :);
    testlabel3 = glove3(foldsfull{i}, :);

%     for k = 1:5
%         crosscorr1(i, k) = corr(testlabel1(:, k), up1(:, k));
%         crosscorr2(i, k) = corr(testlabel2(:, k), up2(:, k));
%         crosscorr3(i, k) = corr(testlabel3(:, k), up3(:, k));
%     end
    
    for q = 1:5
        crosscorr1(i, q) = corr(testlabel1(:, q), up1(1:30000, q).*resub1(:,q));
        crosscorr2(i, q) = corr(testlabel2(:, q), up2(1:30000, q).*resub2(:,q));
        crosscorr3(i, q) = corr(testlabel3(:, q), up3(1:30000, q).*resub3(:,q));
    end
end

avgcorr1 = mean(crosscorr1)
avgcorr2 = mean(crosscorr2)
avgcorr3 = mean(crosscorr3)

totalcorr = [avgcorr1([1, 2, 3, 5]), avgcorr2([1, 2, 3, 5]), avgcorr3([1, 2, 3, 5])];
avgcorr = mean(totalcorr)

%% BELOW IS ALL LOG REG STUFF - NOT BEING USED RIGHT NOW 
%  instead I built in the knn binary predictor 

%% Logistic Regression: Create Labels

% Threshold at 1.4 
threshold = 1.4;
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

figure % change threshold for glove 3 to be at 0.5
plot(glove3(:,3))
hold on
plot(biglove{1, 3}(:,3))

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

%% Train logistic classifier

log1 = mnrfit(feat1, biglove1_down, 'Interactions', 'off');
prob1 = mnrval(log1, feat1);
log2 = mnrfit(feat1, biglove1_down, 'Interactions', 'off');
prob2 = mnrval(log2, feat2);
log3 = mnrfit(feat1, biglove1_down, 'Interactions', 'off');
prob3 = mnrval(log3, feat3);

%% Combine logistic with linear regression