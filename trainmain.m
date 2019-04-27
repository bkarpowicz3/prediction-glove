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

feat1 = extractFeatures_v1(ecog1, sR);
feat2 = extractFeatures_v1(ecog2, sR);
feat3 = extractFeatures_v1(ecog3, sR);

save('features.mat', 'feat1', 'feat2', 'feat3');

%%
% subject 1 - 55
% subject 2 - 21 & 38

feat1 = [feat1(:, 1:329) feat1(:, 336:end)];
feat2 = [feat2(:, 1:125) feat2(:, 132:227) feat2(:, 234:end)];

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
numFeats = 6;

Y1 = linreg(feat1, glove1_down, feat1, numFeats);
Y2 = linreg(feat2, glove2_down, feat2, numFeats);
Y3 = linreg(feat3, glove3_down, feat3, numFeats);

%% lasso test

R1 = makeR(feat1);
R2 = makeR(feat2);
R3 = makeR(feat3);
Y1 = zeros(size(R1, 1), 5);
Y2 = zeros(size(R2, 1), 5);
Y3 = zeros(size(R3, 1), 5);
for i = 1:5
    Y1(:, i) = lassoReg(R1, glove1_down(:, i), R1);
    disp(['subject 1 finger ' num2str(i) ' done'])
    
    Y2(:, i) = lassoReg(R2, glove2_down(:, i), R2);
    disp(['subject 1 finger ' num2str(i) ' done'])
    
    Y3(:, i) = lassoReg(R3, glove3_down(:, i), R3);
    disp(['subject 1 finger ' num2str(i) ' done'])
end

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

up1 = [zeros(150, 5); up1; zeros(99, 5)];   % pad equivalent of 2 windows in the beginning
up2 = [zeros(150, 5); up2; zeros(99, 5)];
up3 = [zeros(150, 5); up3; zeros(99, 5)];

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
for i = 1:length(folds)     % fold that is testing set
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
    
%     Y1 = linreg_new(trainfold1, fingers1, feat1(folds{i}, :));
%     Y2 = linreg_new(trainfold2, fingers2, feat2(folds{i}, :));
%     Y3 = linreg_new(trainfold3, fingers3, feat3(folds{i}, :));
    
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