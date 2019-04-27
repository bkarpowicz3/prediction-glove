%% Load data from IEEG
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

%% Load Features 

load('features_9.mat'); %gives feat1, feat2, feat3
testfeat = load('testfeatures.mat');

%% Downsample Glove Data

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

%% Bagged Tree Ensemble 

mdl = TreeBagger(100, feat1, glove1_down(:,1), 'Method', 'regression', 'OOBPrediction', 'on');
figure;
oobErrorBaggedEnsemble = oobError(mdl);
plot(oobErrorBaggedEnsemble);
xlabel 'Number of grown trees';
ylabel 'Out of bag classification error';
title 'Error with diff num trees';

labels = predict(mdl, feat1);
corr(labels, glove1_down(:,1))

%% Produce crossval inds 

num = 10;
folds = cell(num,1);

allinds = randperm(5999);
c = 1;
breaks = (0:num)*600;
breaks(end) = breaks(end) -1;

for i = 1:length(breaks)-1
    folds{c} = allinds(breaks(i)+1:breaks(i+1));
    c = c + 1;
end 

length(unique([folds{:}]))

%% Manual Cross Validation 

accs = zeros(1, num);
for i = 1:num
    testinds = folds{i};
    traininds = [folds{setdiff(1:num, i)}];
    
    testData = feat1(testinds, :);
    trainData = feat1(traininds, :);
    testLabels = glove1_down(testinds, 3);
    trainLabels = glove1_down(traininds, 3);
    
    mdl = TreeBagger(30, trainData, trainLabels, 'Method', 'regression', ...
        'OOBPrediction', 'on', 'PredictorSelection', 'curvature');
    preds = predict(mdl, testData);
    
    acc = corr(testLabels, preds);
    accs(i) = acc;
end 

mean(accs)

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
    Y1 = TreeBagger(30, trainfold1, fingers1, 'Method', 'regression', ...
        'OOBPrediction', 'on', 'PredictorSelection', 'curvature');
    Y2 = TreeBagger(30, trainfold2, fingers2, 'Method', 'regression', ...
        'OOBPrediction', 'on', 'PredictorSelection', 'curvature');
    Y3 = TreeBagger(30, trainfold3, fingers3, 'Method', 'regression', ...
        'OOBPrediction', 'on', 'PredictorSelection', 'curvature');
    
    up1 = [];
    up2 = [];
    up3 = [];
    
    for l = 1:5
        up1(:, l) = spline(1:size(Y1, 1), Y1(:, l), 1:1/50:size(Y1, 1)); %off by 1 problem?? should be 1/50
        up2(:, l) = spline(1:size(Y2, 1), Y2(:, l), 1:1/50:size(Y2, 1));
        up3(:, l) = spline(1:size(Y3, 1), Y3(:, l), 1:1/50:size(Y3, 1));
    end
    
    up1 = [zeros(50, 5); up1; zeros(49, 5)];   % pad equivalent of 2 windows in the beginning
    up2 = [zeros(50, 5); up2; zeros(49, 5)];
    up3 = [zeros(50, 5); up3; zeros(49, 5)];
    
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

%% On Test Data 

patient1 = zeros(147500, 5);
patient2 = zeros(147500, 5);
patient3 = zeros(147500, 5);

for i = 1:5
    mdl1 = TreeBagger(30, feat1, glove1_down(:,i), 'Method', 'regression', 'OOBPrediction', 'on',...
        'PredictorSelection', 'curvature');
    preds1 = predict(mdl1, testfeat.testfeat1);
    preds1_= spline(1:size(preds1, 1), preds1, 1:1/50:size(preds1, 1)); 
    patient1(:,i) = [zeros(50, 1);  preds1_'; zeros(49, 1)];

    mdl2 = TreeBagger(30, feat2, glove2_down(:,i), 'Method', 'regression', 'OOBPrediction', 'on',...
        'PredictorSelection', 'curvature');
    preds2 = predict(mdl2, testfeat.testfeat2);
    preds2_= spline(1:size(preds2, 1), preds2, 1:1/50:size(preds2, 1)); 
    patient2(:,i) = [zeros(50, 1);  preds2_'; zeros(49, 1)];

    mdl3 = TreeBagger(30, feat3, glove3_down(:,i), 'Method', 'regression', 'OOBPrediction', 'on',...
        'PredictorSelection', 'curvature');
    preds3 = predict(mdl3, testfeat.testfeat3);
    preds3_= spline(1:size(preds3, 1), preds3, 1:1/50:size(preds3, 1)); 
    patient3(:,i) = [zeros(50, 1);  preds3_'; zeros(49, 1)];
end 

%% Visualize and save 

figure() 
plot(glove1(:,1));
hold on;
plot(patient1(:,1));

%% 

predicted_dg = cell(3, 1);
predicted_dg{1} = testup1(1:147500, 1:5);
predicted_dg{2} = testup2(1:147500, 1:5);
predicted_dg{3} = testup3(1:147500, 1:5);

save('checkpoint1.mat', 'predicted_dg');

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
    Y1 = zeros(599, 5);
    Y2 = zeros(599, 5);
    Y3 = zeros(599, 5);
    for m = 1:5
        model1 = TreeBagger(30, trainfold1, fingers1(:, m), 'Method', 'regression', ...
            'OOBPrediction', 'on', 'PredictorSelection', 'curvature');
        Y1(:, m) = predict(model1, feat1(folds{i}, :));
        model2 = TreeBagger(30, trainfold2, fingers2(:, m), 'Method', 'regression', ...
            'OOBPrediction', 'on', 'PredictorSelection', 'curvature');
        Y2(:, m) = predict(model2, feat2(folds{i}, :));
        model3 = TreeBagger(30, trainfold3, fingers3(:, m), 'Method', 'regression', ...
            'OOBPrediction', 'on', 'PredictorSelection', 'curvature');
        Y3(:, m) = predict(model3, feat3(folds{i}, :));
    end
    up1 = [];
    up2 = [];
    up3 = [];
    
    for l = 1:5
        up1(:, l) = spline(1:size(Y1, 1), Y1(:, l), 1:1/50:size(Y1, 1)); %off by 1 problem?? should be 1/50
        up2(:, l) = spline(1:size(Y2, 1), Y2(:, l), 1:1/50:size(Y2, 1));
        up3(:, l) = spline(1:size(Y3, 1), Y3(:, l), 1:1/50:size(Y3, 1));
    end
    
    up1 = [zeros(50, 5); up1; zeros(49, 5)];   % pad equivalent of 2 windows in the beginning
    up2 = [zeros(50, 5); up2; zeros(49, 5)];
    up3 = [zeros(50, 5); up3; zeros(49, 5)];
    
    testlabel1 = glove1(foldsfull{i}, :);
    testlabel2 = glove2(foldsfull{i}, :);
    testlabel3 = glove3(foldsfull{i}, :);
    for k = 1:5
        crosscorr1(i, k) = corr(testlabel1(:, k), up1(:, k));
        crosscorr2(i, k) = corr(testlabel2(:, k), up2(:, k));
        crosscorr3(i, k) = corr(testlabel3(:, k), up3(:, k));
    end
    
    disp(['fold' num2str(i)])
    disp(crosscorr1(i, :))
    disp(crosscorr2(i, :))
    disp(crosscorr3(i, :))
end

avgcorr1 = mean(crosscorr1)
avgcorr2 = mean(crosscorr2)
avgcorr3 = mean(crosscorr3)

totalcorr = [avgcorr1([1, 2, 3, 5]), avgcorr2([1, 2, 3, 5]), avgcorr3([1, 2, 3, 5])];
avgcorr = mean(totalcorr)