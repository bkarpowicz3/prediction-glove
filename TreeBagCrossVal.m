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