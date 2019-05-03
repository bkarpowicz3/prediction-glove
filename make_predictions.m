function [predicted_dg] = make_predictions(test_ecog)

% Inputs: test_ecog - 3 x 1 cell array containing ECoG for each subject, where test_ecog{i} 
% to the ECoG for subject i. Each cell element contains a N x M testing ECoG,
% where N is the number of samples and M is the number of EEG channels.
% Outputs: predicted_dg - 3 x 1 cell array, where predicted_dg{i} contains the 
% data_glove prediction for subject i, which is an N x 5 matrix (for
% fingers 1:5)

% Run time: The script has to run less than 1 hour. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load weights from linear regression and logistic regression.

load linweights10.mat
load logweights10.mat

% Predict using linear predictor in combination with logistic for each subject

% create cell array with one element for each subject
predicted_dg = cell(3,1);

% number of features to extract 
numFeats = 10;
sR = 1000;

% for each subject
for subj = 1:3 
    
    % get the testing ecog
    testset = test_ecog{subj}; 
    
    % feature extraction 
    feat = extractFeatures(testset, sR, numFeats);
    
    % make R matrix 
    R = makeR(feat, numFeats);
    % normalize the R matrix per feature
    features = R;
    for i = 2:size(features, 2) % iterate over column
        fmean = mean(features(:, i));
        frange = max(features(:, i)) - min(features(:, i));
        features(:, i) = features(:, i) - fmean;
        features(:, i) = features(:, i) / frange;
    end
    R = features;
        
    % predict dg based on ECOG for each finger
    Y = R*linweights{subj};
    Yl = logsig(R * logweights{subj});
    yhat = Y.*(Yl.^0.1);
       
    % upsample yhat 
    numNeeded = size(testset, 1) - 150;
    up = zeros(size(testset,1),5);
    for i = 1:5
        x = spline(1:size(yhat, 1), yhat(:, i), 1:1/50:size(yhat, 1));
        up(150:150+numNeeded, i) = x(1:numNeeded+1);
    end 
    
    predicted_dg{subj} = up;
    
end
