%% Load subject data 
% According to guide/recitation 

session = IEEGSession('I521_Sub1_Training_ecog', 'bkarpowicz3', 'bka_ieeglogin.bin');
ecog1 = session.data(1).getvalues(1:300000, 1:62);
session = IEEGSession('I521_Sub1_Training_dg', 'bkarpowicz3', 'bka_ieeglogin.bin');
glove1 = session.data(1).getvalues(1:147500, 1:5);
session = IEEGSession('I521_Sub1_Leaderboard_ecog', 'bkarpowicz3', 'bka_ieeglogin.bin');
test1 = session.data(1).getvalues(1:300000, 1:62);

sR = session.data.sampleRate;

session = IEEGSession('I521_Sub2_Training_ecog', 'bkarpowicz3', 'bka_ieeglogin.bin');
ecog2 = session.data(1).getvalues(1:300000, 1:62);
session = IEEGSession('I521_Sub2_Training_dg', 'bkarpowicz3', 'bka_ieeglogin.bin');
glove2 = session.data(1).getvalues(1:147500, 1:5);
session = IEEGSession('I521_Sub2_Leaderboard_ecog', 'bkarpowicz3', 'bka_ieeglogin.bin');
test2 = session.data(1).getvalues(1:300000, 1:62);

session = IEEGSession('I521_Sub3_Training_ecog', 'bkarpowicz3', 'bka_ieeglogin.bin');
ecog3 = session.data(1).getvalues(1:300000, 1:62);
session = IEEGSession('I521_Sub3_Training_dg', 'bkarpowicz3', 'bka_ieeglogin.bin');
glove3 = session.data(1).getvalues(1:147500, 1:5);
session = IEEGSession('I521_Sub3_Leaderboard_ecog', 'bkarpowicz3', 'bka_ieeglogin.bin');
test3 = session.data(1).getvalues(1:300000, 1:62);

%% Extract Features 

feat1 = extractFeatures_v1(ecog1, sR);
feat2 = extractFeatures_v1(ecog2, sR);
feat3 = extractFeatures_v1(ecog3, sR);

save('features.mat', 'feat1', 'feat2', 'feat3');

%% Downsample glove data 
% Need to bring samples down to every 50ms to align with features.
% ----- Not sure we need this anymore if we use 
% ----- winLen = 80 ms and winDisp = 40 ms so that the features line up
% ----- exactly with the data glove time points
glove1_down = [];
glove2_down = [];
glove3_down = [];
for i = 1:5
    glove1_down(:,end+1) = decimate(glove1(:,i), 50);
    glove2_down(:,end+1) = decimate(glove2(:,i), 50);
    glove3_down(:,end+1) = decimate(glove3(:,i), 50);
end 

%% Linear Regression 

Y1 = linreg(feat1, glove1_down);
Y2 = linreg(feat2, glove2_down);
Y3 = linreg(feat3, glove3_down);

%% Cubic Interpolation of Results 
% Bring data from every 50ms back to 1000 Hz. 
% ----- if we use winLen = 80 ms and winDisp = 40 ms, also do not need this
up1 = [];
up2 = [];
up3 = [];

for i = 1:5
    up1(:,i) = spline(1:2950, Y1(:,i), 1:1/51:2950); %off by 1 problem?? should be 1/50
    up2(:,i) = spline(1:2950, Y2(:,i), 1:1/51:2950);
    up3(:,i) = spline(1:2950, Y3(:,i), 1:1/51:2950);
end 

%% Visualize prediction of train data 

figure();
plot(up1(:,1));
hold on;
plot(glove1(:,1)); 

% this looks quite bad.

%% Save 

predicted_dg = cell{1,3};
predicted_dg{1} = up1(1:147500);
predicted_dg{2} = up2(1:147500);
predicted_dg{3} = up3(1:147500);

save('checkpoint1.mat', 'predicted_dg');

