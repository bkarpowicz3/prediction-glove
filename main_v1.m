%% Load subject data 
% According to guide/recitation 

userID = 'renyueqi';
userlog = 'ren_ieeglogin.bin';

session = IEEGSession('I521_Sub1_Training_ecog', userID, userlog);
ecog1 = session.data(1).getvalues(1:300000, 1:62);
session = IEEGSession('I521_Sub1_Training_dg', userID, userlog);
glove1 = session.data(1).getvalues(1:147500, 1:5);
session = IEEGSession('I521_Sub1_Leaderboard_ecog', userID, userlog);
test1 = session.data(1).getvalues(1:300000, 1:62);

sR = session.data.sampleRate;

% session = IEEGSession('I521_Sub2_Training_ecog', userID, userlog);
% % Here's where I ran into this error (calling next line):
% %   Index exceeds the number of array elements (48).
% %   Error in IEEGDataset/getvalues (line 441)
% %               allReqSampleRate = allSampleRates(chIdx);
ecog2 = session.data(1).getvalues(1:300000, 1:62); 
session = IEEGSession('I521_Sub2_Training_dg', userID, userlog);
glove2 = session.data(1).getvalues(1:147500, 1:5);
session = IEEGSession('I521_Sub2_Leaderboard_ecog', userID, userlog);
test2 = session.data(1).getvalues(1:300000, 1:62);

session = IEEGSession('I521_Sub3_Training_ecog', userID, userlog);
ecog3 = session.data(1).getvalues(1:300000, 1:62);
session = IEEGSession('I521_Sub3_Training_dg', userID, userlog);
glove3 = session.data(1).getvalues(1:147500, 1:5);
session = IEEGSession('I521_Sub3_Leaderboard_ecog', userID, userlog);
test3 = session.data(1).getvalues(1:300000, 1:62);

%% Extract Features 

feat1 = extractFeatures_v1(ecog1, sR);
feat2 = extractFeatures_v1(ecog2, sR);
feat3 = extractFeatures_v1(ecog3, sR);

%save('features.mat', 'feat1');
save('features.mat', 'feat1', 'feat2', 'feat3');

%% Downsample glove data 
% Need to bring samples down to every 40ms to align with features.
factor = 40; % sR (Hz) * winDisp (s) 
glove1_down = [];
glove2_down = [];
glove3_down = [];
for i = 1:5
    glove1_down(:,end+1) = decimate(glove1(:,i), factor); % Does not line up with features length (# time windows)
    glove2_down(:,end+1) = decimate(glove2(:,i), factor);
    glove3_down(:,end+1) = decimate(glove3(:,i), factor);
end 

%% Linear Regression 

Y1 = linreg(feat1, glove1_down, size(feat1, 1));
Y2 = linreg(feat2, glove2_down, size(feat2, 1));
Y3 = linreg(feat3, glove3_down, size(feat3, 1));

%% Cubic Interpolation of Results 
% Bring data from every 40ms back to 1000 Hz. 
up1 = [];
up2 = [];
up3 = [];
resized = length(glove1)/(factor*1e-3*sR);

for i = 1:5
    up1(:,i) = spline(1:resized, Y1(:,i), 1:1/factor:resized); %off by 1 problem?? should be 1/50
    up2(:,i) = spline(1:resized, Y2(:,i), 1:1/factor:resized);
    up3(:,i) = spline(1:resized, Y3(:,i), 1:1/factor:resized);
end 

%% Visualize prediction of train data 

figure();
plot(up1(:,1));
hold on;
plot(glove1(:,1)); 


%% Save 

predicted_dg = cell{1,3};
predicted_dg{1} = up1(1:147500);
predicted_dg{2} = up2(1:147500);
predicted_dg{3} = up3(1:147500);

save('checkpoint1.mat', 'predicted_dg');

