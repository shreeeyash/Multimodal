%% This code is the Self-Organising Fuzzy Logic (SOF) classifier
clear all
clc
close all
%% Example 1
% load example1.mat
Dtrain = load("IEMOCAP_TRAIN_DATA.mat");
Dtrain = Dtrain.val;
Ltrain = load("IEMOCAP_TRAIN_LABEL.mat");
Ltrain = Ltrain.val;
Dtest = load("IEMOCAP_TEST_DATA.mat");
Dtest = Dtest.val;
Ltest = load("IEMOCAP_TEST_LABEL.mat");
Ltest = Ltest.val;
%% The SOF classifier conducts offline learning from static datasets
Input.TrainingData=Dtrain;    % Input data samples
Input.TrainingLabel=Ltrain;   % Labels of the input data samples
GranLevel=33;                % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
DistanceType='Euclidean';  % Type of distance/dissimilarity SOF classifier uses, which can be 'Mahalanobis', 'Cosine' or 'Euclidean'
Mode='OfflineTraining';      % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output1]=SOFClassifier(Input,GranLevel,Mode,DistanceType); 
% Output1.TrainedClassifier  - Offline primed SOF classifier
%% The SOF classifier conducts validation on testing data
Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtest;     % Testing 
Input.TestingLabel=Ltest;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
cm = Output2.ConfusionMatrix;
save("cm_iemocap_sdk_33.mat",'cm');
[cm,a,p,r,f] = getcm(Ltest,Output2.EstimatedLabel,1:2);
disp("Accuracy:");
disp(100*a/1807);
disp("F score:")
disp(mean(f));
disp("Precision:")
disp(mean(p));
disp("Recall:");
disp(mean(r));
disp(cm);
% Output2.TrainedClassifier  - Trained SOF classifier (same as the input)
% Output.EstimatedLabel      - Estimated label of validation data
% Output.ConfusionMatrix     - onfusion matrix of the result



