clear all
clc
close all

Dtrain_video = load("Dtrain_happy_video.mat");
Dtrain_video = Dtrain_video.val;
Ltrain = load("Ltrain_happy.mat");
Ltrain = Ltrain.val;
Dtest_video = load("Dtest_happy_video.mat");
Dtest_video = Dtest_video.val;
Ltest = load("Ltest_happy.mat");
Ltest = Ltest.val;
Dtrain_audio = load("Dtrain_happy_audio.mat");
Dtrain_audio = Dtrain_audio.val;
Dtest_audio = load("Dtest_happy_audio.mat");
Dtest_audio = Dtest_audio.val;
Dtrain_text = load("Dtrain_happy_text.mat");
Dtrain_text = Dtrain_text.val;
Dtest_text = load("Dtest_happy_text.mat");
Dtest_text = Dtest_text.val;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% VIDEO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Input.TrainingData=Dtrain_video;    % Input data samples
Input.TrainingLabel=Ltrain;   % Labels of the input data samples
GranLevel=12;                % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
DistanceType='Euclidean';  % Type of distance/dissimilarity SOF classifier uses, which can be 'Mahalanobis', 'Cosine' or 'Euclidean'
Mode='OfflineTraining';      % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output1]=SOFClassifier(Input,GranLevel,Mode,DistanceType); 
% Output1.TrainedClassifier  - Offline primed SOF classifier
%% The SOF classifier conducts validation on testing data
Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtest_video;     % Testing 
Input.TestingLabel=Ltest;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
cm = Output2.ConfusionMatrix;
v_te = Output2.EstimatedLabel;
save("cm/video_happy_12.mat",'cm');
[cm,a,p,r,f] = getcm(Ltest,Output2.EstimatedLabel,1:2);
disp("video Accuracy:");
disp(100*a/4832);
disp("video F score:")
disp(mean(f));
% disp("video Precision:")
% disp(mean(p));
% disp("video Recall:");
% disp(mean(r));

Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtrain_video;     % Testing 
Input.TestingLabel=Ltrain;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
v_tr = Output2.EstimatedLabel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% audio %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Input.TrainingData=Dtrain_audio;    % Input data samples
Input.TrainingLabel=Ltrain;   % Labels of the input data samples
GranLevel=12;                % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
DistanceType='Euclidean';  % Type of distance/dissimilarity SOF classifier uses, which can be 'Mahalanobis', 'Cosine' or 'Euclidean'
Mode='OfflineTraining';      % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output1]=SOFClassifier(Input,GranLevel,Mode,DistanceType); 
% Output1.TrainedClassifier  - Offline primed SOF classifier
%% The SOF classifier conducts validation on testing data
Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtest_audio;     % Testing 
Input.TestingLabel=Ltest;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
cm = Output2.ConfusionMatrix;
a_te = Output2.EstimatedLabel;
save("cm/audio_happy_12.mat",'cm');
[cm,a,p,r,f] = getcm(Ltest,Output2.EstimatedLabel,1:2);
disp("audio Accuracy:");
disp(100*a/4832);
disp("audio F score:")
disp(mean(f));
% disp("audio Precision:")
% disp(mean(p));
% disp("audio Recall:");
% disp(mean(r));

Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtrain_audio;     % Testing 
Input.TestingLabel=Ltrain;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
a_tr = Output2.EstimatedLabel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% text %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Input.TrainingData=Dtrain_text;    % Input data samples
Input.TrainingLabel=Ltrain;   % Labels of the input data samples
GranLevel=12;                % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
DistanceType='Euclidean';  % Type of distance/dissimilarity SOF classifier uses, which can be 'Mahalanobis', 'Cosine' or 'Euclidean'
Mode='OfflineTraining';      % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output1]=SOFClassifier(Input,GranLevel,Mode,DistanceType); 
% Output1.TrainedClassifier  - Offline primed SOF classifier
%% The SOF classifier conducts validation on testing data
Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtest_text;     % Testing 
Input.TestingLabel=Ltest;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
cm = Output2.ConfusionMatrix;
t_te = Output2.EstimatedLabel;
save("cm/text_happy_12.mat",'cm');
[cm,a,p,r,f] = getcm(Ltest,Output2.EstimatedLabel,1:2);
disp("text Accuracy:");
disp(100*a/4832);
disp("text F score:")
disp(mean(f));
% disp("text Precision:")
% disp(mean(p));
% disp("text Recall:");
% disp(mean(r));

Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtrain_text;     % Testing 
Input.TestingLabel=Ltrain;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
t_tr = Output2.EstimatedLabel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 2-stage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dtrain = cat(2,v_tr,a_tr,t_tr);
Dtest = cat(2,v_te,a_te,t_te);
Input.TrainingData=Dtrain;    % Input data samples
Input.TrainingLabel=Ltrain;   % Labels of the input data samples
GranLevel=12;                % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
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
save("cm/stage_2_happy_12.mat",'cm');
[cm,a,p,r,f] = getcm(Ltest,Output2.EstimatedLabel,1:2);
disp("2 stage Accuracy:");
disp(100*a/4832);
disp("2 stage F score:")
disp(mean(f));
% disp("2 stage Precision:")
% disp(mean(p));
% disp("2 stage Recall:");
% disp(mean(r));