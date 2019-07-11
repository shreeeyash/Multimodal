clear all
clc
close all

Dtrain = load("Dtrain.mat");
Dtrain = Dtrain.val;
Dtest = load("Dtest.mat");
Dtest = Dtest.val;
Ltrain_e1 = load("Ltrain_e1.mat");
Ltrain_e1 = Ltrain_e1.val;
Ltrain_e2 = load("Ltrain_e2.mat");
Ltrain_e2 = Ltrain_e2.val;
Ltrain_e3 = load("Ltrain_e3.mat");
Ltrain_e3 = Ltrain_e3.val;
Ltrain_e4 = load("Ltrain_e4.mat");
Ltrain_e4 = Ltrain_e4.val;
Ltrain_e5 = load("Ltrain_e5.mat");
Ltrain_e5 = Ltrain_e5.val;
Ltrain_e6 = load("Ltrain_e6.mat");
Ltrain_e6 = Ltrain_e6.val;
Ltrain = load("Ltrain.mat");
Ltrain = Ltrain.val;

Ltest = load("Ltest.mat");
Ltest = Ltest.val;
Ltest_e1 = load("Ltest_e1.mat");
Ltest_e1 = Ltest_e1.val;
Ltest_e2 = load("Ltest_e2.mat");
Ltest_e2 = Ltest_e2.val;
Ltest_e3 = load("Ltest_e3.mat");
Ltest_e3 = Ltest_e3.val;
Ltest_e4 = load("Ltest_e4.mat");
Ltest_e4 = Ltest_e4.val;
Ltest_e5 = load("Ltest_e5.mat");
Ltest_e5 = Ltest_e5.val;
Ltest_e6 = load("Ltest_e6.mat");
Ltest_e6 = Ltest_e6.val;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% e1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Input.TrainingData=Dtrain;    % Input data samples
Input.TrainingLabel=Ltrain_e1;   % Labels of the input data samples
GranLevel=12;                % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
DistanceType='Euclidean';  % Type of distance/dissimilarity SOF classifier uses, which can be 'Mahalanobis', 'Cosine' or 'Euclidean'
Mode='OfflineTraining';      % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output1]=SOFClassifier(Input,GranLevel,Mode,DistanceType); 
% Output1.TrainedClassifier  - Offline primed SOF classifier
%% The SOF classifier conducts validation on testing data
Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtest;     % Testing 
Input.TestingLabel=Ltest_e1;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
cm = Output2.ConfusionMatrix;
e1_te = Output2.EstimatedLabel;
% save("cm/stage_2_iemocap_EF_12.mat",'cm');
% [cm,a,p,r,f] = getcm(Ltest,Output2.EstimatedLabel,1:2);
% disp("video Accuracy:");
% disp(100*a/4832);
% disp("video F score:")
% disp(mean(f));
% disp("video Precision:")
% disp(mean(p));
% disp("video Recall:");
% disp(mean(r));

Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtrain;     % Testing 
Input.TestingLabel=Ltrain_e1;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
e1_tr = Output2.EstimatedLabel;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% e2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Input.TrainingData=Dtrain;    % Input data samples
Input.TrainingLabel=Ltrain_e2;   % Labels of the input data samples
GranLevel=12;                % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
DistanceType='Euclidean';  % Type of distance/dissimilarity SOF classifier uses, which can be 'Mahalanobis', 'Cosine' or 'Euclidean'
Mode='OfflineTraining';      % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output1]=SOFClassifier(Input,GranLevel,Mode,DistanceType); 
% Output1.TrainedClassifier  - Offline primed SOF classifier
%% The SOF classifier conducts validation on testing data
Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtest;     % Testing 
Input.TestingLabel=Ltest_e2;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
cm = Output2.ConfusionMatrix;
e2_te = Output2.EstimatedLabel;
% save("cm/stage_2_iemocap_EF_12.mat",'cm');
% [cm,a,p,r,f] = getcm(Ltest,Output2.EstimatedLabel,1:2);
% disp("video Accuracy:");
% disp(100*a/4832);
% disp("video F score:")
% disp(mean(f));
% disp("video Precision:")
% disp(mean(p));
% disp("video Recall:");
% disp(mean(r));

Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtrain;     % Testing 
Input.TestingLabel=Ltrain_e2;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
e2_tr = Output2.EstimatedLabel;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% e3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Input.TrainingData=Dtrain;    % Input data samples
Input.TrainingLabel=Ltrain_e3;   % Labels of the input data samples
GranLevel=12;                % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
DistanceType='Euclidean';  % Type of distance/dissimilarity SOF classifier uses, which can be 'Mahalanobis', 'Cosine' or 'Euclidean'
Mode='OfflineTraining';      % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output1]=SOFClassifier(Input,GranLevel,Mode,DistanceType); 
% Output1.TrainedClassifier  - Offline primed SOF classifier
%% The SOF classifier conducts validation on testing data
Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtest;     % Testing 
Input.TestingLabel=Ltest_e3;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
cm = Output2.ConfusionMatrix;
e3_te = Output2.EstimatedLabel;
% save("cm/stage_2_iemocap_EF_12.mat",'cm');
% [cm,a,p,r,f] = getcm(Ltest,Output2.EstimatedLabel,1:2);
% disp("video Accuracy:");
% disp(100*a/4832);
% disp("video F score:")
% disp(mean(f));
% disp("video Precision:")
% disp(mean(p));
% disp("video Recall:");
% disp(mean(r));

Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtrain;     % Testing 
Input.TestingLabel=Ltrain_e3;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
e3_tr = Output2.EstimatedLabel;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% e4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Input.TrainingData=Dtrain;    % Input data samples
Input.TrainingLabel=Ltrain_e4;   % Labels of the input data samples
GranLevel=12;                % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
DistanceType='Euclidean';  % Type of distance/dissimilarity SOF classifier uses, which can be 'Mahalanobis', 'Cosine' or 'Euclidean'
Mode='OfflineTraining';      % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output1]=SOFClassifier(Input,GranLevel,Mode,DistanceType); 
% Output1.TrainedClassifier  - Offline primed SOF classifier
%% The SOF classifier conducts validation on testing data
Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtest;     % Testing 
Input.TestingLabel=Ltest_e4;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
cm = Output2.ConfusionMatrix;
e4_te = Output2.EstimatedLabel;
% save("cm/stage_2_iemocap_EF_12.mat",'cm');
% [cm,a,p,r,f] = getcm(Ltest,Output2.EstimatedLabel,1:2);
% disp("video Accuracy:");
% disp(100*a/4832);
% disp("video F score:")
% disp(mean(f));
% disp("video Precision:")
% disp(mean(p));
% disp("video Recall:");
% disp(mean(r));

Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtrain;     % Testing 
Input.TestingLabel=Ltrain_e4;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
e4_tr = Output2.EstimatedLabel;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% e5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Input.TrainingData=Dtrain;    % Input data samples
Input.TrainingLabel=Ltrain_e5;   % Labels of the input data samples
GranLevel=12;                % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
DistanceType='Euclidean';  % Type of distance/dissimilarity SOF classifier uses, which can be 'Mahalanobis', 'Cosine' or 'Euclidean'
Mode='OfflineTraining';      % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output1]=SOFClassifier(Input,GranLevel,Mode,DistanceType); 
% Output1.TrainedClassifier  - Offline primed SOF classifier
%% The SOF classifier conducts validation on testing data
Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtest;     % Testing 
Input.TestingLabel=Ltest_e5;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
cm = Output2.ConfusionMatrix;
e5_te = Output2.EstimatedLabel;
% save("cm/stage_2_iemocap_EF_12.mat",'cm');
% [cm,a,p,r,f] = getcm(Ltest,Output2.EstimatedLabel,1:2);
% disp("video Accuracy:");
% disp(100*a/4832);
% disp("video F score:")
% disp(mean(f));
% disp("video Precision:")
% disp(mean(p));
% disp("video Recall:");
% disp(mean(r));

Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtrain;     % Testing 
Input.TestingLabel=Ltrain_e5;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
e5_tr = Output2.EstimatedLabel;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% e6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Input.TrainingData=Dtrain;    % Input data samples
Input.TrainingLabel=Ltrain_e6;   % Labels of the input data samples
GranLevel=12;                % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
DistanceType='Euclidean';  % Type of distance/dissimilarity SOF classifier uses, which can be 'Mahalanobis', 'Cosine' or 'Euclidean'
Mode='OfflineTraining';      % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output1]=SOFClassifier(Input,GranLevel,Mode,DistanceType); 
% Output1.TrainedClassifier  - Offline primed SOF classifier
%% The SOF classifier conducts validation on testing data
Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtest;     % Testing 
Input.TestingLabel=Ltest_e6;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
cm = Output2.ConfusionMatrix;
e6_te = Output2.EstimatedLabel;
% save("cm/stage_2_iemocap_EF_12.mat",'cm');
% [cm,a,p,r,f] = getcm(Ltest,Output2.EstimatedLabel,1:2);
% disp("video Accuracy:");
% disp(100*a/4832);
% disp("video F score:")
% disp(mean(f));
% disp("video Precision:")
% disp(mean(p));
% disp("video Recall:");
% disp(mean(r));

Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtrain;     % Testing 
Input.TestingLabel=Ltrain_e6;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
e6_tr = Output2.EstimatedLabel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 2-stage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dtrain = cat(2,e1_tr,1-e1_tr,e2_tr,1-e2_tr,e3_tr,1-e3_tr,e4_tr,1-e4_tr,e5_tr,1-e5_tr,e6_tr,1-e6_tr);
Dtest = cat(2,e1_te,1-e1_te,e2_te,1-e2_te,e3_te,1-e3_te,e4_te,1-e4_te,e5_te,1-e5_te,e6_te,1-e6_te);
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
save("cm/stage_2_surprise_12.mat",'cm');
[cm,a,p,r,f] = getcm(Ltest,Output2.EstimatedLabel,1:2);
disp("2 stage Accuracy:");
disp(100*a/1807);
disp("2 stage F score:")
disp(mean(f));
disp("2 stage Precision:")
disp(mean(p));
disp("2 stage Recall:");
disp(mean(r));