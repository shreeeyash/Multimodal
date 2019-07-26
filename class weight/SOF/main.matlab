clear all
clc
close all

Dtrain = load("Dtrain_sad_lstm640.mat");
Dtrain = double(Dtrain.val);
Ltrain = load("Ltrain_sad.mat");
Ltrain = double(Ltrain.val);
Dtest = load("Dtest_sad_lstm640.mat");
Dtest = double(Dtest.val);
Ltest = load("Ltest_sad.mat");
Ltest = double(Ltest.val);

[rank,weight] = relieff(Dtest,Ltest,33);
rank = rank(1:390);
Dtest = Dtest(:,rank);
disp(size(Dtest));

[rank,weight] = relieff(Dtrain,Ltrain,33);
rank = rank(1:390);
Dtrain = Dtrain(:,rank);
disp(size(Dtrain));

disp("======================== gran : 5 ===============================")
Input.TrainingData=Dtrain;    % Input data samples
Input.TrainingLabel=Ltrain;   % Labels of the input data samples
GranLevel=5;                % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
DistanceType='Euclidean';  % Type of distance/dissimilarity SOF classifier uses, which can be 'Mahalanobis', 'Cosine' or 'Euclidean'
Mode='OfflineTraining';      % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output1]=SOFClassifier(Input,GranLevel,Mode,DistanceType); 



Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtest;     % Testing 
Input.TestingLabel=Ltest;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
cm = Output2.ConfusionMatrix;
[cm,a,p,r,f] = getcm(Ltest,Output2.EstimatedLabel,1:2);

TP = cm(2,2);
TN = cm(1,1);
P = sum(Ltest-1);
N = 4832 - P;
WA = ((TP*N/P)+TN)/(2*N);

disp('WA');
disp(WA*100);

disp("======================== gran : 6 ===============================")
Input.TrainingData=Dtrain;    % Input data samples
Input.TrainingLabel=Ltrain;   % Labels of the input data samples
GranLevel=6;                % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
DistanceType='Euclidean';  % Type of distance/dissimilarity SOF classifier uses, which can be 'Mahalanobis', 'Cosine' or 'Euclidean'
Mode='OfflineTraining';      % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output1]=SOFClassifier(Input,GranLevel,Mode,DistanceType); 



Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtest;     % Testing 
Input.TestingLabel=Ltest;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
cm = Output2.ConfusionMatrix;
[cm,a,p,r,f] = getcm(Ltest,Output2.EstimatedLabel,1:2);

TP = cm(2,2);
TN = cm(1,1);
P = sum(Ltest-1);
N = 4832 - P;
WA = ((TP*N/P)+TN)/(2*N);

disp('WA');
disp(WA*100);

disp("======================== gran : 8 ===============================")
Input.TrainingData=Dtrain;    % Input data samples
Input.TrainingLabel=Ltrain;   % Labels of the input data samples
GranLevel=8;                % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
DistanceType='Euclidean';  % Type of distance/dissimilarity SOF classifier uses, which can be 'Mahalanobis', 'Cosine' or 'Euclidean'
Mode='OfflineTraining';      % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output1]=SOFClassifier(Input,GranLevel,Mode,DistanceType); 



Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtest;     % Testing 
Input.TestingLabel=Ltest;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
cm = Output2.ConfusionMatrix;
[cm,a,p,r,f] = getcm(Ltest,Output2.EstimatedLabel,1:2);

TP = cm(2,2);
TN = cm(1,1);
P = sum(Ltest-1);
N = 4832 - P;
WA = ((TP*N/P)+TN)/(2*N);

disp('WA');
disp(WA*100);


disp("======================== gran : 12 ===============================")
Input.TrainingData=Dtrain;    % Input data samples
Input.TrainingLabel=Ltrain;   % Labels of the input data samples
GranLevel=12;                % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
DistanceType='Euclidean';  % Type of distance/dissimilarity SOF classifier uses, which can be 'Mahalanobis', 'Cosine' or 'Euclidean'
Mode='OfflineTraining';      % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output1]=SOFClassifier(Input,GranLevel,Mode,DistanceType); 



Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtest;     % Testing 
Input.TestingLabel=Ltest;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
cm = Output2.ConfusionMatrix;
[cm,a,p,r,f] = getcm(Ltest,Output2.EstimatedLabel,1:2);

TP = cm(2,2);
TN = cm(1,1);
P = sum(Ltest-1);
N = 4832 - P;
WA = ((TP*N/P)+TN)/(2*N);

disp('WA');
disp(WA*100);

disp("======================== gran : 20 ===============================")
Input.TrainingData=Dtrain;    % Input data samples
Input.TrainingLabel=Ltrain;   % Labels of the input data samples
GranLevel=20;                % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
DistanceType='Euclidean';  % Type of distance/dissimilarity SOF classifier uses, which can be 'Mahalanobis', 'Cosine' or 'Euclidean'
Mode='OfflineTraining';      % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output1]=SOFClassifier(Input,GranLevel,Mode,DistanceType); 



Input=Output1;               % Offline primed SOF classifier
Input.TestingData=Dtest;     % Testing 
Input.TestingLabel=Ltest;    % Labels of the tetsing data samples
Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
[Output2]=SOFClassifier(Input,GranLevel,Mode,DistanceType);
cm = Output2.ConfusionMatrix;
[cm,a,p,r,f] = getcm(Ltest,Output2.EstimatedLabel,1:2);

TP = cm(2,2);
TN = cm(1,1);
P = sum(Ltest-1);
N = 4832 - P;
WA = ((TP*N/P)+TN)/(2*N);

disp('WA');
disp(WA*100);