%% Copyright (c) 2018, Plamen P. Angelov and Xiaowei Gu

%% All rights reserved. Please read the "license.txt" for license terms.

%% This code is the Self-Organising Fuzzy Logic (SOF) classifier described in:
%==========================================================================================================
%% X. Gu, P. Angelov, "Self-organising fuzzy logic classifier," 
%% Information Sciences, vol. 447, pp. 36-51, 2018.
%==========================================================================================================
%% Please cite the paper above if this code helps.

%% For any queries about the code, please contact Prof. Plamen P. Angelov and Dr. Xiaowei Gu 
%% {p.angelov,x.gu3}@lancaster.ac.uk

%% Programmed by Xiaowei Gu
function [Output]=SOFClassifier(Input,GranLevel,Mode,DistanceType)
%% GranLevel Level of Granularity, which can be any postive integer
%% DistanceType  Currently SOF classifier supports Euclidean distance, Mahahalnobis distance and cosine dissimilarityx
% DistanceType=='Cosine' SOF Classifier uses cosine dissimilarity
% DistanceType=='Euclidean' SOF Classifier uses Euclidean distance
% DistanceType=='Mahalanobis' SOF Classifier uses Mahalanobis distance
%% Mode  The operating mode of the SOF classifier
%% Mode=='OfflineTraining' SOF Classifier learns from static data
% Input.TrainingData        -   Training data
% Input.TrainingLabel       -   Labels of the training data
% Output.TrainedClassifier  -   Trained SOF classifier
%% Mode=='EvolvingTraining' SOF Classifier continues to learn from streaming data after primed offline
% Input.TrainingData        -   Training data
% Input.TrainingLabel       -   Labels of the training data
% Input.TrainedClassifier   -   Offline primed SOF classifier
% Output.TrainedClassifier  -   Online trained SOF classifier
%% Mode=='Validation' SOF Classifier conducts testing on validation data
% Input.TrainedClassifier   -   The trained SOF classifier
% Input.TestingData         -   Validation data
% Input.TestingLabel        -   Labels of the validation data (ground truth)
%                               Used for producing confusion matrix only
% Output.EstimatedLabel     -   Estimated label of validation data
% Output.TrainedClassifier  -   The SOF classifier, same as the input
% Output.ConfusionMatrix    -   Confusion matrix of the result

if strcmp(Mode,'OfflineTraining')==1
    data_train=Input.TrainingData;
    label_train=Input.TrainingLabel;
    seq=unique(label_train);
    data_train1={};
    N=length(seq);
    %%
    if strcmp(DistanceType,'Cosine')==1
        data_train=data_train./(repmat(sqrt(sum(data_train.^2,2)),1,size(data_train,2)));
        DistanceType='Euclidean';
    end
    if strcmp(DistanceType,'Euclidean')==1
        for i=1:1:N
            data_train1{i}=data_train(label_train==seq(i),:);
            delta(i)=mean(sum(data_train1{i}.^2,2))-sum(mean(data_train1{i},1).^2);
            [centre{i},Member{i},averdist{i}]=offline_training_Euclidean(data_train1{i},delta(i),GranLevel);
        end
        L=zeros(1,N);
        mu={};
        XX=zeros(1,N);
        ratio=zeros(1,N);
        for i=1:1:N
            mu{i}=mean(data_train1{i},1);
            [L(i),W]=size(data_train1{i});
            XX(i)=0;
            for ii=1:1:L(i)
                XX(i)=XX(i)+sum(data_train1{i}(ii,:).^2);
            end
            XX(i)=XX(i)./L(i);
            ratio(i)=averdist{i}/(2*(XX(i)-sum(mu{i}.^2)));
        end
        TrainedClassifier.seq=seq;
        TrainedClassifier.ratio=ratio;
        TrainedClassifier.miu=mu;
        TrainedClassifier.XX=XX;
        TrainedClassifier.L=L;
        TrainedClassifier.centre=centre;
        TrainedClassifier.Member=Member;
        TrainedClassifier.averdist=averdist;
        TrainedClassifier.NoC=N;
        TrainedClassifier.delta=delta;
    end
    if strcmp(DistanceType,'Mahalanobis')==1
        for i=1:1:N
            data_train1{i}=data_train(label_train==seq(i),:);
            cov_data{i}=cov(data_train1{i});
            [centre{i},Member{i},averdist{i}]=offline_training_mahalanobis(data_train1{i},cov_data{i},GranLevel);
        end
        L=zeros(1,N);
        mu={};
        XX={};
        threshold={};
        for i=1:1:N
            mu{i}=mean(data_train1{i},1);
            [L(i),W]=size(data_train1{i});
            XX{i}=zeros(W);
            for ii=1:1:L(i)
                XX{i}=XX{i}+data_train1{i}(ii,:)'*data_train1{i}(ii,:);
            end
            XX{i}=XX{i}./L(i);
            threshold{i}=averdist{i};
        end
        TrainedClassifier.seq=seq;
        TrainedClassifier.miu=mu;
        TrainedClassifier.XX=XX;
        TrainedClassifier.L=L;
        TrainedClassifier.centre=centre;
        TrainedClassifier.Member=Member;
        TrainedClassifier.threshold=threshold;
        TrainedClassifier.NoC=N;
        TrainedClassifier.covMatrix=cov_data;
    end
    Output.TrainedClassifier=TrainedClassifier;
end
%%
if strcmp(Mode,'EvolvingTraining')==1
    data_train=Input.TrainingData;
    label_train=Input.TrainingLabel;
    TrainedClassifier=Input.TrainedClassifier;
    if strcmp(DistanceType,'Cosine')==1
        data_train=data_train./(repmat(sqrt(sum(data_train.^2,2)),1,size(data_train,2)));
        DistanceType='Euclidean';
    end
    if strcmp(DistanceType,'Euclidean')==1
        seq=TrainedClassifier.seq;
        ratio=TrainedClassifier.ratio;
        mu=TrainedClassifier.miu;
        XX=TrainedClassifier.XX;
        L=TrainedClassifier.L;
        centre=TrainedClassifier.centre;
        Member=TrainedClassifier.Member;
        averdist=TrainedClassifier.averdist;
        delta=TrainedClassifier.delta;
        N=TrainedClassifier.NoC;
        for i=1:1:N
            seq2=find(label_train==seq(i));
            data_train2{i}=data_train(seq2,:);
        end
        for i=1:1:N
            for j=1:1:size(data_train2{i},1)
                L(i)=L(i)+1;
                XX(i)=XX(i).*(L(i)-1)/(L(i))+sum(data_train2{i}(j,:).^2)./(L(i));
                mu{i}=mu{i}.*(L(i)-1)/(L(i))+data_train2{i}(j,:)./(L(i));
                delta(i)=XX(i)-sum(mu{i}.^2);
                threshold=2*delta(i)*ratio(i);
                [centre{i},Member{i}]=evolving_training_Euclidean(data_train2{i}(j,:),mu{i},centre{i},Member{i},delta(i),threshold);
                threshold=[];
            end
        end
        TrainedClassifier.ratio=ratio;
        TrainedClassifier.miu=mu;
        TrainedClassifier.XX=XX;
        TrainedClassifier.L=L;
        TrainedClassifier.centre=centre;
        TrainedClassifier.Member=Member;
        TrainedClassifier.averdist=averdist;
        TrainedClassifier.NoC=N;
        TrainedClassifier.delta=delta;
    end
    if strcmp(DistanceType,'Mahalanobis')==1
        seq=TrainedClassifier.seq;
        mu=TrainedClassifier.miu;
        XX=TrainedClassifier.XX;
        L=TrainedClassifier.L;
        centre=TrainedClassifier.centre;
        Member=TrainedClassifier.Member;
        cov_data=TrainedClassifier.covMatrix;
        threshold=TrainedClassifier.threshold;
        N=TrainedClassifier.NoC;
        for i=1:1:N
            seq2=find(label_train==seq(i));
            data_train2{i}=data_train(seq2,:);
        end
        for i=1:1:N
            for j=1:1:size(data_train2{i},1)
                L(i)=L(i)+1;
                XX{i}=XX{i}.*(L(i)-1)/(L(i))+data_train2{i}(j,:)'*data_train2{i}(j,:)./(L(i));
                mu{i}=mu{i}.*(L(i)-1)/(L(i))+data_train2{i}(j,:)./(L(i));
                cov_data{i}=(L(i))/(L(i)-1)*(XX{i}-mu{i}'*mu{i});
                threshold1=threshold{i};
                [centre{i},Member{i}]=evolving_training_mahalanobis(data_train2{i}(j,:),mu{i},centre{i},Member{i},cov_data{i},threshold1);
                threshold1=[];
            end
        end
         TrainedClassifier.seq=seq;
        TrainedClassifier.miu=mu;
        TrainedClassifier.XX=XX;
        TrainedClassifier.L=L;
        TrainedClassifier.centre=centre;
        TrainedClassifier.Member=Member;
        TrainedClassifier.threshold=threshold;
        TrainedClassifier.NoC=N;
        TrainedClassifier.covMatrix=cov_data;
    end
    Output.TrainedClassifier=TrainedClassifier;
end
%%
if strcmp(Mode,'Validation')==1
    TrainedClassifier=Input.TrainedClassifier;
    seq=TrainedClassifier.seq;
    data_test=Input.TestingData;
    label_test=Input.TestingLabel;
    N=TrainedClassifier.NoC;
    if strcmp(DistanceType,'Cosine')==1
        data_test=data_test./(repmat(sqrt(sum(data_test.^2,2)),1,size(data_test,2)));
        DistanceType='Euclidean';
    end
    if strcmp(DistanceType,'Euclidean')==1
        centre=TrainedClassifier.centre;
        dist=zeros(size(data_test,1),N);
        for i=1:1:N
            dist(:,i)=min(pdist2(data_test,centre{i},'euclidean').^2,[],2);
        end
        Output.dist = dist;
        label_estimated = softmax(-dist')';     % changed here by SYG
        [~,label_est]=min(dist,[],2);
        label_est=seq(label_est);
    end
    if strcmp(DistanceType,'Mahalanobis')==1
        cov_data=TrainedClassifier.covMatrix;
        centre=TrainedClassifier.centre;
        dist=zeros(size(data_test,1),N);
        for i=1:1:N
            dist(:,i)=min(pdist2(data_test,centre{i},'mahalanobis',cov_data{i}).^2,[],2);
        end
        [~,label_est]=min(dist,[],2);
        label_est=seq(label_est);
    end
    Output.TrainedClassifier=Input.TrainedClassifier;
    Output.ConfusionMatrix=confusionmat(label_test,label_est);
    Output.EstimatedLabel=label_est;
    Output.Estimatedprob=label_estimated;            % changed here by SYG

end
end
function [centre,member]=evolving_training_Euclidean(data,mu,centre,member,delta,threshold)
dist1=pdist2(centre,mu,'euclidean').^2;
dist2=pdist2(data,mu,'euclidean').^2;
if dist2>max(dist1)||dist2<min(dist1)
    centre(end+1,:)=data;
    member(end+1,1)=1;
else
    [dist3,pos3]=min(pdist2(data,centre,'euclidean').^2);
    if dist3>threshold
        centre(end+1,:)=data;
        member(end+1,1)=1;
    else
        centre(pos3,:)=member(pos3,1)/(member(pos3,1)+1)*centre(pos3,:)+1/(member(pos3,1)+1)*data;
        member(pos3,1)=member(pos3,1)+1;
    end
    
end
end
function [centre3,Mnumber,averdist]=offline_training_Euclidean(data,delta,GranLevel)
[L,W]=size(data);
dist00=pdist(data,'euclidean').^2;
dist0=squareform(dist00);
% dist1=dist0;
dist00=sort(dist00,'ascend');
for i=1:GranLevel
    dist00(dist00>mean(dist00))=[];
end
averdist=mean(dist00);

[UD,J,K]=unique(data,'rows');
F = histc(K,1:numel(J));
LU=length(UD(:,1));
%%
density=sum(dist0,2)./sum(sum(dist0,2));
density=F./density(J);
dist=dist0(J,J);
[~,pos]=max(density);
seq=1:1:LU;
seq=seq(seq~=pos);
Rank=zeros(LU,1);
Rank(1,:)=pos;
for i=2:1:LU
    [aa,pos0]=min(dist(pos,seq));
    pos=seq(pos0);
    Rank(i,:)=pos;
    seq=seq(seq~=pos);
end
data2=UD(Rank,:);
data2den=density(Rank);
Gradient=zeros(2,LU-2);
Gradient(1,:)=data2den(1:1:LU-2)-data2den(2:1:LU-1);
Gradient(2,:)=data2den(2:1:LU-1)-data2den(3:1:LU);
seq2=2:1:LU-1;
seq1=find(Gradient(1,:)<0&Gradient(2,:)>0);
if Gradient(2,LU-2)<0
    seq3=[1,seq2(seq1),LU];
else
    seq3=[1,seq2(seq1)];
end
centre0=data2(seq3,:);
% centre0=unique(data2(seq3,:),'rows');
nc=size(centre0,1);
dist3=pdist2(centre0,data,'euclidean').^2;
[~,seq4]=min(dist3,[],1);
centre1=zeros(nc,W);
Mnumber=zeros(nc,1);
miu=mean(data,1);
cenden=zeros(1,nc);
for i=1:1:nc
    seq5=find(seq4==i);
    Mnumber(i)=length(seq5);
    centre1(i,:)=mean(data(seq5,:),1);
    cenden(i)= Mnumber(i)/(1+sum((centre1(i,:)-miu).^2)/delta);
end
dist4=pdist(centre1,'euclidean').^2;
dist5=squareform(dist4);
seqme2=zeros(nc);
seqme2(dist5<=averdist)=1;
cendenmex=seqme2.*(repmat(cenden,nc,1));
seq6=find(abs(max(cendenmex,[],2)-cenden')==0);
centre2=centre1(seq6,:);
nc=size(centre2,1);
dist6=pdist2(centre2,data,'euclidean').^2;
[~,seq7]=min(dist6,[],1);
centre3=zeros(nc,W);
Mnumber=zeros(nc,1);
for i=1:1:nc
    seq8=find(seq7==i);
    Mnumber(i)=length(seq8);
    centre3(i,:)=mean(data(seq8,:),1);
end
end
function [centre,member]=evolving_training_mahalanobis(data,mu,centre,member,cov_data,threshold)
dist1=pdist2(centre,mu,'mahalanobis',cov_data).^2;
dist2=pdist2(data,mu,'mahalanobis',cov_data).^2;
if dist2>max(dist1)||dist2<min(dist1)
    centre(end+1,:)=data;
    member(end+1,1)=1;
else
    [dist3,pos3]=min(pdist2(data,centre,'mahalanobis',cov_data).^2);
    if dist3>threshold
        centre(end+1,:)=data;
        member(end+1,1)=1;
    else
        centre(pos3,:)=member(pos3,1)/(member(pos3,1)+1)*centre(pos3,:)+1/(member(pos3,1)+1)*data;
        member(pos3,1)=member(pos3,1)+1;
    end
    
end
end
function [centre3,Mnumber,averdist]=offline_training_mahalanobis(data,cov_data,GranLevel)
[L,W]=size(data);
dist00=pdist(data,'mahalanobis',cov_data).^2;
dist0=squareform(dist00);
% dist1=dist0;
dist00=sort(dist00,'ascend');
for i=1:GranLevel
    dist00(dist00>mean(dist00))=[];
end
averdist=mean(dist00);

[UD,J,K]=unique(data,'rows');
F = histc(K,1:numel(J));
LU=length(UD(:,1));
%%
density=sum(dist0,2)./sum(sum(dist0,2));
density=F./density(J);
dist=dist0(J,J);
[~,pos]=max(density);
seq=1:1:LU;
seq=seq(seq~=pos);
Rank=zeros(LU,1);
Rank(1,:)=pos;
for i=2:1:LU
    [aa,pos0]=min(dist(pos,seq));
    pos=seq(pos0);
    Rank(i,:)=pos;
    seq=seq(seq~=pos);
end
data2=UD(Rank,:);
data2den=density(Rank);
Gradient=zeros(2,LU-2);
Gradient(1,:)=data2den(1:1:LU-2)-data2den(2:1:LU-1);
Gradient(2,:)=data2den(2:1:LU-1)-data2den(3:1:LU);
seq2=2:1:LU-1;
seq1=find(Gradient(1,:)<0&Gradient(2,:)>0);
if Gradient(2,LU-2)<0
    seq3=[1,seq2(seq1),LU];
else
    seq3=[1,seq2(seq1)];
end
centre0=data2(seq3,:);
nc=size(centre0,1);
dist3=pdist2(centre0,data,'mahalanobis',cov_data).^2;
[~,seq4]=min(dist3,[],1);
centre1=zeros(nc,W);
Mnumber=zeros(nc,1);
miu=mean(data,1);
cenden=zeros(1,nc);
for i=1:1:nc
    seq5=find(seq4==i);
    Mnumber(i)=length(seq5);
    centre1(i,:)=mean(data(seq5,:),1);
    cenden(i)= Mnumber(i)/(1+(centre1(i,:)-miu)/cov_data*(centre1(i,:)-miu)'/(W));
end
dist4=pdist(centre1,'mahalanobis',cov_data).^2;
dist5=squareform(dist4);
seqme2=zeros(nc);
seqme2(dist5<=averdist)=1;
cendenmex=seqme2.*(repmat(cenden,nc,1));
seq6=find(abs(max(cendenmex,[],2)-cenden')==0);
centre2=centre1(seq6,:);
nc=size(centre2,1);
dist6=pdist2(centre2,data,'mahalanobis',cov_data).^2;
[~,seq7]=min(dist6,[],1);
centre3=zeros(nc,W);
Mnumber=zeros(nc,1);
for i=1:1:nc
    seq8=find(seq7==i);
    Mnumber(i)=length(seq8);
    centre3(i,:)=mean(data(seq8,:),1);
end
end