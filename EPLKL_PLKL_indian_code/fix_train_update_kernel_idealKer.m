function[parameters,SVMResultTest_w_regu,SVMResultTest_s_regu,SVMResultTest_ws_regu,SVMResultTest2_w,SVMResultTest2_s,SVMResultTest2_mss,SVMResultTest_mss_regu,SVMResultTest_w_mss,SVMResultTest_w_mss_regu,SVMResultTest_ws,test_labels] = fix_train_update_kernel_idealKer(K_whole_mss,vector_pos_to_union_all_index,Data,DataClass,train_SL,test_SL)
%% A simple Matlab demo for the following paper:
% J. Peng, H. Chen, Y. Zhou, L. Li
% Ideal regularized composite kernel for hyperspectral image classification
% IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
% DOI: 10.1109/JSTARS.2016.2621416
% If you use this code, please kindly cite the paper.
% If you have any problem, please contact J. Peng (pengjt1982@126.com)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Indian£¬ideal kernel




[nr nc nDim] = size(Data);  nAll = nr*nc;
gt2 = reshape(DataClass, nAll, 1);
tmp = unique(gt2);  classLabel = tmp(tmp~=0); nClass = length(classLabel);
data2 = reshape(Data, nAll, nDim);
clear Data* indian*

% Parameter setting
Cck = 1e4;  sigma1 = 0.25;  sigma2 = 2;  % SVM-CK
wopt = 9;  mu = 0.8;  gamma = 1e0;
mu_wm=0.1; gamma_wm=1e0;


postrn=train_SL;
postst=test_SL;
XTrn = data2(postrn(1,:),:);  YTrn = postrn(2, :)';
XTst = data2(postst(1,:),:);  YTst = postst(2, :)';
ntrn = length(YTrn);  ntst = length(YTst); nSam = ntrn + ntst;
% L2 normalize
XTrnL2 = XTrn./ repmat(sqrt(sum(XTrn.*XTrn,2)),[1 nDim]); % L2 norm
XTstL2 = XTst./ repmat(sqrt(sum(XTst.*XTst,2)),[1 nDim]); % L2 norm

% ideal kernel
T = zeros(ntrn,ntrn);
for i = 1 : ntrn
    for j = 1 : ntrn
        T(i,j) = (YTrn(i)==YTrn(j));
    end
end

% SVM on original spectral data
bestSig = 0.0625; bestC = 100;
K  = calckernel('rbf', bestSig, XTrnL2, XTrnL2);
Kt = calckernel('rbf', bestSig, XTrnL2, XTstL2);
cmd = [' -t 4 -c ', num2str(bestC) ];
model = svmtrain(YTrn, [(1:ntrn)' K], cmd);
Ypre_svm = svmpredict(YTst, [(1:ntst)' Kt], model);
OA_SVM = calcError(YTst, Ypre_svm, classLabel);
[OA_SVM, AA_SVM, K_SVM, CA_SVM]=confusion(YTst,Ypre_svm);

% ideal regularization on Kspe
Kid = exp(log(K)+gamma*T);
invK = pinv(K); S = invK*(Kid+K)*invK; Ktid = - Kt + Kt*S*K;
model = svmtrain(YTrn, [(1:ntrn)' Kid], cmd);
Ypre_svmid = svmpredict(YTst, [(1:ntst)' Ktid], model);
OA_SVMid = calcError(YTst, Ypre_svmid, classLabel);
[OA_SVMid, AA_SVMid, K_SVMid, CA_SVMid]=confusion(YTst,Ypre_svmid);

% set for squared neighborhood
w = wopt; wc = (w-1)/2;  vv = -wc : wc;
idw0 = repmat(vv*nr,w,1) + repmat(vv',1,w); idw0 = reshape(idw0,1,w*w);
XTrnm = zeros(ntrn, nDim);  XTstm = zeros(ntst, nDim);
for i = 1 : ntrn
    idw = idw0 + postrn(1,i); idw = [postrn(1,i) idw]; idw((w^2+1)/2+1)=[];
    idw(idw>nAll | idw<1) = [];
    Xtmp = data2(idw,:);  XTrnm(i,:)   = mean(Xtmp,1);
end
for i = 1 : ntst
    idw = idw0 + postst(1,i); idw = [postst(1,i) idw]; idw((w^2+1)/2+1)=[];
    idw(idw>nAll | idw<1) = [];
    Xtmp = data2(idw,:);  XTstm(i,:)   = mean(Xtmp,1);
end
XTrnmL2 = XTrnm./ repmat(sqrt(sum(XTrnm.*XTrnm,2)),[1 nDim]); % L2 norm
XTstmL2 = XTstm./ repmat(sqrt(sum(XTstm.*XTstm,2)),[1 nDim]); % L2 norm

% SVM on original spatial data
bestSig = 0.5; bestC = 1e4;
K  = calckernel('rbf', bestSig, XTrnmL2, XTrnmL2);
Kt = calckernel('rbf', bestSig, XTrnmL2, XTstmL2);
cmd = [' -t 4 -c ', num2str(bestC) ];
model = svmtrain(YTrn, [(1:ntrn)' K], cmd);
Ypre_svmmean = svmpredict(YTst, [(1:ntst)' Kt], model);
OA_SVMMean = calcError(YTst, Ypre_svmmean, classLabel);
[OA_SVMMean, AA_SVMMean, K_SVMMean, CA_SVMMean]=confusion(YTst,Ypre_svmmean);

% ideal regularization on Kspa
Kid = exp(log(K)+gamma*T);
invK = inv(K+1e-5*eye(size(K))); S = invK*(Kid+K)*invK; Ktid = - Kt + Kt*S*K;
model = svmtrain(YTrn, [(1:ntrn)' Kid], cmd);
Ypre_svmmeanid = svmpredict(YTst, [(1:ntst)' Ktid], model);
OA_SVMMeanid = calcError(YTst, Ypre_svmmeanid, classLabel);
[OA_SVMMeanid, AA_SVMMeanid, K_SVMMeanid, CA_SVMMeanid]=confusion(YTst,Ypre_svmmeanid);

% SVM-CK on original spectral-spatial data
K1  = calckernel('rbf', sigma2, XTrnL2, XTrnL2);
K2  = calckernel('rbf', sigma2, XTrnL2, XTstL2);
Km1 = calckernel('rbf', sigma1, XTrnmL2, XTrnmL2);
Km2 = calckernel('rbf', sigma1, XTrnmL2, XTstmL2);
KTrn = mu*Km1 + (1-mu)*K1;  KTst = mu*Km2 + (1-mu)*K2;
cmd = [' -t 4 -c ', num2str(Cck) ];
model = svmtrain(YTrn, [(1:ntrn)' KTrn], cmd);
Ypre_svmck = svmpredict(YTst, [(1:ntst)' KTst], model);
OA_SVMCK = calcError(YTst, Ypre_svmck, classLabel);
[OA_SVMCK, AA_SVMCK, K_SVMCK, CA_SVMCK]=confusion(YTst,Ypre_svmck);

% ideal regularization on Ksvmck
KTrnid = exp(log(KTrn)+gamma*T);
invK1 = inv(KTrn+1e-5*eye(size(KTrn))); S = invK1*(KTrnid+KTrn)*invK1; KTstid = - KTst + KTst*S*KTrn;
model = svmtrain(YTrn, [(1:ntrn)' KTrnid], cmd);
Ypre_svmckid = svmpredict(YTst, [(1:ntst)' KTstid], model);
OA_SVMCKid = calcError(YTst, Ypre_svmckid, classLabel);
[OA_SVMCKid, AA_SVMCKid, K_SVMCKid, CA_SVMCKid]=confusion(YTst,Ypre_svmckid);

%% 
% load K_whole_mss_indianbestSig_consider0
% load vector_pos_to_union_all_index
[KLL_ms,KUL_ms]= compute_meanspatial_spatialkernel_quick(K_whole_mss,vector_pos_to_union_all_index,postrn(1,:),postst(1,:));
K_mss =  [ (1:ntrn)' KLL_ms];
KK_mss=   [(1:ntst)',KUL_ms];
cmd = [' -t 4 -c ', num2str(Cck) ];

model_mss = svmtrain(YTrn, K_mss, cmd);
% GroudTest = double(test_labels(:,1));
[Ypre_svmckm,acc_mss,decVals_mss]= svmpredict(YTst,KK_mss,model_mss); 
OA_SVMkm1= calcError(YTst, Ypre_svmckm, classLabel);
[OA_SVMkm, AA_SVMkm, K_SVMkm, CA_SVMkm]=confusion(YTst,Ypre_svmckm);
KTrn=KLL_ms;
KTst=KUL_ms;
% ideal regularization on Ksvmck
KTrnid = exp(log(KTrn)+gamma_wm*T);
invK1 = inv(KTrn+1e-5*eye(size(KTrn))); S = invK1*(KTrnid+KTrn)*invK1; KTstid = - KTst + KTst*S*KTrn;
model = svmtrain(YTrn, [(1:ntrn)' KTrnid], cmd);
Ypre_svmckmid = svmpredict(YTst, [(1:ntst)' KTstid], model);
OA_SVMCKmid = calcError(YTst, Ypre_svmckmid, classLabel);
[OA_SVMCKmid, AA_SVMCKmid, K_SVMCKmid, CA_SVMCKmid]=confusion(YTst,Ypre_svmckmid);

%% added 
% SVM-CK on original spectral-mean spatial data
K1  = calckernel('rbf', sigma2, XTrnL2, XTrnL2);
K2  = calckernel('rbf', sigma2, XTrnL2, XTstL2);
KTrn = mu_wm*KLL_ms + (1-mu_wm)*K1;  KTst = mu_wm*KUL_ms + (1-mu_wm)*K2;
cmd = [' -t 4 -c ', num2str(Cck) ];
model = svmtrain(YTrn, [(1:ntrn)' KTrn], cmd);
Ypre_svmckwm = svmpredict(YTst, [(1:ntst)' KTst], model);
OA_SVMKwm = calcError(YTst, Ypre_svmckwm, classLabel);
[OA_SVMKwm, AA_SVMKwm, K_SVMKwm, CA_SVMKwm]=confusion(YTst,Ypre_svmckwm);

% ideal regularization on Ksvmck
KTrnid = exp(log(KTrn)+gamma_wm*T);
invK1 = inv(KTrn+1e-5*eye(size(KTrn))); S = invK1*(KTrnid+KTrn)*invK1; KTstid = - KTst + KTst*S*KTrn;
model = svmtrain(YTrn, [(1:ntrn)' KTrnid], cmd);
Ypre_svmckwmid = svmpredict(YTst, [(1:ntst)' KTstid], model);
OA_SVMCKwmid = calcError(YTst, Ypre_svmckwmid, classLabel);
[OA_SVMCKwmid, AA_SVMCKwmid, K_SVMCKwmid, CA_SVMCKwmid]=confusion(YTst,Ypre_svmckwmid);

[OA_SVM  OA_SVMid  OA_SVMMean OA_SVMMeanid OA_SVMCK OA_SVMCKid OA_SVMkm OA_SVMCKmid OA_SVMKwm OA_SVMCKwmid]
parameters.Cck = Cck;  
parameters.sigma1 = sigma1;  
parameters.sigma2 = sigma2;  % SVM-CK
parameters.wopt = wopt;  
parameters.mu = mu;  
parameters.gamma = gamma;
parameters.mu_wm=mu_wm;
parameters.gamma_wm=gamma_wm;

parameters.bestSig_w=0.0625;
parameters.bestSig_s=0.5;
parameters.bestC_w=100;
parameters.bestC_s=1e4;
SVMResultTest_w_regu=Ypre_svmid;
SVMResultTest_s_regu=Ypre_svmmeanid;
SVMResultTest_ws_regu=Ypre_svmckid;
SVMResultTest2_w=Ypre_svm;
SVMResultTest2_s=Ypre_svmmean;
SVMResultTest2_mss=Ypre_svmckm;
SVMResultTest_mss_regu=Ypre_svmckmid;
SVMResultTest_w_mss=Ypre_svmckwm;
SVMResultTest_w_mss_regu=Ypre_svmckwmid;
SVMResultTest_ws=Ypre_svmck;
test_labels=YTst;




