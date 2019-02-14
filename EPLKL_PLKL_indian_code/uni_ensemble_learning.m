function [pre,f_pre,oa, aa, K, ua] = uni_ensemble_learning(GroudTest,SVMResultTest,num_sample,num_sampling,no_classes )
%UNI_ENSEMBLE_LEARNING 此处显示有关此函数的摘要
%   此处显示详细说明
f_pre=zeros(num_sample,no_classes);
pre=zeros(num_sample,1);
iter=num_sampling;
for i=1:num_sample
    for j=1:iter
      f_pre(i,SVMResultTest(j,i))=f_pre(i,SVMResultTest(j,i))+1;
    end
end
for i=1:num_sample
    value=find(f_pre(i,:)==max(f_pre(i,:)));
    pre(i,1)=value(1);
end
[oa, aa, K, ua]=confusion(GroudTest,pre);
oa
aa
K
ua
end

