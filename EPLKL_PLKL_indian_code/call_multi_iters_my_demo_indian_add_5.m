function [  ] = call_multi_iters_my_demo_indian_add_5(  )
%CALL_MULTI_ITERS_MY_DEMO_PAVIAU �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
addpath('../backup_codes/');
load ./K_whole_mss_indianbestSig_consider0
load ./vector_pos_to_union_all_index
max_iter=10;%%������10��
mode=1;
max_iter_num=5;%% joint learning ������10��
start_iter=0;
% method='E-PLKL';
method='E-PLKL';
if strcmp(method,'E-PLKL')
ratio_cluster=5;%E-PLKL %result are saved in result_joint_se_cluster/5perPU
else
ratio_cluster=100;%PLKL  %result are saved in result_joint_se_cluster/100perPU
end
for each_class_num=5 %% number of default labeled samples per class default as 5
for va_iter=1:10    %% random times (from 1st time to 10th time)
% my_demo_joint_mean_spatial_indian_quick(va_iter,mode,each_class_num,max_iter_num,start_iter)
 my_demo_joint_mean_spatial_indian_quick_load_index(ratio_cluster,va_iter,mode,each_class_num,max_iter_num,start_iter,K_whole_mss,vector_pos_to_union_all_index)
end
end
end

