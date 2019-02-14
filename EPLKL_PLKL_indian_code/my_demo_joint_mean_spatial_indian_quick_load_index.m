function [  ] = my_demo_joint_mean_spatial_indian_quick_load_index(ratio_cluster,va_iter,mode,each_class_num,max_iter_num,start_iter,K_whole_mss,vector_pos_to_union_all_index)
%MY_DEMO_JOINT_PAVIAU 此处显示有关此函数的摘要
addpath('../');

result_path3='./result_joint_se_cluster/';
% ratio_cluster=5;
result_path3=strcat(strcat(result_path3,num2str(ratio_cluster)),'perPU/');

%% set mode
%%
% mode=1; % no_train=each_class*no_classes
% % mode=2; % no_train=select ratio*100 percent of each class

mode_spatial=1;
mode_ck=1;
%% 

%%%% give the parameters
if mode==1
no_classes       = 16;
% each_class_num=40;
no_train         = round(each_class_num*no_classes);
file_name=strcat(num2str(each_class_num),'indian_svm');

end
%% start code add by me 
%%%


if mode_ck==1
    file_name=strcat(file_name,'_ck');
    file_name=strcat(result_path3,file_name);
end

%% end code add by me

data_type='indian';
load ./IndiaP.mat;
load ./Indian_pines_corrected.mat;
load ./Indian_pines_gt.mat;
result_path='./result/'
Data = indian_pines_corrected;  DataClass = indian_pines_gt;
%%%% estimate the size of the input image
[no_lines, no_rows, no_bands] = size(img);
img_w=img;
%% spatial kernel
if mode_spatial==1
 window_width=9;

end
%% 

GroundT=GroundT';


%%%% construct training and test datasets

% %%% random selection
if mode==1
indexes=train_test_random_new(GroundT(2,:),fix(no_train/no_classes),no_train);
ori_indexes=indexes;

end

%%  end add by me
%% test
%% end test
%%% get the training-test indexes
train_SL = GroundT(:,indexes);
test_SL = GroundT;
test_SL(:,indexes) = [];
[dim,ori_no_train]=size(train_SL);
train_labels= train_SL(2,:)';
test_labels = test_SL(2,:)';
ori_indexes=indexes;
no_test=size(test_labels,1);
ori_file_name=file_name;

%% 
if start_iter==0
[parameters,SVMResultTest_w_regu,SVMResultTest_s_regu,SVMResultTest_ws_regu,SVMResultTest2_w,SVMResultTest2_s,SVMResultTest2_mss,SVMResultTest_mss_regu,SVMResultTest_w_mss,SVMResultTest_w_mss_regu,SVMResultTest_ws,test_labels]= fix_train_update_kernel_idealKer(K_whole_mss,vector_pos_to_union_all_index,Data,DataClass,train_SL,test_SL);
GroudTest = double(test_labels(:,1));
[SVMOA_ws_regular,SVMAA_ws_regular,SVMkappa_ws_regular,SVMCA_ws_regular]=confusion(GroudTest,SVMResultTest_ws_regu)
[SVMOA_w_regular,SVMAA_w_regular,SVMkappa_w_regular,SVMCA_w_regular]=confusion(GroudTest,SVMResultTest_w_regu);
[SVMOA_s_regular,SVMAA_s_regular,SVMkappa_s_regular,SVMCA_s_regular]=confusion(GroudTest,SVMResultTest_s_regu)
[SVMOA2_s,SVMAA2_s,SVMkappa2_s,SVMCA2_s]=confusion(GroudTest,SVMResultTest2_s);
[SVMOA2_w,SVMAA2_w,SVMkappa2_w,SVMCA2_w]=confusion(GroudTest,SVMResultTest2_w);
[SVMOA_ws,SVMAA_ws,SVMkappa_ws,SVMCA_ws]=confusion(GroudTest,SVMResultTest_ws);

[SVMOA2_mss,SVMAA2_mss,SVMkappa2_mss,SVMCA2_mss]=confusion(GroudTest,SVMResultTest2_mss);
[SVMOA_mss_regular,SVMAA_mss_regular,SVMkappa_mss_regular,SVMCA_mss_regular]=confusion(GroudTest,SVMResultTest_mss_regu)
[SVMOA_w_mss,SVMAA_w_mss,SVMkappa_w_mss,SVMCA_w_mss]=confusion(GroudTest,SVMResultTest_w_mss);
[SVMOA_w_mss_regular,SVMAA_w_mss_regular,SVMkappa_w_mss_regular,SVMCA_w_mss_regular]=confusion(GroudTest,SVMResultTest_w_mss_regu)


file_name=strcat(strcat(strcat(ori_file_name,'iter_'),num2str(va_iter)),'_result.mat');

save(file_name,'indexes','train_labels','parameters','SVMOA_ws_regular','SVMAA_ws_regular','SVMkappa_ws_regular','SVMCA_ws_regular','SVMCA_ws','SVMOA_ws','SVMAA_ws','SVMkappa_ws','SVMCA_ws','SVMOA2_s','SVMAA2_s','SVMkappa2_s','SVMCA2_s','SVMOA_s_regular','SVMAA_s_regular','SVMkappa_s_regular','SVMCA_s_regular','SVMOA_w_regular','SVMAA_w_regular','SVMkappa_w_regular','SVMCA_w_regular','SVMOA2_w','SVMAA2_w','SVMkappa2_w','SVMCA2_w','SVMOA2_mss','SVMAA2_mss','SVMkappa2_mss','SVMCA2_mss','SVMOA_mss_regular','SVMAA_mss_regular','SVMkappa_mss_regular','SVMCA_mss_regular','SVMOA_w_mss','SVMAA_w_mss','SVMkappa_w_mss','SVMCA_w_mss','SVMOA_w_mss_regular','SVMAA_w_mss_regular','SVMkappa_w_mss_regular','SVMCA_w_mss_regular','SVMResultTest_ws_regu','SVMResultTest_w_regu','SVMResultTest_s_regu','SVMResultTest2_w','SVMResultTest2_s','SVMResultTest_ws','SVMResultTest2_mss','SVMResultTest_mss_regu','SVMResultTest_w_mss','SVMResultTest_w_mss_regu','GroudTest');
%% results of initial training data
old_parameter=parameters;
end
%%
%% test

joint_ws_w_s_regu_no_regu=[];

%% test end
GroudTest = double(test_labels(:,1));

for  iter_no=start_iter+1:max_iter_num
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[num_sampling,no_sub_per_iter,tol_samples,sub_samples_train,joint_ws_w_s_regu_no_regu,update_acc,added_train_index,add_index_inGroundT,predict_train_label] = fix_kernel_update_train_bagging_mean_ck(ratio_cluster,GroundT,test_SL,SVMResultTest_w_regu,SVMResultTest_s_regu,SVMResultTest_ws_regu,SVMResultTest2_w,SVMResultTest2_s,SVMResultTest_ws,SVMResultTest2_mss,SVMResultTest_mss_regu,SVMResultTest_w_mss,SVMResultTest_w_mss_regu,GroudTest,no_classes);
  sub_sample_joint_ws_w_s_regu_no_regu=sub_samples_train.joint_ws_w_s_regu_no_regu;
  sub_sample_added_select_train_index=sub_samples_train.added_select_train_index;
  sub_sample_add_se_index_inGroundT=sub_samples_train.add_se_index_inGroundT;
  sub_sample_predict_selected_train_label=sub_samples_train.predict_selected_train_label;
  

 SVMResultTest_w_regu_v=zeros(num_sampling,no_test);
SVMResultTest_s_regu_v=zeros(num_sampling,no_test);
SVMResultTest_ws_regu_v=zeros(num_sampling,no_test);
SVMResultTest2_w_v=zeros(num_sampling,no_test);
SVMResultTest2_s_v=zeros(num_sampling,no_test);
SVMResultTest_ws_v=zeros(num_sampling,no_test);
%% SVMResultTest_mss_regu_v SVMResultTest_mss_v SVMResultTest_wmss_v SVMResultTest_w_mss_regu_v


SVMResultTest_mss_regu_v=zeros(num_sampling,no_test);
SVMResultTest_mss_v=zeros(num_sampling,no_test);
SVMResultTest_wmss_v=zeros(num_sampling,no_test);
SVMResultTest_w_mss_regu_v=zeros(num_sampling,no_test);

SVMOA_ws_regular_v=zeros(num_sampling,1);
SVMAA_ws_regular_v=zeros(num_sampling,1);
SVMkappa_ws_regular_v=zeros(num_sampling,1);


SVMOA_w_regular_v=zeros(num_sampling,1);
SVMAA_w_regular_v=zeros(num_sampling,1);
SVMkappa_w_regular_v=zeros(num_sampling,1);


SVMOA_s_regular_v=zeros(num_sampling,1);
SVMAA_s_regular_v=zeros(num_sampling,1);
SVMkappa_s_regular_v=zeros(num_sampling,1);


SVMOA2_s_v=zeros(num_sampling,1);
SVMAA2_s_v=zeros(num_sampling,1);
SVMkappa2_s_v=zeros(num_sampling,1);


 SVMOA2_w_v=zeros(num_sampling,1);
SVMAA2_w_v=zeros(num_sampling,1);
SVMkappa2_w_v=zeros(num_sampling,1);


SVMOA_ws_v=zeros(num_sampling,1);
SVMAA_ws_v=zeros(num_sampling,1);
SVMkappa_ws_v=zeros(num_sampling,1);

SVMOA_wmss_v=zeros(num_sampling,1);
SVMAA_wmss_v=zeros(num_sampling,1);
SVMkappa_wmss_v=zeros(num_sampling,1);

SVMOA_mss_v=zeros(num_sampling,1);
SVMAA_mss_v=zeros(num_sampling,1);
SVMkappa_mss_v=zeros(num_sampling,1);

SVMOA_wmss_regular_v=zeros(num_sampling,1);
SVMAA_wmss_regular_v=zeros(num_sampling,1);
SVMkappa_wmss_regular_v=zeros(num_sampling,1);

SVMOA_mss_regular_v=zeros(num_sampling,1);
SVMAA_mss_regular_v=zeros(num_sampling,1);
SVMkappa_mss_regular_v=zeros(num_sampling,1);

update_acc_sample=zeros(num_sampling,1);
for no_sample=1:num_sampling   
   no_sub_sample=no_sub_per_iter(no_sample);
     add_se_index_inGroundT=sub_sample_add_se_index_inGroundT(no_sample,1:no_sub_sample);
     predict_selected_train_label=sub_sample_predict_selected_train_label(no_sample,1:no_sub_sample);
     added_select_train_index=sub_sample_added_select_train_index(no_sample,1:no_sub_sample);

add_se_index_inGroundT=add_se_index_inGroundT';
indexes=[ori_indexes;add_se_index_inGroundT];
[no_add_train,dim]=size(add_se_index_inGroundT);
train_SL(:,1:ori_no_train) = GroundT(:,ori_indexes);
train_SL(2,ori_no_train+1:ori_no_train+no_add_train)=predict_selected_train_label';

train_SL(1,ori_no_train+1:ori_no_train+no_add_train)=added_select_train_index;
update_acc_sample(no_sample,1)=compute_acc(GroundT,train_SL,add_se_index_inGroundT,added_select_train_index,ori_indexes,predict_selected_train_label);

train_labels= train_SL(2,:)';

[parameters,SVMResultTest_w_regu_v(no_sample,:),SVMResultTest_s_regu_v(no_sample,:),SVMResultTest_ws_regu_v(no_sample,:),SVMResultTest2_w_v(no_sample,:),SVMResultTest2_s_v(no_sample,:),SVMResultTest_mss_v(no_sample,:),SVMResultTest_mss_regu_v(no_sample,:),SVMResultTest_wmss_v(no_sample,:),SVMResultTest_w_mss_regu_v(no_sample,:),SVMResultTest_ws_v(no_sample,:),test_labels]= fix_train_update_kernel_idealKer(K_whole_mss,vector_pos_to_union_all_index,Data,DataClass,train_SL,test_SL);

[SVMOA_ws_regular_v(no_sample),SVMAA_ws_regular_v(no_sample),SVMkappa_ws_regular_v(no_sample),SVMCA_ws_regular]=confusion(GroudTest,SVMResultTest_ws_regu_v(no_sample,:)')
[SVMOA_w_regular_v(no_sample),SVMAA_w_regular_v(no_sample),SVMkappa_w_regular_v(no_sample),SVMCA_w_regular]=confusion(GroudTest,SVMResultTest_w_regu_v(no_sample,:)');
[SVMOA_s_regular_v(no_sample),SVMAA_s_regular_v(no_sample),SVMkappa_s_regular_v(no_sample),SVMCA_s_regular]=confusion(GroudTest,SVMResultTest_s_regu_v(no_sample,:)')
[SVMOA2_s_v(no_sample),SVMAA2_s_v(no_sample),SVMkappa2_s_v(no_sample),SVMCA2_s]=confusion(GroudTest,SVMResultTest2_s_v(no_sample,:)');
[SVMOA2_w_v(no_sample),SVMAA2_w_v(no_sample),SVMkappa2_w_v(no_sample),SVMCA2_w]=confusion(GroudTest,SVMResultTest2_w_v(no_sample,:)');
[SVMOA_ws_v(no_sample),SVMAA_ws_v(no_sample),SVMkappa_ws_v(no_sample),SVMCA_ws]=confusion(GroudTest,SVMResultTest_ws_v(no_sample,:)');

[SVMOA_wmss_regular_v(no_sample),SVMAA_wmss_regular_v(no_sample),SVMkappa_wmss_regular_v(no_sample),SVMCA_wms_reg]=confusion(GroudTest,SVMResultTest_w_mss_regu_v(no_sample,:)');
[SVMOA_wmss_v(no_sample),SVMAA_wmss_v(no_sample),SVMkappa_wmss_v(no_sample),SVMCA_wms]=confusion(GroudTest,SVMResultTest_wmss_v(no_sample,:)');
[SVMOA_mss_regular_v(no_sample),SVMAA_mss_regular_v(no_sample),SVMkappa_mss_regular_v(no_sample),SVMCA_ms_regu]=confusion(GroudTest,SVMResultTest_mss_regu_v(no_sample,:)');
[SVMOA_mss_v(no_sample),SVMAA_mss_v(no_sample),SVMkappa_mss_v(no_sample),SVMCA_ms]=confusion(GroudTest,SVMResultTest_mss_v(no_sample,:)');


 end
[ Pred, F_pre] = ensemble_learning_add_meanck(no_classes,num_sampling,GroudTest,SVMResultTest_w_regu_v,SVMResultTest_s_regu_v,SVMResultTest_ws_regu_v,SVMResultTest2_w_v,SVMResultTest2_s_v,SVMResultTest_ws_v,SVMResultTest_w_mss_regu_v,SVMResultTest_wmss_v,SVMResultTest_mss_regu_v,SVMResultTest_mss_v);
[ result_ws_r ] = asign_result( SVMOA_ws_regular_v,SVMAA_ws_regular_v,SVMkappa_ws_regular_v );
[ result_ws ] = asign_result( SVMOA_ws_v,SVMAA_ws_v,SVMkappa_ws_v );
[ result_s_r ] = asign_result( SVMOA_s_regular_v,SVMAA_s_regular_v,SVMkappa_s_regular_v );
[ result_s] = asign_result( SVMOA2_s_v,SVMAA2_s_v,SVMkappa2_s_v );
[ result_w] = asign_result( SVMOA2_w_v,SVMAA2_w_v,SVMkappa2_w_v );
[ result_w_r] = asign_result( SVMOA_w_regular_v,SVMAA_w_regular_v,SVMkappa_w_regular_v );

[result_wms_r]=asign_result(SVMOA_wmss_regular_v,SVMAA_wmss_regular_v,SVMkappa_wmss_regular_v);
[result_wms]=asign_result(SVMOA_wmss_v,SVMAA_wmss_v,SVMkappa_wmss_v);
[result_ms]=asign_result(SVMOA_mss_v,SVMAA_mss_v,SVMkappa_mss_v);
[result_ms_r]=asign_result(SVMOA_mss_regular_v,SVMAA_mss_regular_v,SVMkappa_mss_regular_v);


SVMResultTest_w_regu=Pred.w_regu;
SVMResultTest_s_regu=Pred.s_regu;
SVMResultTest_ws_regu=Pred.ws_regu;
SVMResultTest2_w=Pred.w;
SVMResultTest2_s=Pred.s;
SVMResultTest_ws=Pred.ws;

SVMResultTest2_mss=Pred.ms;
SVMResultTest_mss_regu=Pred.ms_regu;
SVMResultTest_w_mss=Pred.wms;
SVMResultTest_w_mss_regu=Pred.wms_regu;

[SVMOA_ws_regular,SVMAA_ws_regular,SVMkappa_ws_regular,SVMCA_ws_regular]=confusion(GroudTest,SVMResultTest_ws_regu)
[SVMOA_w_regular,SVMAA_w_regular,SVMkappa_w_regular,SVMCA_w_regular]=confusion(GroudTest,SVMResultTest_w_regu);
[SVMOA_s_regular,SVMAA_s_regular,SVMkappa_s_regular,SVMCA_s_regular]=confusion(GroudTest,SVMResultTest_s_regu)
[SVMOA2_s,SVMAA2_s,SVMkappa2_s,SVMCA2_s]=confusion(GroudTest,SVMResultTest2_s);
[SVMOA2_w,SVMAA2_w,SVMkappa2_w,SVMCA2_w]=confusion(GroudTest,SVMResultTest2_w);
[SVMOA_ws,SVMAA_ws,SVMkappa_ws,SVMCA_ws]=confusion(GroudTest,SVMResultTest_ws);

[SVMOA2_mss,SVMAA2_mss,SVMkappa2_mss,SVMCA2_mss]=confusion(GroudTest,SVMResultTest2_mss);
[SVMOA_mss_regular,SVMAA_mss_regular,SVMkappa_mss_regular,SVMCA_mss_regular]=confusion(GroudTest,SVMResultTest_mss_regu)
[SVMOA_w_mss,SVMAA_w_mss,SVMkappa_w_mss,SVMCA_w_mss]=confusion(GroudTest,SVMResultTest_w_mss);
[SVMOA_w_mss_regular,SVMAA_w_mss_regular,SVMkappa_w_mss_regular,SVMCA_w_mss_regular]=confusion(GroudTest,SVMResultTest_w_mss_regu)



if va_iter>=10
    va_iter=mod(va_iter,10);
end
file_name=strcat(strcat(strcat(strcat(ori_file_name,'iter_'),num2str(va_iter)),num2str(iter_no)),'_joint_once_meanck_result.mat');
save(file_name,'ori_indexes','Pred', 'F_pre','result_ws_r','result_ws','result_s_r','result_s','result_w','result_w_r','result_wms_r','result_wms','result_ms','result_ms_r','SVMResultTest_w_regu_v','SVMResultTest_s_regu_v','SVMResultTest_ws_regu_v','SVMResultTest2_w_v','SVMResultTest2_s_v','SVMResultTest_ws_v','SVMResultTest_mss_regu_v','SVMResultTest_mss_v','SVMResultTest_wmss_v', 'SVMResultTest_w_mss_regu_v','joint_ws_w_s_regu_no_regu','update_acc','added_train_index','add_index_inGroundT','num_sampling','no_sub_sample','tol_samples','sub_samples_train','ratio_cluster','sub_sample_added_select_train_index','sub_sample_add_se_index_inGroundT','sub_sample_predict_selected_train_label','update_acc_sample','indexes','train_SL','parameters','SVMOA_ws_regular','SVMAA_ws_regular','SVMkappa_ws_regular','SVMCA_ws_regular','SVMCA_ws','SVMOA_ws','SVMAA_ws','SVMkappa_ws','SVMCA_ws','SVMOA2_s','SVMAA2_s','SVMkappa2_s','SVMCA2_s','SVMOA_s_regular','SVMAA_s_regular','SVMkappa_s_regular','SVMCA_s_regular','SVMOA_w_regular','SVMAA_w_regular','SVMkappa_w_regular','SVMCA_w_regular','SVMOA2_w','SVMAA2_w','SVMkappa2_w','SVMCA2_w','SVMResultTest_ws_regu','SVMResultTest_w_regu','SVMResultTest_s_regu','SVMResultTest2_w','SVMResultTest2_s','SVMResultTest_ws','SVMResultTest2_mss','SVMResultTest_mss_regu','SVMResultTest_w_mss','SVMResultTest_w_mss_regu','SVMOA2_mss','SVMAA2_mss','SVMkappa2_mss','SVMCA2_mss','SVMOA_mss_regular','SVMAA_mss_regular','SVMkappa_mss_regular','SVMCA_mss_regular','SVMOA_w_mss','SVMAA_w_mss','SVMkappa_w_mss','SVMCA_w_mss','SVMOA_w_mss_regular','SVMAA_w_mss_regular','SVMkappa_w_mss_regular','SVMCA_w_mss_regular','GroudTest');
%% results of augmented training data

end

