function [num_sampling,no_sub_per_iter,tol_samples,sub_samples_train,joint_ws_w_s_regu_no_regu,update_acc,added_train_index,add_index_inGroundT,predict_train_label] = fix_kernel_update_train_bagging_mean_ck(ratio_cluster,GroundT,test_SL,SVMResultTest_w_regu,SVMResultTest_s_regu,SVMResultTest_ws_regu,SVMResultTest2_w,SVMResultTest2_s,SVMResultTest_ws,SVMResultTest2_mss,SVMResultTest_mss_regu,SVMResultTest_w_mss,SVMResultTest_w_mss_regu,GroudTest,no_classes)
%FIX_KERNEL_UPDATE_TRAIN_SE_CLUSTER 此处显示有关此函数的摘要
%   此处显示详细说明
%% 当样本大量时 我们考虑类似于bagging的方法解决




%% ensemble kws-IR kw-IR ks-IR kmss-IR Kw_mss-IR predict result
[inter_ws_s_regu,values]=find(SVMResultTest_ws_regu==SVMResultTest_s_regu);
[inter_ws_w_regu,values]=find(SVMResultTest_ws_regu==SVMResultTest_w_regu);
[joint_ws_s_w_regu]=intersect(inter_ws_s_regu,inter_ws_w_regu);

[inter_w_mss_mss_regu,values]=find(SVMResultTest_w_mss_regu==SVMResultTest_mss_regu);
[inter_w_mss_w_regu,values]=find(SVMResultTest_w_mss_regu==SVMResultTest_w_regu);
[joint_wmss_mss_w_regu]=intersect(inter_w_mss_mss_regu,inter_w_mss_w_regu);

[joint_ws_regus_wmss_regus]=intersect(joint_ws_s_w_regu,joint_wmss_mss_w_regu);


%% ensemble kws kw ks kmss Kw_mss predict result
[inter_ws_s,values]=find(SVMResultTest_ws==SVMResultTest2_s);
[inter_ws_w,values]=find(SVMResultTest_ws==SVMResultTest2_w);
[joint_ws_s_w]=intersect(inter_ws_s,inter_ws_w);

[inter_w_mss_mss,values]=find(SVMResultTest_w_mss==SVMResultTest2_mss);
[inter_w_mss_w,values]=find(SVMResultTest_w_mss==SVMResultTest2_w);
[joint_wmss_mss_w]=intersect(inter_w_mss_mss,inter_w_mss_w);

[joint_ws_wmss_origins]=intersect(joint_ws_s_w,joint_wmss_mss_w);
%% 

%% ensemble kws-IR kw-IR ks-IR kmss-IR Kw_mss-IR  kws kw ks kmss Kw_mss predict result
[joint_ws_w_s_regu_no_regu]=intersect(joint_ws_regus_wmss_regus,joint_ws_wmss_origins);

[inter_truth_ws_regu,values]=find(SVMResultTest_ws_regu==GroudTest);
joint_ws_w_s_regu_noregu_truth=intersect(joint_ws_w_s_regu_no_regu,inter_truth_ws_regu);

update_acc=size(joint_ws_w_s_regu_noregu_truth,1)/size(joint_ws_w_s_regu_no_regu,1);





added_train_index=test_SL(1,joint_ws_w_s_regu_no_regu);
predict_train_label=SVMResultTest_ws_regu(joint_ws_w_s_regu_no_regu);
acc_no2=0;
add_index_inGroundT=zeros(size(added_train_index,2),1);
for i=1:size(added_train_index,2)
    index_id=added_train_index(i);
    predict_train_label(i)
    [id_no]=find(GroundT(1,:)==index_id);
    add_index_inGroundT(i)=id_no;
    if predict_train_label(i)==GroundT(2,id_no)
        acc_no2=acc_no2+1;
    end
end
update_acc=acc_no2/size(joint_ws_w_s_regu_no_regu,1)

notice='finish update_train'
[num_samples,dim]=size(predict_train_label);
class_dis_vector=zeros(no_classes,1);
class_indexes_vector=zeros(no_classes,num_samples);

for class =1:no_classes
    indexes=find(predict_train_label(:)==class);
    class_indexes_vector(class,1:size(indexes,1))=indexes;
    class_dis_vector(class)=size(indexes,1);
end
[tol_samples]=size(joint_ws_w_s_regu_no_regu,1);
num_sampling=round(100/ratio_cluster);
 no_sub_sample=floor(tol_samples*ratio_cluster/100);
max_dim=max(tol_samples-(num_sampling-1)*no_sub_sample,no_sub_sample);
sub_sample_joint_ws_w_s_regu_no_regu=zeros(num_sampling,max_dim);
sub_sample_added_select_train_index=zeros(num_sampling,max_dim);
sub_sample_add_se_index_inGroundT=zeros(num_sampling,max_dim);
sub_sample_predict_selected_train_label=zeros(num_sampling,max_dim);
no_sub_per_iter=zeros(num_sampling,1);
for i=1:num_sampling-1
    sec_indexes=[];
    for class =1:no_classes
        sec_no=floor(class_dis_vector(class)*ratio_cluster/100);
        sec_indexes=[sec_indexes  class_indexes_vector(class,(i-1)*sec_no+1:i*sec_no)];
    end

     no_sub_sample=size(sec_indexes,2);
     no_sub_per_iter(i)=no_sub_sample;
     sub_sample_joint_ws_w_s_regu_no_regu(i,1:no_sub_sample)=joint_ws_w_s_regu_no_regu(sec_indexes);
     sub_sample_added_select_train_index(i,1:no_sub_sample)=added_train_index(sec_indexes);
     sub_sample_add_se_index_inGroundT(i,1:no_sub_sample)=add_index_inGroundT(sec_indexes);
     sub_sample_predict_selected_train_label(i,1:no_sub_sample)=predict_train_label(sec_indexes);
end
i=num_sampling;
sec_indexes=[];
no_remain=tol_samples-(num_sampling-1)*no_sub_sample;
no_sub_per_iter(i)=no_remain;

     for class =1:no_classes
        sec_no=floor(class_dis_vector(class)*ratio_cluster/100);
        sec_indexes=[sec_indexes  class_indexes_vector(class,(i-1)*sec_no+1:class_dis_vector(class))];
    end
     no_remain=size(sec_indexes,2);

     sub_sample_joint_ws_w_s_regu_no_regu(i,1:no_remain)=joint_ws_w_s_regu_no_regu(sec_indexes);
     sub_sample_added_select_train_index(i,1:no_remain)=added_train_index(sec_indexes);
     sub_sample_add_se_index_inGroundT(i,1:no_remain)=add_index_inGroundT(sec_indexes);
     sub_sample_predict_selected_train_label(i,1:no_remain)=predict_train_label(sec_indexes);
     
sub_samples_train.joint_ws_w_s_regu_no_regu=sub_sample_joint_ws_w_s_regu_no_regu;
sub_samples_train.added_select_train_index=sub_sample_added_select_train_index;
sub_samples_train.add_se_index_inGroundT=sub_sample_add_se_index_inGroundT;
sub_samples_train.predict_selected_train_label=sub_sample_predict_selected_train_label;
end


