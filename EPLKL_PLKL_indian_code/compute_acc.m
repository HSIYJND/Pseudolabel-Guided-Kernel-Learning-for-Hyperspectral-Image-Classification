function [acc]=compute_acc(GroundT,train_SL,add_se_index_inGroundT,added_select_train_index,ori_indexes,predict_selected_train_label)

%COMPUTE_ACC �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��


gt=GroundT(2,add_se_index_inGroundT);
cor_indexes=find(gt==predict_selected_train_label);
acc=size(cor_indexes,2)/size(add_se_index_inGroundT,1);
end

