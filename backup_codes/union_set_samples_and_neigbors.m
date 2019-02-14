function [union_train_and_neigbors]=union_set_samples_and_neigbors(sample_pos_to_neigbor_vector_pos,sample_vector_index)

%UNION_SET_SAMPLES_AND_NEIGBORS 此处显示有关此函数的摘要
%   此处显示详细说明

[num_samples,dim]=size(sample_pos_to_neigbor_vector_pos);
union_train_and_neigbors=[];
for i=1:num_samples
    union_train_and_neigbors=union(sample_pos_to_neigbor_vector_pos(i,1:sample_vector_index(i)),union_train_and_neigbors);
end
end

