function  [KLL_mss,KUL_mss]= compute_meanspatial_spatialkernel_quick(K_whole_mss,vector_pos_to_union_all_and_neigbors_index,train_vector_index,test_vector_index)
%COMPUTE_MEANSPATIAL_SPATIALKERNEL_QUICK 此处显示有关此函数的摘要
%   此处显示详细说明
% file_name='K_whole_mss_';
% file_name=strcat(file_name,data_type);
% load(file_name);
% sigma_s = Gcv_s;
% rbfKernel_s = @(X,Y) exp(-sigma_s .* pdist2(X,Y,'euclidean').^2);

train_vector_index_to_union_all_and_neigbors_index=vector_pos_to_union_all_and_neigbors_index(train_vector_index);
test_vector_index_to_union_all_and_neigbors_index=vector_pos_to_union_all_and_neigbors_index(test_vector_index);
KLL_mss=K_whole_mss(train_vector_index_to_union_all_and_neigbors_index,train_vector_index_to_union_all_and_neigbors_index);
KUL_mss=K_whole_mss(test_vector_index_to_union_all_and_neigbors_index,train_vector_index_to_union_all_and_neigbors_index);

end

