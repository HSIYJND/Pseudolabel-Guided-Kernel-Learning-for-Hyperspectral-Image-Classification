function [val_to_index]=val_to_index_map(union_train_and_neigbors)

%VAL_TO_INDEX_MAP 此处显示有关此函数的摘要
%   此处显示详细说明
[no_samples,dim]=size(union_train_and_neigbors);
max_val=max(union_train_and_neigbors);
val_to_index=zeros(max_val,1);

for i=1:no_samples
    val=union_train_and_neigbors(i);
    val_to_index(val)=i;
end

end

