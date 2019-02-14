function [ground_matrix,neigbor_matrix,neigbor_num_matrix,neigbor_pos_matrix,neigbor_pos_vectorid,Vector_pos_to_neigbor_vector_pos,Vector_pos_to_neigbor_num]=mean_spatial_kernel_consider0(img,window_width,GroundT)
%MEAN_SPATIAL_KERNEL_CONSIDER0 此处显示有关此函数的摘要
%   此处显示详细说明


%% tovector 后的img(i,j,:) 对应 id是 (j-1)*no_lines+i
[no_lines, no_rows, no_bands] =size(img);
mean_img=zeros(no_lines, no_rows, no_bands);
ground_matrix=zeros(no_lines,no_rows);
flag_matrix=zeros(no_lines,no_rows);
neigbor_num_matrix=zeros(no_lines,no_rows);
neigbor_matrix=zeros(no_lines,no_rows,window_width*window_width,no_bands);
neigbor_pos_matrix=zeros(no_lines,no_rows,window_width*window_width,2);
POS_to_Vector_id=zeros(no_lines,no_rows,2);
neigbor_pos_vectorid=zeros(no_lines,no_rows,window_width*window_width);
%% map img(i,j)tovector后的id对应的i,j位置
for i=1:no_lines
    for j=1:no_rows
        POS_to_Vector_id(i,j,1)=i;
        POS_to_Vector_id(i,j,2)=j;
    end
end
Vector_id_to_POS=ToVector(POS_to_Vector_id);
%% no_lines*(j-1)+i对应vector以后的id
for i=1:no_lines
    for j=1:no_rows
        id=(j-1)*no_lines+i;
        [DX]=find(GroundT(:,1)==id);
        [s1,s2]=size(DX);
%         if s1 && s2
%             ground_matrix(i,j)=GroundT(DX,2);
%             flag_matrix(i,j)=1;
%         end
        if s1 && s2
           ground_matrix(i,j)=GroundT(DX,2);
        end
        flag_matrix(i,j)=1;%% consider 0 就算label=0也算进去
    end
end
img2=img;
for i=1:no_lines
    for j=1:no_rows
        img2(i,j,:)=img(i,j,:)*flag_matrix(i,j);
    end
end
for i=1:no_lines
    for j=1:no_rows
        left_i=max(i-(window_width-1)/2,1);
        right_i=min(i+(window_width-1)/2,no_lines);
        top_j=max(j-(window_width-1)/2,1);
        bottom_j=min(j+(window_width-1)/2,no_rows);
        sub_matrix=img2(left_i:right_i,top_j:bottom_j,:);
        sub_flag=flag_matrix(left_i:right_i,top_j:bottom_j,:);
%         mean_img(i,j,:)=sum(sum(sub_matrix))/sum(sum(sub_flag));
        neigbor_num_matrix(i,j)=sum(sum(sub_flag));
        sub_matrix_v=ToVector(sub_matrix);
        [temp_sub_no,no_bands]=size(sub_matrix_v);
        sub_flag_v=ToVector_Flag(sub_flag);
        idx=[1:temp_sub_no].*sub_flag_v';
        
        idx_index=find(idx~=0);
        idx1=idx(idx_index); 
        neigbor_matrix(i,j,1:neigbor_num_matrix(i,j),:)=sub_matrix_v(idx1,:);
        sub_matrix_pos=zeros(neigbor_num_matrix(i,j),2);
        no_neigbor=0;
        for i1=left_i:right_i
            for j1=top_j:bottom_j
            if flag_matrix(i1,j1)
                no_neigbor=no_neigbor+1;
            sub_matrix_pos(no_neigbor,1)=i1;
            sub_matrix_pos(no_neigbor,2)=j1;
            neigbor_pos_matrix(i,j,no_neigbor,1)=i1;
            neigbor_pos_matrix(i,j,no_neigbor,2)=j1;
            neigbor_pos_vectorid(i,j,no_neigbor)= no_lines*(j1-1)+i1;
            end
            end
        end
        
    end
end
Vector_pos_to_neigbor_vector_pos=ToVector(neigbor_pos_vectorid);
Vector_pos_to_neigbor_num=ToVector_Flag(neigbor_num_matrix);
end





