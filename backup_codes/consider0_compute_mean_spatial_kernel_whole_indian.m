function [] = consider0_compute_mean_spatial_kernel_whole_indian()
%addpath('../../../revisedTGRS2013_EPF&IFR');

%addpath('../../my_add_code');
c_max=2;


%% set mode
%%
mode=1; % no_train=each_class*no_classes

mode_spatial=1;


%%%% give the parameters
if mode==1
no_classes       = 16;
each_class_num=15;

no_train         = round(each_class_num*no_classes);

end
%% start code add by me 
%%%




%% end code add by me

load('./IndiaP.mat');
%%%% estimate the size of the input image
[no_lines, no_rows, no_bands] = size(img);
img_w=img;
%% spatial kernel
if mode_spatial==1
 window_width=9;

 [ground_matrix,neigbor_matrix,neigbor_num_matrix,neigbor_pos_matrix,neigbor_pos_vectorid,Vector_pos_to_neigbor_vector_pos,Vector_pos_to_neigbor_num]= mean_spatial_kernel_consider0(img,window_width,GroundT);

end
%% 
%%%% vectorization

img_w=ToVector(img_w);

GroundT=GroundT';


ori_img_w=img_w;



[nr nDim] = size(img_w);  
load Indian_pines_corrected.mat;
load Indian_pines_gt.mat;
Data = indian_pines_corrected;  DataClass = indian_pines_gt;
[nr nc nDim] = size(Data);  nAll = nr*nc;
gt2 = reshape(DataClass, nAll, 1);
tmp = unique(gt2);  classLabel = tmp(tmp~=0); nClass = length(classLabel);
data2 = reshape(Data, nAll, nDim);
clear Data* indian*
% L2 normalize
img_wL2 = img_w./ repmat(sqrt(sum(img_w.*img_w,2)),[1 nDim]); % L2 norm

all_indexes=GroundT(1,:);
all_indexes=sort(all_indexes,'ascend');
[dim,no_whole_samples]=size(all_indexes);

load Indian_pines_corrected.mat;
load Indian_pines_gt.mat;
% Parameter setting
Cck = 1e4;  sigma1 = 0.25;  sigma2 = 2;  % SVM-CK

bestSig = 0.0625;
rbfKernel_s = @(X,Y) exp(-sigma_s .* pdist2(X,Y,'euclidean').^2);
all_pos_to_neigbor_vector_pos=Vector_pos_to_neigbor_vector_pos(all_indexes,:);
all_Vector_pos_to_neigbor_num=Vector_pos_to_neigbor_num(all_indexes);
union_all_and_neigbors=union_set_samples_and_neigbors(all_pos_to_neigbor_vector_pos,all_Vector_pos_to_neigbor_num);
samples_union_all_and_neigbors=img_wL2(union_all_and_neigbors,:);

K_union_all_neigbors_to_union_all_neigbors = calckernel('rbf', bestSig, samples_union_all_and_neigbors, samples_union_all_and_neigbors);

vector_pos_to_union_all_and_neigbors_index=val_to_index_map(union_all_and_neigbors);
K_whole_mss=zeros(no_whole_samples,no_whole_samples);
vector_pos_to_union_all_index=val_to_index_map(all_indexes');
save('vector_pos_to_union_all_index.mat','vector_pos_to_union_all_index');
%% compute K_whole_mss
for i=1:no_whole_samples
    for j=1:no_whole_samples
%         i
%         j
        index1=all_indexes(i);
        index2=all_indexes(j);
        neigbor_indexes_index1= Vector_pos_to_neigbor_vector_pos(index1,:);
        neigbor_indexes_index2= Vector_pos_to_neigbor_vector_pos(index2,:);
        neigbor_num_index1=Vector_pos_to_neigbor_num(index1);
        neigbor_num_index2=Vector_pos_to_neigbor_num(index2);

        union_all_and_neigbors_index1=vector_pos_to_union_all_and_neigbors_index(neigbor_indexes_index1(1:neigbor_num_index1));
        union_all_and_neigbors_index2=vector_pos_to_union_all_and_neigbors_index(neigbor_indexes_index2(1:neigbor_num_index2));
        sub_train_samples1_samples2=K_union_all_neigbors_to_union_all_neigbors(union_all_and_neigbors_index1,union_all_and_neigbors_index2);
        val3=sum(sum(sub_train_samples1_samples2));

        K_whole_mss(i,j)= val3/(neigbor_num_index1*neigbor_num_index2);
    end
end
save('K_whole_mss_indianbestSig_consider0.mat','K_whole_mss','vector_pos_to_union_all_and_neigbors_index');

end

