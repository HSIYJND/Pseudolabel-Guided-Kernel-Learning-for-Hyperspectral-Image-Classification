function [ v ] = ToVector_Flag( flag_m )
%TOVECTOR_FLAG 此处显示有关此函数的摘要
%   此处显示详细说明

sz = size(flag_m);
v = reshape(flag_m, [prod(sz(1:2)) 1] );
end

