function [ v ] = ToVector_Flag( flag_m )
%TOVECTOR_FLAG �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

sz = size(flag_m);
v = reshape(flag_m, [prod(sz(1:2)) 1] );
end

