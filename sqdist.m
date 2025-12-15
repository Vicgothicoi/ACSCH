function d = sqdist(a,b)
% 计算两个矩阵中所有列向量对的欧氏距离（向量2范数）平方，返回一个大小为 [size(a,2) × size(b,2)] 的矩阵
% SQDIST - computes squared Euclidean distance matrix
%          computes a rectangular matrix of pairwise distances
% between points in A (given in columns) and points in B

% NB: very fast implementation taken from Roland Bunschoten

aa = sum(a.^2,1); bb = sum(b.^2,1); ab = a'*b; 
d = (repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);

