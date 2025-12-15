function [Z] = CCA(X1, X2)

X1_samples = X1';  % n × d1
X2_samples = X2';  

X1_normalized = zscore(X1_samples);  
X2_normalized = zscore(X2_samples);  

[A, B, r, U, V] = canoncorr(X1_normalized, X2_normalized);

cumulative_r = cumsum(r)/sum(r);
k = find(cumulative_r > 0.15, 1);  

U_k = U(:, 1:k);  % n×k
V_k = V(:, 1:k); 
A_k = A(:, 1:k); 
B_k = B(:, 1:k);

mu = 0.5; %(mu = 0.2 when using IAPR-TC12)
Z = mu * U_k + (1 - mu) * V_k;