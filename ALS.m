function [V] = ALS(L)

[labels, samples, ratings] = find(L);
m = max(labels); 
n = max(samples); 

k = 10;          
lambda = 0.1;    
max_iter = 10;            

U = 0.01 * randn(m, k); 
V = 0.01 * randn(n, k); 

label_cell = cell(m, 1);
for i = 1:m
    mask = (labels == i);
    label_cell{i} = struct(...
        'samples', samples(mask), ...
        'ratings', ratings(mask) ...
    );
end

sample_cell = cell(n, 1);
for j = 1:n
    mask = (samples == j);
    sample_cell{j} = struct(...
        'labels', labels(mask), ...
        'ratings', ratings(mask) ...
    );
end

for iter = 1:max_iter
    % Fix V, update U
    for i = 1:m
        data = label_cell{i};
        items_i = data.samples;
        ratings_i = data.ratings;
        
        if isempty(items_i)
            continue;
        end
        
        V_i = V(items_i, :);
        C = V_i' * V_i + lambda * eye(k);
        d = V_i' * ratings_i;
        U(i, :) = (C \ d)'; 
    end
    
    % Fix U, update V
    for j = 1:n
        data = sample_cell{j};
        users_j = data.labels;
        ratings_j = data.ratings;
        
        if isempty(users_j)
            continue;
        end
        
        U_j = U(users_j, :);
        C = U_j' * U_j + lambda * eye(k);
        d = U_j' * ratings_j;
        V(j, :) = (C \ d)';
    end
    
end