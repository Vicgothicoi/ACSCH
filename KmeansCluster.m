function [Label] = KmeansCluster(X1, X2)

num_clusters = 100; remainLabel = 15; sigma = 5;

Z = CCA(X1, X2);
[cluster_labels, centers] = kmeans(Z, num_clusters);
Label = sqdist(centers', Z');

for col = 1:size(Label, 2)
    col_data = Label(:, col);
    [~, sorted_idx] = sort(col_data, 'ascend');
    mask = zeros(size(col_data));
    mask(sorted_idx(1:remainLabel)) = 1; 
    Label(:, col) = col_data .* mask;

    % Handle Extreme values
    while any(Label(:, col) > 100)
       Label(:, col) = Label(:, col) / 2;  
    end
end

for col = 1:size(Label, 2)
    col_data = Label(:, col);
    non_zero_mask = col_data ~= 0;
    non_zero_values = col_data(non_zero_mask);

    % Skip all-zero columns
    if isempty(non_zero_values)
        continue;
    end
    
    % Compute numerators and denominator
    numerator = exp(-non_zero_values / sigma);
    denominator = sum(numerator);
    
    % Normalization
    Label(non_zero_mask, col) = numerator / denominator;
end

end







