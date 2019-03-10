function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

m = size(X, 1);
indexes = zeros(m);

idx = 1;
while (idx <= K)
  i = randi(m);
  if (indexes(i) == 0)
    centroids(idx, :) = X(i, :);
    idx++;
    indexes(i) = 1;
  endif
end



% =============================================================

end

