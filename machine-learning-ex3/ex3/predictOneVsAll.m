function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels
%are in the range 1..K, where K = size(all_theta, 1).
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples)

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the
%       max element, for more information see 'help max'. If your examples
%       are in rows, then, you can use max(A, [], 2) to obtain the max
%       for each row.
%

% X size = m, n + 1
% all_theta size = num_labels, n + 1

A = zeros(num_labels, m); % A size = num_labels, m

%disp(size(X));
%disp(size(all_theta));

for c = 1:num_labels
  theta = all_theta(c, :);
  c_predict = sigmoid(X * theta'); % c_predict size = m, 1
  %disp(size(c_predict));
  %disp(c_predict);
  A(c, :) = c_predict';
endfor

[w, iw] = max(A); % iw size = 1, m
p = iw';

%disp('iw size'), disp(size(iw));
%disp('w'), disp(w(1, 1:10)), disp(w(1, 501:510));
%disp('iw'), disp(iw(1, 1:10)), disp(iw(1, 501:510));


% =========================================================================


end
