function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% X size: m, n
% input1 size: m, n+1
% Theta1 size: t1r, n+1
% Theta2 size: t2r, t1r+1
input1 = [ones(m, 1) X];
%disp('input1 size'), disp(size(input1));
%disp('Theta1 size'), disp(size(Theta1));
%disp('Theta2 size'), disp(size(Theta2));

% output1 size: m, t1r
output1 = sigmoid(input1 * Theta1');
input2 = [ones(m, 1) output1];

% output2 size: m, t2r
output2 = sigmoid(input2 * Theta2');

%disp('output2 size'), disp(size(output2));
[w, iw] = max(output2');
p = iw';


% =========================================================================


end
