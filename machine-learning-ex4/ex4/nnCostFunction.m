function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% step1: forward propagation and compute J
a1 = [ones(m, 1) X]; % a1 size: m, input_layer_size+1
z2 = a1 * Theta1'; % Theta1 size: hidden_layer_size, input_layer_size+1

a2 = [ones(size(z2), 1) sigmoid(z2)]; % a2 size: m, hidden_layer_size+1
z3 = a2 * Theta2'; % Theta2 size: num_labels, hidden_layer_size+1

a3 = sigmoid(z3); % a3 size: m, num_labels

% convert y to matrice, y_matrice size: m, num_labels
y_matrice = zeros(size(a3));
for i = 1:m
    y_matrice(i, y(i)) = 1;
endfor

% J regularization, need to remove the first theta in Theta1 and Theta2,
% as the first theta is for the added 1 in each layer.
J_reg = lambda * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2))) / (2 * m);
J = - sum(sum(y_matrice .* log(a3) + (1 .- y_matrice) .* log(1 .- a3))) / m + J_reg;


% step2: backpropagation + regularization
delta3 = a3 - y_matrice; % delta3 size: m, num_labels

% delta2 excludes the first column, as it is for the bias of the hidden layer
delta2 = ((delta3 * Theta2) .* a2 .* (1 - a2))(:, 2:end);

Delta2 = delta3' * a2;
Delta1 = delta2' * a1;

% exclude the first column in Theta2, as regularization is not needed for the bias
Theta2_reg = lambda / m * [zeros(size(Theta2), 1) Theta2(:, 2:end)];
Theta1_reg = lambda / m * [zeros(size(Theta1), 1) Theta1(:, 2:end)];

Theta2_grad = Delta2 / m + Theta2_reg; % Theta2_grad size: num_labels, hidden_layer_size+1
Theta1_grad = Delta1 / m + Theta1_reg;

% Explanations:
%
%  L1    Theta1    L2     Theta2     L3
% Input -- a1 --> Hidden -- a2 --> Output -> a3
%
% Chain rule: grad(J/Theta2) = grad(J/a3) * grad(a3/z3) * grad(z3/Theta2) + grad(J_reg/Theta2)
%             grad(J/a3) = sum(sum(y_matrice / a3 - (1 .- y_matrice) / (1 .- a3))) / m;  % a constant value
%             grad(a3/z3) = a3 .* (1 .- a3);  % size: m, num_labels
%             grad(J/a3) * grad(a3/z3) = (a3 - y_matrice) / m;  % size: m, num_labels
%             grad(z3/Theta2) = a2;  % size: m, hidden_layer_size+1
% so grad(J/Theta2) = Delta2 * a2 / m + reg
%
% Chain rule: grad(J/Theta1) = grad(J/a2) * grad(a2/z2) * grad(z2/Theta1) + grad(J_reg/Theta1)
%             grad(J/a2) = grad(J/a3) * grad(a3/z3) * grad(z3/a2)
%                        = delta3 * Theta2 / m
%             grad(a2/z2) = a2 .* (1 .- a2)
%             grad(J/a2) * grad(a2/z2) = delta2 / m
%             grad(z2/Theta1) = a1
% so grad(J/Theta1) = Delta1 / m + reg


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
