function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% X size: m * 2; theta size: 2 * 1; h size: m * 1;
h = X * theta;
% exclude theta_1 from reg
temp = theta;
temp(1) = 0;
J = (sum((h .- y) .^ 2)  + lambda * temp' * temp) / (2 * m);
grad = (X' * (h .- y) + lambda .* temp) / m;



% =========================================================================

grad = grad(:);

end
