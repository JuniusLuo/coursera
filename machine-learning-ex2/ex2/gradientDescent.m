function [theta, cost] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    z = X * theta;
    h = sigmoid(z);
    theta = theta - alpha * (X' * (h - y)) / m;


    % ============================================================

    % Save the cost J in every iteration
    cost = - (y' * log(h) + (1 .- y)' * log(1 .- h)) / m;

    if (rem(iter, 100) == 0)
        disp('iter:'), disp(iter), disp('theta:'), disp(theta), disp('J:'), disp(cost);
        %fprintf('iter=%d, theta=%f, J=%f\n', iter, theta, J_history(iter));
    endif


end

end
