function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%polyX = [X(:, 1:2) X(:, 3) .^ 2];

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


    h = X * theta;
    % theta = theta - alpha * (sum((h - y) .* X))' / m;
    theta = theta - alpha * (X' * (h - y)) / m;
    %h = polyX * theta;
    %theta = theta - alpha * (polyX' * (polyX * theta - y)) / m;


    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);

    % if (rem(iter, 100) == 0)
    %    disp('iter:'), disp(iter), disp('theta:'), disp(theta), disp('J:'), disp(J_history(iter));
        % fprintf('iter=%d, theta=%f, J=%f\n', iter, theta, J_history(iter));
    % endif
end

end
