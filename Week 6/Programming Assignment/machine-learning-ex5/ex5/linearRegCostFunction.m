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

h_x=X*theta;
J=(1/(2*m))* sum((h_x-y).^2) + (lambda/(2*m))*sum(theta(2:end).^2);
grad(1) = (1/m)*(X(:,1)'*(h_x-y)); 
grad(2:end) = (1/m)*(X(:,2:end)'*(h_x-y)) + (lambda/m)*theta(2:end);


% m = length(y); % number of training examples

% h_theta = X * theta;

% J = 1/(2*m) * (h_theta - y)' * (h_theta - y) + ...
%     (lambda/(2*m)) * (theta(2:length(theta)))' * theta(2:length(theta));

% thetaZero = theta;
% thetaZero(1) = 0;

% grad = ((1 / m) * (h_theta - y)' * X) + ...
%     lambda / m * thetaZero';

grad = grad(:);

end
