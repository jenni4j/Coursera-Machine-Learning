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


% Part 1: Feedforward 

% add column of 1s to X

X = [ones(m,1) X];

% compute a^l for all layer units

a1 = X; 

z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];

z3 = a2*Theta2'; 
a3 = sigmoid(z3);

% matrix that classifies each training example by what digit they are via '1' or '0'

result_y = zeros(size(y, 1),num_labels);
for i = 1:m
    result_y(i, y(i)) = 1;
end

% regularized cost function
cost = result_y.*log(a3) + (1-result_y).*log(1-a3);
J = -sum(sum(cost, 2))/m;

regularization = sum(sum(Theta1(:,2:size(Theta1,2)).^2)) + sum(sum(Theta2(:,2:size(Theta2,2)).^2));

J = J + lambda/(2*m)*regularization;

% Part 2: Backprop

% create matrices to store partial derivatives

grad1 = zeros(size(Theta1));
grad2 = zeros(size(Theta2));

for i = 1:m,
	bpa1 = X(i,:)';

	bpz2 = Theta1*bpa1;
	bpa2 = sigmoid(bpz2);
	bpa2 = [1; bpa2];

	bpz3 = Theta2*bpa2;
	bpa3 = sigmoid(bpz3);

	delta3 = bpa3 - result_y(i, :)';
	delta2 = (Theta2'*delta3)(2:size(Theta2,2),1).*sigmoidGradient(bpz2);
% had to resize result of Theta2'*delta3 cause there was some flipping duck going on 

	grad1 = grad1 + delta2*bpa1';
	grad2 = grad2 + delta3*bpa2';
end

% Part 3: regularization

Theta1_grad = grad1/m + lambda*[zeros(hidden_layer_size,1) Theta1(:,2:size(Theta1,2))]/m;

Theta2_grad = grad2/m + lambda*[zeros(num_labels,1) Theta2(:,2:size(Theta2,2))]/m;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
