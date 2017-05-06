function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
test_C = [0 0.03 0.1 0.3 1 3 10 30];
test_sigma = [0 0.03 0.1 0.3 1 3 10 30];

kernelFunction = 'gaussianKernel';
for i = 1:length(test_C)
    for j = 1:length(test_sigma)
        model= svmTrain(X, y, test_C(i), @(x1, x2) gaussianKernel(x1, x2, test_sigma(j)));
        prediction = svmPredict(model, Xval);
        prediction_error(i,j) = mean(double(prediction ~= yval));
    end
end
prediction_error = prediction_error';
prediction_error = prediction_error(:);
[b, indx] = min(prediction_error);
ind1 = int16(ceil(indx/length(test_C)));
ind2 = int16(length(test_sigma)*((indx/length(test_C)) - floor(indx/length(test_C))));
C = test_C(ind1);
sigma = test_sigma(ind2);




% =========================================================================

end
