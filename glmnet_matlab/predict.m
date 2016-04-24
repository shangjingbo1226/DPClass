function [ accuracy ] = predict( train_file, test_file, pred_file )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
train = load(train_file);
test = load(test_file);
[m, n] = size(train);

train_x = train(:, 1:n - 1);
train_y = train(:, n);
test_x = test(:, 1:n - 1);
test_y = test(:, n);

options = glmnetSet;
%options.nlambda = n - 1;
options.alpha = 0;

fit = cvglmnet(train_x, train_y, 'binomial', options);
pred = glmnetPredict(fit.glmnet_fit, test_x, fit.lambda_min, 'response');

%fit = glmnet(train_x, train_y, 'binomial');
%pred = glmnetPredict(fit, test_x, fit.lambda(end), 'link');

m = length(pred);
for i = 1:m
    if pred(i) >= 0.5
        pred(i) = 1;
    else
        pred(i) = 0;
    end
end

dlmwrite(pred_file, pred, '\n');

hit = 0;
for i = 1:m
    if pred(i) == test_y(i)
        hit = hit + 1;
    end
end
accuracy = hit / m
exit;
end

