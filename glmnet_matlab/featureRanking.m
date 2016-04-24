function [ ranking ] = featureRanking( train_file, output_file)
values = load(train_file);
data = sparse(values(:, 1), values(:, 2), values(:, 3));
%data = values;
[m, n] = size(data)
x = data(:, 1:n-1);
y = data(:, n);
rank = zeros(1, n - 1);
for i = 1:n-1
    rank(i) = n;
end

options = glmnetSet;
%options.nlambda = n - 1;
options.alpha = 1;

fit = glmnet(x, y, 'binomial', options);
weights = fit.beta;
[m, lambda] = size(weights)

size(nonzeros(weights))

count = 0;
for i = 1:lambda
    if count >= m
        break;
    end
    for j = 1:m
        if abs(weights(j, i)) > 0 && rank(j) == n
			%fit.lambda(i);
            rank(j) = count;
            count = count + 1;
        end            
    end
end
count
%pred = glmnetPredict(fit, x, fit.lambda(i - 1), 'link');
[tmp, ranking] = sort(rank);
csvwrite(output_file, ranking - 1);
exit;
end

