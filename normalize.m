function [x, mu, sigma] = normalize(x)
    
    mu = mean(x);
    sigma = std(x);
    x = (x - mu)./sigma;
end

