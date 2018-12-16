function [loss] = cost(X,y,w,lambda)
   m = length(y);
   loss =  (1 / (2*m) )*( sum(((X * w)-y).^2) +  (lambda)*(w'*w)) ;
end

