function [w, lossHistory] = gd(maxit,x,y,w,alpha,lambda)
    m = length(y);
    lossHistory = zeros(maxit,1);
    
    for i = 1: maxit
        loss = cost(x,y,w,lambda);
        lossHistory(i) = loss;
        temp = x * w;
        error = temp - y;
        newX = error' * x;
        
        w = w*(1-((alpha*lambda)/m)) - ((alpha/m)*((temp-y)'*x)');
        if(loss <= 10^-7 || loss == 0 || norm(gradient(w)) <= 10^-8)
           break; 
        end
       
    end
end

