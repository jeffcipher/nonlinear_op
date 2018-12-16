function [w,lossHistory] = gdLS(maxit,x,y,w,lambda)

    m = length(y);
    lossHistory = zeros(maxit,1);
    gradient =@(w) (1/m)*(x'*(x*w - y));
    beta = 0.6;
    
    for i = 1: maxit
         loss = cost(x,y,w,lambda);
         lossHistory(i) = loss;
        temp = x * w;
        error = temp - y;
        newX = error' * x;
        
        alpha = 1;
        
        update = w - (alpha/m)*(x'*(x*w-y));
        a = cost(x,y,update,lambda);
        b = cost(x,y,w,lambda) - (alpha)*0.4*norm(gradient(w))^2;
        
        while(a > b)
            
            alpha = alpha*beta;
            update = w - (alpha/m)*(x'*(x*w-y));
            a = cost(x,y,update,lambda);
            b = cost(x,y,w,lambda) -(alpha)*0.4*norm(gradient(w))^2;
        end
        
      fprintf("Iter:%d loss:%e  |g|%e  alpha:%e \n",i,cost(x,y,w,lambda),norm(gradient(w)),alpha);
         w = w - (alpha/m)*(x'*(x*w-y));
        
         if(loss <= 10^-7 || loss == 0 || norm(gradient(w)) <= 10^-4)
                   
             break; 
          end
        
    end

end

