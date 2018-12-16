function [w,lossHistory] = bfgsBT(maxit,x,Y,w)
    %Y is the target variable
    %y is g(w_{k + 1}) - g(w)
    %Note: Loss function is defined with respect to w, L(w)
    %Note: the regularization parameter is zero
    %To add regularization term: set H = (1/m)*(x'*x + lambda)
    
    m = length(Y);
    lossHistory = zeros(maxit,1);
    B = eye(5);
    gradient =@(w) (1/m)*(x'*(x*w - Y));
    H = x'*x;
    beta = 0.5;    
    for i = 1 : maxit
    
          
         loss = cost(x,Y,w,0);
         lossHistory(i) = loss;
        
        
         p = B \ -gradient(w);

        alpha = 1;
        update = w + alpha*p;
        a = cost(x,Y,update,0);
        b = cost(x,Y,w,0) + alpha*0.5*gradient(w)'*p;
        
        while(a > b)
            
            alpha = alpha*beta;
            update = w + alpha*p;
            a = cost(x,Y,update,0);
            
            b = cost(x,Y,w,0) + alpha*0.5*gradient(w)'*p;
        end
         
         
         wn = w + alpha*p;
         s = wn - w;
         y = gradient(wn) - gradient(w);
        % fprintf("y's:%e\n",y'*s);te
       
        
         if(y'*s > 0)
             B = B - (1/(s'*B*s))*(B*s*s'*B) + (1/(y'*s))*(y*y');
         end
         
         w= wn;
         fprintf("Iter:%d loss:%e |g|:%e\n", i,loss,norm(gradient(w)));
         
    end
end

