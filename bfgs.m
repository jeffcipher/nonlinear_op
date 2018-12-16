function [w,lossHistory] = bfgs(maxit,x,Y,w)
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

    for i = 1 : maxit
    
         loss = cost(x,Y,w,0);
         lossHistory(i) = loss;
        
         p = B \ -gradient(w);

        alpha = (-gradient(w)'*p)/(p'*H*p);
        update = w + alpha*p;
         
         wn = w + alpha*p;
         s = wn - w;
         y = gradient(wn) - gradient(w);
        % fprintf("y's:%e\n",y'*s);
       
         if(y'*s > 0)
             B = B - (1/(s'*B*s))*(B*s*s'*B) + (1/(y'*s))*(y*y');
         end
         
         w= wn;

    end
end

