function [w,lossHistory] = cg(maxit,x,y,w)

    disp("Running PRP");
    m = length(y);
    lossHistory = zeros(maxit,1);
    B = eye(5);
    g =@(w) (1/m)*(x'*(x*w - y));
    H = (1/m)*x'*x;
    wo = -1;
    wc = w;
    pc = -1;
    po = -1;
    gc = g(wc);
    go = -1;
    loss = -1;
    
    for i = 0 : maxit
      
       if(i == 0)
         
          pc = -g(wc);
          po = -g(wc);
          
       else
            
            b = (g(wc)'*(g(wc)-g(wo)))/(g(wo)'*g(wo));
           % b= (gc'*(gc - (gc/go)*gc))/(norm(go)^2 + norm(gc'*po));
            temp = pc;
            pc = -g(wc) + b*pc;
            po = temp;
       end
       
     alpha = (-g(wc)'*pc)/(pc'*H*pc);

       
       
       
       loss = cost(x,y,wc,0);
       lossHistory(i+1) = loss;

       
       fprintf("Iter:%d loss:%e |g|:%e alpha:%e\n", i,loss,norm(g(wc)),alpha);
       
       wtemp = wc;
       wc = wc + alpha * pc;
       wo = wtemp;
       
       gtemp = gc;
       gc= g(wc);
       go = gtemp;
        if(norm(gc) < 10^-4 )
             break;
        end
        
    end
end

