function [h_theta] = FuncTheta(w2,w3,w4,b2,b3,b4,X, p)
% given NN params calculates h_tehta
    switch p 
        case 'softmax'
        a1=X';
        z2=w2*a1+b2; 
        a2=sigmoid(z2);
        z3=w3*a2+b3;
        a3=sigmoid(z3);
        z4=w4*a3+b4;
        a4=softmax(z4);
        h_theta=a4;
        
        otherwise

        a1=X';
        z2=w2*a1+b2;
        a2=sigmoid(z2); 
        z3=w3*a2+b3;
        a3=sigmoid(z3);
        z4=w4*a3+b4;
        a4=sigmoid(z4);
        h_theta=a4;
    end

end

