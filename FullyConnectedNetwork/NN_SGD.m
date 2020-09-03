function [J_train, J_val, Accu, F1, Accu_train,F1_train, w2,w3,w4,b2,b3,b4 ]= NN_SGD (hidden_layer_size, lambda, learningRate, X_train, Y_train, X_val, Y_val, CostFunction,initialize)
clear w2 w3 w4 b2 b3 b4;
input_layer=size(X_train,2);
num_labels=size(Y_train,2);
%weigths initialization
switch initialize
    case 'Xavier'
        [w2,b2]=XavierInitializeWeights(input_layer, hidden_layer_size);
        [w3,b3]=XavierInitializeWeights(hidden_layer_size, hidden_layer_size);
        [w4,b4]=XavierInitializeWeights(hidden_layer_size, num_labels);
        
    otherwise        
        [w2,b2]=randInitializeWeights(input_layer, hidden_layer_size);
        [w3,b3]=randInitializeWeights(hidden_layer_size, hidden_layer_size);
        [w4,b4]=randInitializeWeights(hidden_layer_size, num_labels);
end

%other parameters initialization
Nepoch=5;
minibatchsize=50;
evalIter=4000;
J_val=zeros(26,1);
J_train=zeros(26,1);
Accu=zeros(26,1);
F1=zeros(26,1);
Accu_train=zeros(26,1);
F1_train=zeros(26,1);
nn=1;
%Starting training
for i=1:Nepoch
    fprintf('epoch number : %d \n',i);
    for j=1:minibatchsize:20950
        n=j+49;
        %Compute Partial derivatives
        [Cost, Db2, Db3, Db4, Dw2, Dw3, Dw4]=ComputeGrad(w2,w3,w4,b2,b3,b4,X_train(j:n,:),Y_train(j:n,:),CostFunction,lambda);
        fprintf('iteration number: %d       Cost: %d \n',n/50, Cost);
        %Weigth initialization
        w2=w2*(1-learningRate*lambda/minibatchsize)-(learningRate/minibatchsize)*Dw2;
        w3=w3*(1-learningRate*lambda/minibatchsize)-(learningRate/minibatchsize)*Dw3;
        w4=w4*(1-learningRate*lambda/minibatchsize)-(learningRate/minibatchsize)*Dw4;
        b2=b2-(learningRate/minibatchsize)*Db2;
        b3=b3-(learningRate/minibatchsize)*Db3;
        b4=b4-(learningRate/minibatchsize)*Db4;
        %Each 4000 image check validation set
        if n==nn*evalIter-(i-1)*20000
            [J_val(nn), F1(nn), Accu(nn)]=TestNN(w2,w3,w4,b2,b3,b4,X_val, Y_val,CostFunction);
            [J_train(nn), F1_train(nn), Accu_train(nn)]=TestNN(w2,w3,w4,b2,b3,b4,X_train(j:n,:), Y_train(j:n,:),CostFunction);
            
            nn=nn+1;
        end
            

    end
end

%show final result 
fprintf('Fin del entrenamiento')
[J_val(nn), F1(nn), Accu(nn)]=TestNN(w2,w3,w4,b2,b3,b4,X_val, Y_val, CostFunction);
[J_train(nn), F1_train(nn), Accu_train(nn)]=TestNN(w2,w3,w4,b2,b3,b4,X_train(size(X_train,1)-minibatchsize:size(X_train,1),:), Y_train(size(Y_train,1)-minibatchsize:size(Y_train,1),:),CostFunction);
fprintf('valor de Coste final:        %d \n', J_val(nn));
fprintf('valor de F1 score final:     %d \n', F1(nn));
fprintf('valor de Accuracy final:     %d \n', Accu(nn));
end
