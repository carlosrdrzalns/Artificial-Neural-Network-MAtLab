function [Cost,Dbias2,Dbias3,Dbias4,Dweight2,Dweight3,Dweight4] = ComputeGrad(w2,w3,w4,b2,b3,b4,X,Y,s,lambda)


%compute feedforward propagation to get the activation function
%depending on v , this function will compute gradient for w & b using MSE ('MSE'),
%crossentropy ('cross-entropy') cost function, softmax output layer ('softmax'), 
m=size(X,1);
Cx=zeros(m,1);
switch s
    case 'MSE'
        parfor i=1:m %compute for each example of the minibatch the BP algorithm
            %first we are going to compute the feedforward
            a1=X(i,:)';
            z2=w2*a1+b2; %first we calculate the weighted input
            a2=sigmoid(z2); %apply the sigmoid function to each Z
            z3=w3*a2+b3;
            a3=sigmoid(z3);
            z4=w4*a3+b4;
            a4=sigmoid(z4);
            Cx(i)=(norm(Y(i,:)'-a4))^2;
            %we calculate now the errors for each layer using BP
            delta4=(a4-Y(i,:)').*sigmoidGradient(z4);%error for the outputlayer
            delta3=(w4'*delta4).*sigmoidGradient(z3);%calculate the error for each of the other layers
            delta2=(w3'*delta3).*sigmoidGradient(z2); 
            Db4(:,:,i)=delta4;
            Db3(:,:,i)=delta3;
            Db2(:,:,i)=delta2;
            Dw4(:,:,i)=delta4*a3';
            Dw3(:,:,i)=delta3*a2';
            Dw2(:,:,i)=delta2*a1';
        end
        %Calculamos el coste
        Cost=sum(Cx)/(2*m)+(lambda/(2*m))*sum([w2(:);w3(:);w4(:)].^2);
        %calculamos el gradiente para las bias

        Dbias4=sum(Db4,3)/m;
        Dbias3=sum(Db3,3)/m;
        Dbias2=sum(Db2,3)/m;
        %Calculamos el gradiente para los weights
        Dweight4=sum(Dw4,3)/m;
        Dweight3=sum(Dw3,3)/m;
        Dweight2=sum(Dw2,3)/m;
    
    case 'cross-entropy'
        parfor i=1:m %compute for each example of the minibatch the BP algorithm
            %first we are going to compute the feedforward
            a1=X(i,:)';
            z2=w2*a1+b2; %first we calculate the weighted input
            a2=sigmoid(z2); %apply the sigmoid function to each Z
            z3=w3*a2+b3;
            a3=sigmoid(z3);
            z4=w4*a3+b4;
            a4=sigmoid(z4);
            Cx(i)=-sum(Y(i,:)'.*log(a4)+(1-Y(i,:)').*log(1-a4))
            %we calculate now the errors for each layer using BP
            delta4=(a4-Y(i,:)');%error for the outputlayer
            delta3=(w4'*delta4).*sigmoidGradient(z3);%calculate the error for each of the other layers
            delta2=(w3'*delta3).*sigmoidGradient(z2); 
            Db4(:,:,i)=delta4;
            Db3(:,:,i)=delta3;
            Db2(:,:,i)=delta2;
            Dw4(:,:,i)=delta4*a3';
            Dw3(:,:,i)=delta3*a2';
            Dw2(:,:,i)=delta2*a1';
        end
        %Calculamos el coste
        Cost=sum(Cx)/(m)+(lambda/(2*m))*sum([w2(:);w3(:);w4(:)].^2);
        %calculamos el gradiente para las bias

        Dbias4=sum(Db4,3)/m;
        Dbias3=sum(Db3,3)/m;
        Dbias2=sum(Db2,3)/m;
        %Calculamos el gradiente para los weights
        Dweight4=sum(Dw4,3)/m;
        Dweight3=sum(Dw3,3)/m;
        Dweight2=sum(Dw2,3)/m;
    case 'softmax'
      try
          parfor i=1:m
            a1=X(i,:)';
            z2=w2*a1+b2;
            a2=sigmoid(z2);
            z3=w3*a2+b3;
            a3=sigmoid(z3);
            z4=w4*a3+b4;
            a4=softmax(z4);
            [~,I]=max(Y(i,:)');
            Cx(i)=-log(a4(I));
            delta4=(a4-Y(i,:)');%error for the outputlayer
            delta3=(w4'*delta4).*sigmoidGradient(z3);%calculate the error for each of the other layers
            delta2=(w3'*delta3).*sigmoidGradient(z2); 
            Db4(:,:,i)=delta4;
            Db3(:,:,i)=delta3;
            Db2(:,:,i)=delta2;
            Dw4(:,:,i)=delta4*a3';
            Dw3(:,:,i)=delta3*a2';
            Dw2(:,:,i)=delta2*a1';
          end
      catch e
          fprintf(e);
          
      end
        
        %Calculamos el coste
        Cost=sum(Cx)/(m) +(lambda/(2*m))*sum([w2(:);w3(:);w4(:)].^2);
        %calculamos el gradiente para las bias

        Dbias4=sum(Db4,3)/m;
        Dbias3=sum(Db3,3)/m;
        Dbias2=sum(Db2,3)/m;
        %Calculamos el gradiente para los weights
        Dweight4=sum(Dw4,3)/m;
        Dweight3=sum(Dw3,3)/m;
        Dweight2=sum(Dw2,3)/m;
end

