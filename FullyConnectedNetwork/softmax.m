function a = softmax(Z)
% Compute Softmax function
a=exp(Z)./sum(exp(Z));
end

