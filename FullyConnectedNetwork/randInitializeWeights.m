function [W , b]= randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
W = zeros(L_out, L_in);
b = zeros(L_out, 1);
epsilon_init = 0.12;
W = rand(L_out, L_in) * 2 * epsilon_init-epsilon_init;
b=rand(L_out, 1)*2*epsilon_init-epsilon_init;
end
