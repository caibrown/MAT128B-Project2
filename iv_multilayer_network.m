%% IV. Multilayer Network

load mnist_all.mat

input = test0;
w1 = rand(size(input)) % weights mapping input layer to hidden layer
%weights = {w1,w2}

function output = multiLayerNetwork(input, weights)

function NET = computeNET(input, w1)
NET = sum(input.*w1)
end

