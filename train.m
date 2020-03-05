%% VI. Training the Network

load mnist_all.mat

input = double(train0(1,:))'; % must be column vector, converted to double from uint8
target = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0];
nHidLayers = 2; 
nNeurons = 20;
weights = initialize(nHidLayers, nNeurons);
output = multiLayerNetwork(input, weights, nHidLayers);
err = target - output
