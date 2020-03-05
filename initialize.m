load mnist_all.mat

input = double(train0(1,:))'; % must be column vector, converted to double from uint8
nHidLayers = 2; 
weights = cell(1, nHidLayers + 1);% weights: nHidLayers + 1 cell array with each 
                                  % element a matrix of dimension input by output
nNeurons = 20; % arbitrary number of neurons per hidden layer

% initialize weights with random small values
weights{1} = rand(784,nNeurons);
weights(1,2:(end-1)) = {rand(nNeurons,nNeurons)};
weights{end} = rand(nNeurons,10); % output layer must have 10 values (0-9)

function output = multiLayerNetwork(input, weights, nHidLayers)
x = input;

    for i=1:(nHidLayers + 1)
    x = computeNET(x, weights{i});
    x = f(x);
    end
    
output = x;  
end

% Local fxns:
function NET = computeNET(inputs, inputWeights)
NET = inputWeights*inputs;
end

function OUT = f(NET)
OUT = 1/(1 + exp(-NET)); % Sigmoidal activation function
OUT = OUT';
end
