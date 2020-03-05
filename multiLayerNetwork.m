function output = multiLayerNetwork(input, weights, nHidLayers)
% weights: nHidLayers + 1 cell array with each element a matrix of
% dimension input by output

% input must be column vector, converted to double from uint8
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
OUT = 1./(1 + exp(-NET)); % Sigmoidal activation function
%OUT = OUT';
end
