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
OUT = 1./(1 + exp(-NET)); % Sigmoidal activation function
%OUT = OUT';
end
