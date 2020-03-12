function [x, output] = multiLayerNetwork(input, weights, nHidLayers)
x = cell(1,nHidLayers+2);
x(1) = {input};

    for i=1:(nHidLayers+1)
    x(i+1) = {computeNET(x{i}, weights{i})};
    x(i+1) = {f(x{i+1})};
    end
    
output = x{end};
end

% Local fxns:
function NET = computeNET(inputs, inputWeights)
NET = inputWeights*inputs;
end

function OUT = f(NET)
OUT = 1./(1 + exp(-NET)); % Sigmoidal activation function
%OUT = OUT';
end
