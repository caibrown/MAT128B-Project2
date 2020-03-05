function weights = initialize(nHidLayers, nNeurons)

weights = cell(1, nHidLayers + 1);% weights: nHidLayers + 1 cell array with each 
                                  % element a matrix of dimension input by output
% initialize weights with random small values
weights{1} = rand(784,nNeurons);
weights(1,2:(end-1)) = {rand(nNeurons,nNeurons)};
weights{end} = rand(nNeurons,10); % output layer must have 10 values (0-9)

end
