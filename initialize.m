function weights = initialize(nHidLayers, nNeurons)

weights = cell(1, nHidLayers + 1);
weights(1) = {rand(nNeurons,784)};
weights(1,2:(end-1)) = {rand(nNeurons,nNeurons)};
weights(end) = {rand(10,nNeurons)};

end
