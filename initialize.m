function weights = initialize(nHidLayers, nNeurons)

weights = cell(1, nHidLayers + 1);
weights(1) = {rand(nNeurons,784)-0.5};
weights(2:(end-1)) = {rand(nNeurons,nNeurons)-0.5};
weights(end) = {rand(10,nNeurons)-0.5};

end
