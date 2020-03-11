%% VI. Training the Network
%function output = train(input,target,nHidLayers,nNeurons,iterations)

load mnist_all.mat

input = double(train0(1,:))'; % must be column vect, converted to double <- uint8
target = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0];
nHidLayers = 5; 
nNeurons = 30;
weights = initialize(nHidLayers, nNeurons);
eta = 0.001; % training rate coeff, elem of [0.01, 0.1]
% iterations = 100; 
% 
% for n = 1:iterations
    
x,output = multiLayerNetwork(input, weights, nHidLayers); % forward pass
err = target - output;
deltaOut = zeros(size(output)); % preallocate

for i=1:length(output) % d/dx sigmoidal * error
deltaOut(i) = f(output(i)).*(1-f(output(i))).*(err(i));
end

delta = cell(size(weights));
delta = [delta {deltaOut}];

for i = length(weights):-1:1
    delta{i} = zeros(size(x{i})); % preallocate
    
    delta{i} = sum(weights{i}'*delta{i+1})*...
        (f(x{i}).*(1-f(x{i})));

    weights{i} = eta * delta{i} .* x{i};
end    
% end

disp('Network''s guess:');disp(output)



% local fxns
function OUT = f(NET)
OUT = 1./(1 + exp(-NET)); % Sigmoidal activation function
%OUT = OUT';
end