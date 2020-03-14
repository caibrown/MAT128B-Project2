%% VI. testing the Network
load mnist_all.mat

samples = 890; % how many samples from each digit data-frame
selectSamples = randi(890,samples,1);
inputCell = {...
    double(test0(selectSamples,:))',...
    double(test1(selectSamples,:))',...
    double(test2(selectSamples,:))',...
    double(test3(selectSamples,:))',...
    double(test4(selectSamples,:))',...
    double(test5(selectSamples,:))',...
    double(test6(selectSamples,:))',...
    double(test7(selectSamples,:))',...
    double(test8(selectSamples,:))',...
    double(test9(selectSamples,:))',...
    };

targetMatrix = eye(10);
guess = zeros(10);
nHidLayers = 5; 
nNeurons = 50;
weights = initialize(nHidLayers, nNeurons);
eta = 0.01;

for k=1:10
    inputMatrix = inputCell{k}; % select matrix
    target = targetMatrix(:,k);

    for j=1:samples
        input = inputMatrix(:,j); % cycle thru cols
    
        [x,output] = multiLayerNetwork(input, weights, nHidLayers);
        err = target - output;
        deltaOut = zeros(size(output)); % preallocate

        for i=1:length(output) % d/dx sigmoid * error
            deltaOut(i) = f(output(i)).*(1-f(output(i))).*(err(i));
        end

        delta = cell(size(weights));
        delta(end) = {deltaOut};

        for i = length(weights):-1:2
   
            delta{i-1} = zeros(size(x{i-1})); % preallocate
            delta{i-1} = sum(weights{i}'*delta{i})*...
                (f(x{i-1}).*(1-f(x{i-1})));

            weights{i} = weights{i} + eta*delta{i}*x{i}';
        end    
    end
    
    guess(:,k) = output;
end

disp('~Results~')
for l=1:10
    disp('Target:');disp(l-1)
    disp('Network''s guess:');disp(guess(:,l))
end

% local fxns
function weights = initialize(nHidLayers, nNeurons)

weights = cell(1, nHidLayers + 1);
weights(1) = {rand(nNeurons,784)-0.5};
weights(2:(end-1)) = {rand(nNeurons,nNeurons)-0.5};
weights(end) = {rand(10,nNeurons)-0.5};

end

function [x, output] = multiLayerNetwork(input, weights, nHidLayers)
x = cell(1,nHidLayers+2);
x(1) = {input};

    for i=1:(nHidLayers+1)
    x(i+1) = {computeNET(x{i}, weights{i})};
    x(i+1) = {f(x{i+1})};
    end
    
output = x{end};
end

function NET = computeNET(inputs, inputWeights)
NET = inputWeights*inputs;
end

function OUT = f(NET)
OUT = 1./(1 + exp(-NET)); % Sigmoidal activation function
%OUT = OUT';
end
