%% VI. Training the Network
load mnist_all.mat

samples = 1500; % how many samples from each digit data-frame
selectSamples = randi(5400,samples,1);
inputCell = {...
    double(train0(selectSamples,:))',...
    double(train1(selectSamples,:))',...
    double(train2(selectSamples,:))',...
    double(train3(selectSamples,:))',...
    double(train4(selectSamples,:))',...
    double(train5(selectSamples,:))',...
    double(train6(selectSamples,:))',...
    double(train7(selectSamples,:))',...
    double(train8(selectSamples,:))',...
    double(train9(selectSamples,:))',...
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

% %% Test
% testCell = {...
%     double(test0(1:500,:))',...
%     double(test1(1:500,:))',...
%     double(test2(1:500,:))',...
%     double(test3(1:500,:))',...
%     double(test4(1:500,:))',...
%     double(test5(1:500,:))',...
%     double(test6(1:500,:))',...
%     double(test7(1:500,:))',...
%     double(test8(1:500,:))',...
%     double(test9(1:500,:))',...
%     };
% 
% testGuessMean = zeros(10);
% 
% for k=1:10
%     testMatrix = testCell{k};
%     testGuess = zeros(10,500);  
%     for j=1:500
% 
%         testDig = testMatrix(:,j);
%         [~, testGuess(:,j)] = multiLayerNetwork(testDig, weights, nHidLayers);
% 
%     end
%     
%     testGuessMean(:,k) = mean(testGuess,2);
% end
% 
% disp('~Test Results~')
% for l=1:10
%     disp('Digit:');disp(l-1)
%     disp('Network''s guess:');disp(testGuessMean(:,l))
% end

% local fxns
function OUT = f(NET)
OUT = 1./(1 + exp(-NET));
end