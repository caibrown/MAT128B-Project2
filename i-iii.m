%% MAT 128B WQ 2020 Project 2
%  Using algebraic methods for optimization:
%  Backpropagation neural networks
%  Caitlin Brown, Shuai Zhi

%% I. MNIST Database

load mnist_all.mat

digit = train0(1,:);
digitImage = reshape(digit,28,28);
image(rot90(flipud(digitImage),-1)),
colormap(gray(256)), axis square tight off;

%% II. Plot Digits

T = zeros(10,784); % Create matrix T with proper dimensionality

T(1,:) = mean(train0);
T(2,:) = mean(train1);
T(3,:) = mean(train2);
T(4,:) = mean(train3);
T(5,:) = mean(train4);
T(6,:) = mean(train5);
T(7,:) = mean(train6);
T(8,:) = mean(train7);
T(9,:) = mean(train8);
T(10,:) = mean(train9);

for i = 1:10
    subplot(2,5,i)
    avgPic = T(i,:);
    avgPicImage = reshape(avgPic,28,28);
    image(rot90(flipud(avgPicImage),-1)),
    colormap(gray(256)), axis square tight off;
end

%% III. A Neuron

% inputs=[1 2]; inputWeights=[3 4];      % test data
% NET=computeNET(inputs,inputWeights);
% OUT=f(NET)

function NET = computeNET(inputs, inputWeights)
NET = sum(inputs.*inputWeights)
end

function OUT = f(NET)
OUT = 1/(1 + exp(-NET)) % Sigmoidal activation function
end
