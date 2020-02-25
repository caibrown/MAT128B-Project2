%% MAT 128B WQ 2020 Project 2
%  Caitlin Brown, Shuai Zhi

%% I MNIST Database

load mnist_all.mat

digit = train0(1,:);
digitImage = reshape(digit,28,28);
image(rot90(flipud(digitImage),-1)),
colormap(gray(256)), axis square tight off;