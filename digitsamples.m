load mnist_all.mat

zero = train0(1:7,:);
one = train1(1:7,:);
two = train2(1:7,:);
three = train3(1:7,:);
four = train4(1:7,:);
five = train5(1:7,:);
six = train7(1:7,:);
seven = train7(1:7,:);
eight = train8(1:7,:);
nine = train9(1:7,:);
numbers = [zero; one; two; three;...
    four; five; six; seven; eight; nine];

for i=1:length(numbers)
subplot(5,14,i)
pics = reshape(numbers(i,:),28,28);
image(rot90(flipud(pics),-1)),
colormap(gray(256)), axis square tight off;
end
