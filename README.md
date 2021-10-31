# MAT128B-Project2
Backpropagation neural network: MNIST digit recognition.

For MAT128B, numerical linear algebra @ UC Davis, WQ '20.

* Project2_MAT128B.pdf describes project requirements.

* mnist_all.mat is the database of handwritten digits.

* digitsamples.m visualizes some of these data.

* i-iii.m answers questions 1-3 in the pdf, i.e. loads data, visualizes the mean of all digits 0-9, and details 'one neuron' with the sigmoidal activation function.

* iiia.m utilizes MATLAB's Symbolic Math Toolbox to take the derivative of the sigmoid function.

* initialize.m initializes all weight layers for a variable number of layers and neurons per layer.

* multiLayerNetwork.m feeds forward

* train.m and test.m train and test the network respectively, and return its average multi-class guess for a given target 0-9 in the form of a probability distribution corresponding to digits 0-9.
