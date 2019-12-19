
# NeuralNetwork

Simple Back Propagation Neural Network (BPN)

This code is a fork from [Bobby Anguelov's NeuralNetwork](https://github.com/BobbyAnguelov/NeuralNetwork).

The code makes use of [Florian Rappl's command parser](https://github.com/FlorianRappl/CmdParser )

# Disclaimer
This code is meant to be a simple implementation of the back-propagation neural network discussed in the tutorial below:

[https://takinginitiative.wordpress.com/2008/04/03/basic-neural-network-tutorial-theory/](https://takinginitiative.wordpress.com/2008/04/03/basic-neural-network-tutorial-theory/)

[https://takinginitiative.wordpress.com/2008/04/23/basic-neural-network-tutorial-c-implementation-and-source-code/](https://takinginitiative.wordpress.com/2008/04/23/basic-neural-network-tutorial-c-implementation-and-source-code/)


# Project overview

This project contains three programs :

 - `trainPBN` initializes and trains BPN (native C++)
 - `evalNN` evaluates a BPN on given entries (native C++).
 - a graphic interface for the visualization of a BPN (HTML5 + JS).


# Compile and Run with Gnu/Linux

Using Synaptic Manager, the simplest way to compile is to use `cmake`

```
sudo apt-get update
sudo apt-get install g++ make cmake
```

## Compilation
```
mkdir build
cd build
cmake ..
make
```

### Debug VS Release Mode

Since the training of a neural network requires a lot of computational power, one might find useful to compile a debug and a release versions (the release version should run significantly faster):
```
mkdir debug && cd debug && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j
cd ..
mkdir release && cd release && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
```

# Compile and Run Under Windows

TODO



# Typical Usage
Let's train a Network using data from the
``Example/threeShapes/data/threeShapes.csv`` file. In this example, the input
are binary pictures of size 40x40. Consequently, there needs to be 1600 input
neurons on the first layer. The output layer must have size 3 since the inputs
are classified in three categories :

1. Straight line (first output neuron).
2. Rectangle (second output neuron).
3. Triangles (third output neuron).

Here we choose to have 3 hidden layers of size 20, so the "-l" argument is
"1600,20,20,20,3". Finally, we stop the learning once our network has obtained
at least 90% accuracy on the generalization set.

```
./trainBPN -d ../Example/threeShapes/data/threeShapes.csv -l 1600,20,20,20,3 -a 90 -e myTrainedNN
Input file: ../Example/threeShapes/data/threeShapes.csv
Read complete: 100000 inputs loaded

 Neural Network Training Starting: 
==========================================================================
 LR: 0.01, Momentum: 0.9, Max Epochs: 100
 1600 Input Neurons, 3 Hidden Layers, 3 Output Neurons
==========================================================================

Epoch :0 Training Set Accuracy:25.0967%, MSE: 0.187347 Generalization Set Accuracy:25.62%, MSE: 0.182734
Epoch :1 Training Set Accuracy:26.87%, MSE: 0.159222 Generalization Set Accuracy:25.725%, MSE: 0.143087
Epoch :2 Training Set Accuracy:24.225%, MSE: 0.137596 Generalization Set Accuracy:23.925%, MSE: 0.131987
Epoch :3 Training Set Accuracy:24.0633%, MSE: 0.128191 Generalization Set Accuracy:26.735%, MSE: 0.122087
Epoch :4 Training Set Accuracy:32.175%, MSE: 0.115479 Generalization Set Accuracy:41.26%, MSE: 0.10896
Epoch :5 Training Set Accuracy:44.265%, MSE: 0.100699 Generalization Set Accuracy:57.36%, MSE: 0.0884957
[...]
```

Note that in the above example, the trained network is exported to the file ``myTrainedNN``.

# Visualization of BPN

TODO



# Basic customization

To Build a Neural Network requires many choices. For a neophyte, many of these
choices may seem arbitrary. The whole point of this project is to allow the
user to test different choices.

### New activation functions

The addition of a new activation function requires 3 steps:

 - In file __src/ActivationFunctions.h__ each activation function is
   implemented as a class that inherits from class ``ActivationFunction``.
   There are three functions to override : 
     - ``double evaluate( double x )`` which computed the activation function for an input ``x``.
     - ``double evalDerivative( double x, double fx )`` which computed the
       derivative of the function for an input ``x``. Note that a second
       parameter ``fx`` is specified, this parameter is the value of the activation
       function for input ``x`` (not used in general but it does speed up the
       computation in certain cases).
     - ``std::string serialize()`` returns a string that represents the function.

    __Copy__ one of the existing subclasses, __rename__ it and __update__ the three
    abovementioned functions.

 - In file __src/ActivationFunctions.cpp__ update function ``deserialize(const std::string& s)``. 
   This function is the inverse of ``serialize``, it parses a string in order
   return the corresponding activation function.

 - In file __src/trainBPN.cpp__ at the beginning of the ``main`` function,
   update the documentation string for optional parameter "``s``" consequently
   to what you have done.


### Modify weights initialization

When building a new neural networks, random weights are generated in function
``Network::InitializeWeights()`` found in file __src/NeuralNetwork.cpp__. 



