# NeuralNetwork
Simple Back Propagation Neural Network

This code is a fork from [Bobby Anguelov's NeuralNetwork](https://github.com/BobbyAnguelov/NeuralNetwork).

The example code makes use of [Florian Rappl's command parser](https://github.com/FlorianRappl/CmdParser )

# Disclaimer
This code is meant to be a simple implementation of the back-propagation neural network discussed in the tutorial below:

[https://takinginitiative.wordpress.com/2008/04/03/basic-neural-network-tutorial-theory/](https://takinginitiative.wordpress.com/2008/04/03/basic-neural-network-tutorial-theory/)

[https://takinginitiative.wordpress.com/2008/04/23/basic-neural-network-tutorial-c-implementation-and-source-code/](https://takinginitiative.wordpress.com/2008/04/23/basic-neural-network-tutorial-c-implementation-and-source-code/)

It is intended as a reference/example implementation and will not be maintained or supported.


# Compilation
```
mkdir build
cd build
cmake ..
make
```

Alternatively, since one might find useful to use debug/release versions:
```
mkdir debug && cd debug && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j
cd ..
mkdir release && cd release && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
```

# Typical usage
Let's train a Network using data from the ``detectShape`` file. In this
example, the input are binary pictures of size 30x30. Consequently, there needs
to be 900 input neurons on the first layer. The output is layer must have size
3 since the data is classified in three categories :

1. Straight line.
2. Rectangle.
3. Random points.

Here we choose to have 2 hidden layers of size 5 and 5, so the "-l" argument is
"900,5,5,3". Finally, we stop the learning once our network has obtain at least
99,4% accuracy on the genaralization set.

```
./trainBPN -d ../Example/detectShape -l 900,5,5,3 -a 99.4 -e myTrainedNN
Input file: ../Example/detectShape
Read complete: 50000 inputs loaded

 Neural Network Training Starting: 
==========================================================================
 LR: 0.01, Momentum: 0.9, Max Epochs: 100
 900 Input Neurons, 2 Hidden Layers, 3 Output Neurons
==========================================================================

Epoch :0 Training Set Accuracy:0%, MSE: 0.223235 Generalization Set Accuracy:0%, MSE: 0.222293
Epoch :1 Training Set Accuracy:6.70667%, MSE: 0.202158 Generalization Set Accuracy:35.29%, MSE: 0.149463
Epoch :2 Training Set Accuracy:60.71%, MSE: 0.0868884 Generalization Set Accuracy:94.25%, MSE: 0.017969
Epoch :3 Training Set Accuracy:98.27%, MSE: 0.00766275 Generalization Set Accuracy:98.97%, MSE: 0.00440707
Epoch :4 Training Set Accuracy:99.7133%, MSE: 0.00199997 Generalization Set Accuracy:99.34%, MSE: 0.00261908
Epoch :5 Training Set Accuracy:99.89%, MSE: 0.000918018 Generalization Set Accuracy:99.37%, MSE: 0.00224465
Epoch :6 Training Set Accuracy:99.9533%, MSE: 0.000539625 Generalization Set Accuracy:99.35%, MSE: 0.00212363
Epoch :7 Training Set Accuracy:99.9633%, MSE: 0.000404045 Generalization Set Accuracy:99.36%, MSE: 0.00207141
Epoch :8 Training Set Accuracy:99.9667%, MSE: 0.00031342 Generalization Set Accuracy:99.4%, MSE: 0.00202896

Training Complete!!! - > Elapsed Epochs: 9
 Validation Set Accuracy: 99.49
 Validation Set MSE: 0.00199625
```

Note that in the above example, the trained network is exported to the file ``myTrainedNN``.

To visualize the Network in action, use the ``gui`` program which reads the
export of a Neural Network on the standard input.
```
./gui < myTrainedNN
```
The grid on the left shows the state of the input neurons. Click on a square to
change it's value. The neurons layers are displayed to the right. Last column
is the ouput layer.

Here are some examples : 

 - A line ; the first output neuron is active.
![line detection](https://github.com/xprov/NeuralNetwork/blob/master/images/detectLine.png)

 - A rectangle ; the second output neuron is active.
![rectangle detection](https://github.com/xprov/NeuralNetwork/blob/master/images/detectRectangle.png)

 - Some random points ; the third output neuron is active.
![random detection](https://github.com/xprov/NeuralNetwork/blob/master/images/detectRandom.png)

Note : the little red square on the bottom resets the input grid.




